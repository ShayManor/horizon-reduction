"""Tests for the online on-robot path: replay buffer, boundary rebuild, Spot env, and metrics.

Everything runs on CPU with no robot and no SDK. `deploy.spot_client.SpotClient.__init__`,
`scale_action` and `unscale_velocity` import no `bosdyn`, so the fake below subclasses the real
client and stubs only the calls that reach the robot.
"""
import math
import os
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents._fk_wiring import make_fk_batch
from deploy.graphnav_map import CachedMap, _compose, _quat_to_yaw, open_map
from deploy.sim_client import SimSpotClient
from deploy.spot_client import SpotClient
from envs.spot_maze import SIM_HOSTNAME, SpotMazeEnv, load_tasks, make_spot_maze_env
from main import online_step_range
from utils.datasets import Dataset, HGCDataset, ReplayBuffer, get_size, load_spot_datasets
from utils.evaluation import evaluate

OBS_DIM, ACT_DIM = 7, 3


class FakeSpotClient(SpotClient):
    """Integrates commanded velocity in the plane. No SDK, no robot."""

    def __init__(self, start=(0.0, 0.0, 0.0), **kwargs):
        super().__init__('fake-robot', **kwargs)
        self.start = start
        self.pose = list(start)
        self.vel = [0.0, 0.0, 0.0]
        self.stops = 0
        self.releases = 0
        self.navigations = []

    def send_velocity(self, vx, vy, wz, duration=None):
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        dt = self.control_period
        self.pose = [x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + wz * dt]
        self.vel = [vx, vy, wz]

    def stop(self):
        self.stops += 1
        self.vel = [0.0, 0.0, 0.0]

    def release(self):
        self.releases += 1

    def get_state(self):
        return dict(
            x=self.pose[0],
            y=self.pose[1],
            yaw=self.pose[2],
            vx=self.vel[0],
            vy=self.vel[1],
            wz=self.vel[2],
            battery=100.0,
            localization_waypoint='wp-a',
            error='',
        )

    def navigate_to(self, waypoint_id, command_duration=5.0):
        self.navigations.append(waypoint_id)
        self.send_velocity(self.vel_limits[0], 0.0, 0.0)
        return 1

    def navigate_blocking(self, waypoint_id, timeout=180.0):
        self.navigations.append(waypoint_id)
        self.pose = list(self.start)
        self.vel = [0.0, 0.0, 0.0]
        return True


def make_env(max_episode_steps=20, goal_tol=0.5, goal=(1.0, 0.0)):
    client = FakeSpotClient()
    waypoint_xy = {'wp-a': (0.0, 0.0), 'wp-b': goal}
    tasks = [dict(task_name='task1', start='wp-a', goal='wp-b')]
    env = SpotMazeEnv(
        client,
        waypoint_xy,
        tasks,
        control_hz=10.0,
        max_episode_steps=max_episode_steps,
        goal_tol=goal_tol,
        sleep_fn=lambda _: None,
    )
    return env, client


# ---------- environment ----------------------------------------------------------------------


def test_observation_is_the_seven_dim_se2_layout():
    env, client = make_env()
    ob, info = env.reset(options=dict(task_id=1))

    assert ob.shape == (OBS_DIM,)
    assert ob.dtype == np.float32
    assert info['goal'].shape == (2,)

    client.pose = [1.5, -2.0, math.pi / 3]
    client.vel = [0.3, -0.1, 0.4]
    ob, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    # step() commands zero velocity, so the pose above is what the observation reports back.
    np.testing.assert_allclose(ob[:2], [1.5, -2.0], atol=1e-5)
    np.testing.assert_allclose(ob[2:4], [math.cos(math.pi / 3), math.sin(math.pi / 3)], atol=1e-5)


def test_heading_channel_is_continuous_across_the_pi_wrap():
    env, client = make_env()
    env.reset(options=dict(task_id=1))
    obs = []
    for yaw in (math.pi - 1e-3, -math.pi + 1e-3):
        client.pose = [0.0, 0.0, yaw]
        ob, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
        obs.append(ob)
    assert np.abs(obs[0][2:4] - obs[1][2:4]).max() < 1e-2


def test_action_scaling_uses_the_measured_limits_and_the_reverse_cap():
    client = FakeSpotClient(vel_limits=(0.6, 0.4, 0.8), reverse_limit=0.2)
    assert client.scale_action([1.0, 1.0, 1.0]) == pytest.approx((0.6, 0.4, 0.8))
    assert client.scale_action([-1.0, -1.0, -1.0]) == pytest.approx((-0.2, -0.4, -0.8))
    assert client.scale_action([5.0, 0.0, 0.0]) == pytest.approx((0.6, 0.0, 0.0))
    assert client.unscale_velocity(0.6, 0.4, 0.8) == pytest.approx((1.0, 1.0, 1.0))
    assert client.unscale_velocity(-0.2, 0.0, 0.0)[0] == pytest.approx(-1.0)


def test_episode_terminates_inside_the_goal_tolerance():
    env, _ = make_env(max_episode_steps=1000, goal_tol=0.5, goal=(1.0, 0.0))
    env.reset(options=dict(task_id=1))
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(200):
        _, reward, terminated, truncated, info = env.step(forward)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['success'] == 1.0
    assert reward == 0.0


def test_episode_truncates_at_max_episode_steps():
    env, client = make_env(max_episode_steps=5, goal=(100.0, 0.0))
    env.reset(options=dict(task_id=1))
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert truncated and not terminated
    assert info['success'] == 0.0
    assert info['episode_step'] == 5
    assert client.stops == 1


def test_reset_drives_back_to_the_start_waypoint():
    env, client = make_env()
    env.reset(options=dict(task_id=1))
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert client.pose[0] > 0.0
    env.reset(options=dict(task_id=1))
    assert client.navigations[-1] == 'wp-a'
    assert client.pose == [0.0, 0.0, 0.0]


def test_graphnav_step_records_the_achieved_velocity_as_the_action():
    env, client = make_env(goal=(100.0, 0.0))
    env.reset(options=dict(task_id=1))
    _, _, _, _, info = env.step_graphnav()
    assert client.navigations[-1] == 'wp-b'
    # navigate_to drives at the forward cap, which is action 1.0 on the forward axis.
    assert info['action_x'] == pytest.approx(1.0)
    assert info['action_y'] == pytest.approx(0.0)


# ---------- replay buffer and boundary rebuild -------------------------------------------------


def transition(ob, action, next_ob, terminal):
    ob = np.asarray(ob, dtype=np.float32)
    return dict(
        observations=ob,
        next_observations=np.asarray(next_ob, dtype=np.float32),
        actions=np.asarray(action, dtype=np.float32),
        oracle_reps=ob[:2],
        terminals=np.float32(terminal),
    )


def fill_buffer(buffer, episode_lengths, rng):
    for length in episode_lengths:
        for t in range(length):
            ob = rng.randn(OBS_DIM).astype(np.float32)
            buffer.add_transition(
                transition(ob, rng.uniform(-1, 1, ACT_DIM), ob, float(t == length - 1))
            )


def gc_config():
    return dict(
        discount=0.99,
        value_p_curgoal=0.2,
        value_p_trajgoal=0.5,
        value_p_randomgoal=0.3,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=0.5,
        actor_p_randomgoal=0.5,
        actor_geom_sample=True,
        gc_negative=True,
        subgoal_steps=5,
        value_subgoal_steps=None,
        actor_subgoal_steps=None,
    )


def test_rebuild_boundaries_tracks_appended_episodes():
    rng = np.random.RandomState(0)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 1000)
    fill_buffer(buffer, [10, 12], rng)

    dataset = HGCDataset(buffer, gc_config())
    np.testing.assert_array_equal(dataset.terminal_locs, [9, 21])

    fill_buffer(buffer, [8], rng)
    dataset.rebuild_boundaries()
    np.testing.assert_array_equal(dataset.terminal_locs, [9, 21, 29])
    np.testing.assert_array_equal(dataset.initial_locs, [0, 10, 22])
    assert dataset.size == 30


def test_rebuild_boundaries_excludes_the_episode_in_progress():
    rng = np.random.RandomState(1)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 1000)
    fill_buffer(buffer, [10], rng)
    dataset = HGCDataset(buffer, gc_config())

    # Six transitions of an unfinished episode: the buffer holds them, sampling must not see them.
    fill_buffer(buffer, [6], rng)
    buffer['terminals'][buffer.size - 1] = 0.0
    dataset.rebuild_boundaries()

    assert buffer.size == 16
    assert dataset.size == 10
    idxs = np.random.randint(dataset.size, size=64)
    batch = dataset.sample(64, idxs=idxs)
    assert batch['observations'].shape == (64, OBS_DIM)


def test_goals_are_the_xy_slice_via_oracle_reps():
    rng = np.random.RandomState(2)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 1000)
    fill_buffer(buffer, [20, 20], rng)
    dataset = HGCDataset(buffer, gc_config())

    batch = dataset.sample(32, idxs=np.random.randint(dataset.size, size=32))
    assert batch['high_actor_goals'].shape == (32, 2)
    assert batch['high_value_goals'].shape == (32, 2)
    # gc_negative=True: 0 at the goal, -1 elsewhere, so V is a negative step count.
    assert set(np.unique(batch['rewards'])) <= {0.0, -1.0}


def test_buffer_survives_a_save_load_round_trip(tmp_path):
    rng = np.random.RandomState(3)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 1000)
    fill_buffer(buffer, [7, 9], rng)

    path = str(tmp_path / 'buffer.npz')
    buffer.save(path)
    restored = ReplayBuffer.load(path, 1000)

    assert restored.size == buffer.size == 16
    assert restored.max_size == 1000
    np.testing.assert_array_equal(restored['observations'][:16], buffer['observations'][:16])
    np.testing.assert_array_equal(restored['terminals'][:16], buffer['terminals'][:16])

    # A restored buffer keeps growing where it left off.
    fill_buffer(restored, [4], rng)
    assert restored.size == 20


def test_spot_datasets_concatenate_every_saved_buffer(tmp_path):
    """Offline training reads a run's buffer shards the way it reads dataset shards."""
    rng = np.random.RandomState(5)
    for i, lengths in enumerate([[6, 4], [5, 5]]):
        buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 100)
        fill_buffer(buffer, lengths, rng)
        buffer.save(str(tmp_path / f'buffer_{i}.npz'))

    train, val = load_spot_datasets(sorted(str(p) for p in tmp_path.glob('*.npz')), val_fraction=0.25)

    assert get_size(train) + get_size(val) == 20
    assert set(train) == {'observations', 'next_observations', 'actions', 'oracle_reps', 'terminals'}


def test_spot_datasets_split_on_an_episode_boundary(tmp_path):
    """`GCDataset` asserts its last index is a terminal, so neither half may cut mid-episode."""
    rng = np.random.RandomState(6)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 100)
    fill_buffer(buffer, [6, 4, 5, 5], rng)
    buffer.save(str(tmp_path / 'buffer_0.npz'))

    # A proportional cut lands at index 14, which sits inside the third episode.
    train, val = load_spot_datasets([str(tmp_path / 'buffer_0.npz')], val_fraction=0.3)

    assert get_size(train) == 15 and get_size(val) == 5
    assert train['terminals'][-1] == 1.0
    assert val['terminals'][-1] == 1.0
    # The real requirement: both halves survive the boundary assert in `GCDataset.__post_init__`.
    HGCDataset(Dataset.create(**train), gc_config())
    HGCDataset(Dataset.create(**val), gc_config())


def test_spot_datasets_reject_a_run_with_one_episode(tmp_path):
    """One episode cannot be split without leaving a half that ends mid-trajectory."""
    rng = np.random.RandomState(7)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 100)
    fill_buffer(buffer, [6], rng)
    buffer.save(str(tmp_path / 'buffer_0.npz'))

    with pytest.raises(AssertionError, match='two complete episodes'):
        load_spot_datasets([str(tmp_path / 'buffer_0.npz')], val_fraction=0.3)


def test_spot_datasets_drop_superseded_checkpoints_of_one_run(tmp_path):
    """`ReplayBuffer.save` writes the whole filled prefix, so a run's checkpoints overlap."""
    rng = np.random.RandomState(8)
    buffer = ReplayBuffer.create(transition(np.zeros(OBS_DIM), np.zeros(ACT_DIM), np.zeros(OBS_DIM), 0.0), 100)
    fill_buffer(buffer, [6, 4], rng)
    buffer.save(str(tmp_path / 'buffer_10.npz'))
    fill_buffer(buffer, [5, 5], rng)
    buffer.save(str(tmp_path / 'buffer_20.npz'))

    train, val = load_spot_datasets(sorted(str(p) for p in tmp_path.glob('*.npz')), val_fraction=0.25)

    # The 20-step checkpoint contains the 10-step one. Concatenating both would give 30.
    assert get_size(train) + get_size(val) == 20
    np.testing.assert_array_equal(
        np.concatenate([train['observations'], val['observations']]), buffer['observations'][:20]
    )


# ---------- metrics --------------------------------------------------------------------------


class ConstantAgent:
    """Drives straight forward. Enough to make `evaluate` produce episodes."""

    def sample_actions(self, observations, goals=None, seed=None, temperature=0):
        return jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)


def test_evaluate_reports_steps_over_completed_episodes_only():
    env, _ = make_env(max_episode_steps=1000, goal_tol=0.5, goal=(1.0, 0.0))
    stats, trajs, _, episodes = evaluate(
        ConstantAgent(), env, task_id=1, num_eval_episodes=3, num_video_episodes=0
    )
    assert len(episodes) == 3
    assert all(ep['success'] == 1.0 for ep in episodes)
    assert stats['success'] == 1.0
    assert stats['steps'] == pytest.approx(np.mean([ep['steps'] for ep in episodes]))
    assert stats['steps'] == len(trajs[0]['action'])


def test_evaluate_leaves_steps_absent_when_nothing_completes():
    env, _ = make_env(max_episode_steps=4, goal_tol=0.5, goal=(100.0, 0.0))
    stats, _, _, episodes = evaluate(
        ConstantAgent(), env, task_id=1, num_eval_episodes=2, num_video_episodes=0
    )
    assert all(ep['success'] == 0.0 and ep['steps'] == 4 for ep in episodes)
    assert stats['success'] == 0.0
    assert np.isnan(stats['steps'])


# ---------- fk speed source --------------------------------------------------------------------


def test_fk_speed_source_constant_is_flat():
    batch = {
        'observations': jnp.arange(14, dtype=jnp.float32).reshape(2, 7),
        'high_value_goals': jnp.zeros((2, 2), dtype=jnp.float32),
    }
    fk_batch = make_fk_batch(batch)
    np.testing.assert_allclose(np.asarray(fk_batch['speed']), np.ones(2))


def test_fk_speed_source_observation_reads_the_measured_body_speed():
    obs = jnp.array(
        [[0, 0, 1, 0, 3.0, 4.0, 9.0], [0, 0, 1, 0, 0.0, 0.0, 9.0]], dtype=jnp.float32
    )
    batch = {'observations': obs, 'high_value_goals': jnp.zeros((2, 2), dtype=jnp.float32)}
    fk_batch = make_fk_batch(batch, 'observation')
    # Linear speed over dims 4:6. The yaw rate in dim 6 is not part of it.
    np.testing.assert_allclose(np.asarray(fk_batch['speed']), [5.0, 0.0], atol=1e-6)


class _StubFKAgent:
    """Minimal stand-in for the FK proxy: a non-constant V and an identity goal encoder."""

    def __init__(self, config):
        self.config = config

        class _Net:
            def select(self, name):
                if name == 'rep_value':
                    return lambda x, **_kw: x
                if name == 'value':
                    # V = -|xy|, so neighbouring states really do differ in value.
                    return lambda obs, goals, params=None: -jnp.linalg.norm(obs[..., :2], axis=-1)
                raise KeyError(name)

        self.network = _Net()


def _fk_loss_with(fk_kappa):
    from agents.fk_loss import stochastic_fk_loss

    config = dict(viscous_scale=0.01, num_walks=8)
    if fk_kappa is not None:
        config['fk_kappa'] = fk_kappa
    batch = {
        'observations': jnp.array(np.random.RandomState(0).randn(16, OBS_DIM), dtype=jnp.float32),
        'value_goals': jnp.zeros((16, 2), dtype=jnp.float32),
        'speed': jnp.ones(16, dtype=jnp.float32),
    }
    loss, _ = stochastic_fk_loss(_StubFKAgent(config), batch, None, jax.random.PRNGKey(0))
    return float(loss)


def test_fk_kappa_sets_the_slope_cap():
    """The cap is kappa/speed. Raising it far above the observed slope zeroes the penalty."""
    assert _fk_loss_with(0.01) > 0.0
    assert _fk_loss_with(1e6) == pytest.approx(0.0)


def test_fk_kappa_defaults_to_the_previous_hardcoded_value():
    """Runs that never set the flag must be unchanged."""
    assert _fk_loss_with(None) == pytest.approx(_fk_loss_with(0.1))


def test_fk_speed_source_rejects_an_unknown_value():
    batch = {
        'observations': jnp.zeros((2, 7), dtype=jnp.float32),
        'high_value_goals': jnp.zeros((2, 2), dtype=jnp.float32),
    }
    with pytest.raises(ValueError):
        make_fk_batch(batch, 'nonsense')


# ---------- graphnav map helpers ----------------------------------------------------------------


def test_quat_to_yaw_round_trips():
    for yaw in (-2.0, -0.3, 0.0, 1.1, 3.0):
        qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)
        assert _quat_to_yaw(qw, 0.0, 0.0, qz) == pytest.approx(yaw)


def test_compose_chains_planar_transforms():
    a = (1.0, 0.0, math.pi / 2)
    b = (2.0, 0.0, 0.0)
    x, y, yaw = _compose(a, b)
    assert (x, y) == pytest.approx((1.0, 2.0), abs=1e-9)
    assert yaw == pytest.approx(math.pi / 2)


# ---------- occupancy-grid stand-in -------------------------------------------------------------

MAP_PATH = 'spot_data'


def make_sim_client(**kwargs):
    graphnav_map = open_map(MAP_PATH)
    waypoint_xy = {wp_id: (pose[0], pose[1]) for wp_id, pose in graphnav_map.poses.items()}
    client = SimSpotClient(os.path.join(MAP_PATH, 'occupancy.npz'), waypoint_xy, **kwargs)
    return graphnav_map, client


def test_cached_map_resolves_waypoints_without_the_sdk():
    graphnav_map = CachedMap(MAP_PATH)
    assert len(graphnav_map.poses) == 28
    resolved = graphnav_map.resolve('waypoint_15')
    assert resolved in graphnav_map.poses
    assert graphnav_map.xy('waypoint_15') == graphnav_map.poses[resolved][:2]


def test_open_map_falls_back_to_the_cache_when_bosdyn_is_missing(monkeypatch):
    def no_sdk(map_path):
        raise ImportError('No module named bosdyn')

    monkeypatch.setattr('deploy.graphnav_map.GraphNavMap', no_sdk)
    assert isinstance(open_map(MAP_PATH), CachedMap)


def test_sim_client_refuses_a_step_into_an_obstacle():
    """Walking at a wall must not move the robot through it."""
    _, client = make_sim_client(control_hz=10.0)
    # A free cell whose eastward neighbours are obstacles, so driving east walks into them.
    rows, cols = np.nonzero(client.free[:, :-4] & ~client.free[:, 4:])
    row, col = rows[0], cols[0]
    x = client.lo[0] + (col + 0.5) * client.res
    y = client.lo[1] + (row + 0.5) * client.res
    wall_x = client.lo[0] + (col + 4) * client.res

    client.pose = [float(x), float(y), 0.0]  # yaw 0 points along +x
    assert client._is_free(*client.pose[:2])
    for _ in range(50):
        client.send_velocity(*client.scale_action(np.array([1.0, 0.0, 0.0])))
    assert client.pose[0] < wall_x
    assert client._is_free(*client.pose[:2])
    assert client.blocked


def test_sim_client_never_leaves_free_space_under_random_actions():
    _, client = make_sim_client(control_hz=10.0, seed=0)
    graphnav_map = open_map(MAP_PATH)
    client.navigate_blocking(graphnav_map.resolve('waypoint_15'))
    rng = np.random.RandomState(0)
    for _ in range(2000):
        client.send_velocity(*client.scale_action(rng.uniform(-1.0, 1.0, 3)))
        assert client._is_free(*client.pose[:2])


def test_sim_client_navigates_every_task_pair_both_ways():
    """The GraphNav stand-in has to reach the goal, or seed episodes collect nothing.

    Regression for corner cutting: a descent step to a diagonal free cell whose orthogonal
    neighbours are obstacles wedges the robot against the corner for the rest of the episode.
    """
    graphnav_map, _ = make_sim_client()
    tasks = load_tasks(MAP_PATH, graphnav_map)
    for task in tasks:
        for start, goal in ((task['start'], task['goal']), (task['goal'], task['start'])):
            _, client = make_sim_client(control_hz=10.0, seed=0)
            client.navigate_blocking(start)
            for _ in range(600):
                client.navigate_to(goal)
                if client.navigation_reached():
                    break
            assert client.navigation_reached(), f'stuck on {task["task_name"]} toward {goal}'


def test_sim_env_completes_a_graphnav_episode():
    env = make_spot_maze_env(
        'spot-maze-v0', SIM_HOSTNAME, MAP_PATH, control_hz=10.0, max_episode_steps=600, seed=0
    )
    ob, info = env.reset(options=dict(task_id=1))
    assert ob.shape == (OBS_DIM,)
    assert info['goal'].shape == (2,)

    terminated = False
    steps = 0
    while not terminated:
        ob, _, terminated, truncated, info = env.step_graphnav()
        steps += 1
        assert not truncated, 'GraphNav timed out on task 1'
    assert info['distance_to_goal'] <= env.goal_tol
    assert 0 < steps < 600


def test_sim_env_needs_no_wall_clock_per_step():
    """The stand-in drops the control-period wait, or an online run takes its length in real time."""
    env = make_spot_maze_env('spot-maze-v0', SIM_HOSTNAME, MAP_PATH, control_hz=10.0, seed=0)
    env.reset(options=dict(task_id=1))
    started = time.time()
    for _ in range(100):
        env.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert time.time() - started < 100 / env.control_hz


# ---------- offline-to-online -------------------------------------------------------------------


def test_online_step_range_counts_steps_taken_online():
    """A seeded buffer must not eat the online budget."""
    assert len(online_step_range(env_step=142489, seeded_steps=142489, online_steps=10000)) == 10000
    assert online_step_range(142489, 142489, 10000)[0] == 142490


def test_online_step_range_resumes_to_a_total_when_nothing_was_seeded():
    """Restoring this run's own buffer is a resume, so the budget is the original total."""
    assert len(online_step_range(env_step=400, seeded_steps=0, online_steps=1000)) == 600


def test_online_step_range_rejects_a_spent_budget():
    with pytest.raises(AssertionError, match='already spent'):
        online_step_range(env_step=2000, seeded_steps=0, online_steps=1000)


def test_offline_dataset_seeds_a_buffer_that_online_steps_can_extend(tmp_path):
    """The warm start reuses the offline data, and the buffer still takes new transitions."""
    steps = 40
    data = dict(
        observations=np.arange(steps * OBS_DIM, dtype=np.float32).reshape(steps, OBS_DIM),
        next_observations=np.zeros((steps, OBS_DIM), dtype=np.float32),
        actions=np.zeros((steps, ACT_DIM), dtype=np.float32),
        oracle_reps=np.zeros((steps, 2), dtype=np.float32),
        terminals=np.zeros(steps, dtype=np.float32),
    )
    data['terminals'][[9, 19, 29, 39]] = 1.0
    np.savez(tmp_path / 'buffer_40.npz', **data)

    offline, _ = load_spot_datasets([str(tmp_path / 'buffer_40.npz')])
    buffer = ReplayBuffer.create_from_initial_dataset(offline, 100)
    seeded = buffer.size
    assert seeded == get_size(offline)
    assert buffer['terminals'][seeded - 1] == 1.0, 'the seeded buffer must end on an episode'

    buffer.add_transition(
        dict(
            observations=np.ones(OBS_DIM, dtype=np.float32),
            next_observations=np.ones(OBS_DIM, dtype=np.float32),
            actions=np.zeros(ACT_DIM, dtype=np.float32),
            oracle_reps=np.ones(2, dtype=np.float32),
            terminals=np.float32(0.0),
        )
    )
    assert buffer.size == seeded + 1
    assert np.array_equal(buffer['observations'][:seeded], offline['observations'])
