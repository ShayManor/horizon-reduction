"""Tests for the online on-robot path: replay buffer, boundary rebuild, Spot env, and metrics.

Everything runs on CPU with no robot and no SDK. `deploy.spot_client.SpotClient.__init__`,
`scale_action` and `unscale_velocity` import no `bosdyn`, so the fake below subclasses the real
client and stubs only the calls that reach the robot.
"""
import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax.numpy as jnp
import numpy as np
import pytest

from agents._fk_wiring import make_fk_batch
from deploy.graphnav_map import _compose, _quat_to_yaw
from deploy.spot_client import SpotClient
from envs.spot_maze import SpotMazeEnv
from utils.datasets import HGCDataset, ReplayBuffer
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
