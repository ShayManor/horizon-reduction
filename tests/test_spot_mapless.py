"""Tests for the mapless online path. Everything runs on CPU with no robot and no SDK."""
import json
import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pytest

AGENT_FILES = ['agents/sharsa.py', 'agents/sharsa_dual.py', 'agents/sharsa_geodesic.py']


def test_time_agent_reports_every_field():
    from deploy.time_jetson import time_agent

    r = time_agent('agents/sharsa.py', batch_size=8, n=2)
    assert set(r) >= {'agent', 'batch_size', 'update_ms', 'update_p95_ms', 'act_ms', 'act_p95_ms', 'device', 'update_gflop'}
    assert r['update_ms'] > 0 and r['act_ms'] > 0 and r['update_gflop'] > 0


@pytest.mark.parametrize('agent_file', AGENT_FILES)
def test_build_agent_updates_for_each_arm(agent_file):
    from deploy.time_jetson import build_agent, synthetic_batch

    agent, config = build_agent(agent_file, 8)
    assert config['gc_negative'] and config['value_loss_type'] == 'squared' and config['subgoal_steps'] == 10
    agent, info = agent.update(synthetic_batch(8))
    assert all(v == v for v in info.values())


def test_build_agent_phys_overrides():
    from deploy.time_jetson import build_agent

    _, config = build_agent('agents/sharsa.py', 8, {'w_fk': 1.0, 'viscous_scale': 0.01, 'num_walks': 10, 'fk_kappa': 15.0})
    assert config['w_fk'] == 1.0 and config['fk_kappa'] == 15.0


def test_run_frame_round_trips_and_zeroes_the_launch_pose():
    from envs.spot_mapless import RunFrame

    f = RunFrame((2.0, 1.0, math.pi / 2))
    assert np.allclose(f.to_run(2.0, 1.0, math.pi / 2), (0, 0, 0), atol=1e-9)
    assert np.allclose(f.to_run(2.0, 2.0, math.pi / 2)[:2], (1.0, 0.0), atol=1e-9)
    assert np.allclose(f.from_run(*f.to_run(3.3, -1.2, 0.4)), (3.3, -1.2, 0.4), atol=1e-9)
    assert np.allclose(RunFrame.from_state({'x': 2.0, 'y': 1.0, 'yaw': 0.3}).to_run(2.0, 1.0, 0.3), (0, 0, 0))


def test_frontier_is_unvisited_neighbours_at_least_min_dist_away():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5))
    for x in np.arange(0.25, 3.0, 0.5):
        s.observe(x, 0.25)
    f = s.frontier(robot_xy=(2.75, 0.25))
    assert len(f) > 0
    assert np.all(np.hypot(f[:, 0] - 2.75, f[:, 1] - 0.25) >= 1.5)


def test_value_rule_picks_the_highest_value_frontier_cell():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5), p_revisit=0.0)
    for x in np.arange(0.25, 3.0, 0.5):
        s.observe(x, 0.25)
    g = s.sample((0.25, 0.25), value_fn=lambda goals: -goals[:, 1])  # prefers smallest y
    assert g[1] == s.frontier((0.25, 0.25))[:, 1].min()


def test_cell_fails_only_after_two_stuck_and_blocked_episodes_outside_warmup():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5))
    g = np.array([4.25, 4.25])
    s.report(g, 'stuck', blocked=True, warmup=True)
    s.report(g, 'stuck', blocked=False, warmup=False)
    s.report(g, 'stuck', blocked=True, warmup=False)
    assert s.stats()['failed_cells'] == 0
    s.report(g, 'stuck', blocked=True, warmup=False)
    assert s.stats()['failed_cells'] == 1


def test_pinned_robot_does_not_blacklist_distant_goals():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5))
    for _ in range(2):
        s.report(np.array([3.25, 0.25]), 'stuck', True, False, end_xy=(0.25, 0.25))
    assert s.stats()['failed_cells'] == 0


def test_blame_applies_near_the_goal():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5))
    for _ in range(2):
        s.report(np.array([3.25, 0.25]), 'stuck', True, False, end_xy=(2.65, 0.25))
    assert s.stats()['failed_cells'] == 1


def test_fallback_never_returns_a_goal_near_the_robot():
    from envs.spot_mapless import GoalSampler

    robot = (0.25, 0.25)
    for p_revisit in (0.0, 0.5, 1.0):
        s = GoalSampler((0, 0, 4, 4), p_revisit=p_revisit, fail_after=1, seed=1)
        s.observe(*robot)
        keep = {(3, 3), (7, 7), (7, 0)}
        for ix in range(s.nx):
            for iy in range(s.ny):
                if (ix, iy) not in keep:
                    s.failed[ix, iy] = True
        for _ in range(1000):
            g = s.sample(robot, lambda goals: np.zeros(len(goals)))
            assert np.hypot(g[0] - robot[0], g[1] - robot[1]) >= s.min_goal_dist
    # every cell failed or too close: the farthest non-failed cell, never the robot's own
    s = GoalSampler((0, 0, 2, 2), p_revisit=0.0, min_goal_dist=3.0)
    g = s.sample((0.25, 0.25), lambda goals: np.zeros(len(goals)))
    assert np.allclose(g, (1.75, 1.75))


def test_sampler_falls_back_to_random_in_bounds():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 2, 2), p_revisit=0.0)
    g = s.sample((1.0, 1.0), value_fn=lambda goals: np.zeros(len(goals)))
    assert 0 <= g[0] <= 2 and 0 <= g[1] <= 2


def test_sampler_state_round_trips():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 5, 5))
    s.observe(1.0, 1.0)
    s.report(np.array([4.25, 4.25]), 'stuck', True, False)
    s.report(np.array([4.25, 4.25]), 'stuck', True, False)
    t = GoalSampler((0, 0, 5, 5))
    t.load_state_dict(json.loads(json.dumps(s.state_dict())))
    assert t.stats() == s.stats() and t.coverage() == s.coverage()


def test_failed_cells_are_never_sampled_and_nearest_rule_picks_closest():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 2, 2), p_revisit=0.0, fail_after=1)
    for gx in (0.25, 0.75, 1.25, 1.75):
        for gy in (0.25, 0.75, 1.25, 1.75):
            if (gx, gy) != (1.75, 1.75):
                s.report(np.array([gx, gy]), 'stuck', True, False)
    assert np.allclose(s.sample((0.25, 0.25), lambda g: np.zeros(len(g))), (1.75, 1.75))
    n = GoalSampler((0, 0, 5, 5), p_revisit=0.0, rule='nearest')
    for x in np.arange(0.25, 3.0, 0.5):
        n.observe(x, 0.25)
    f = n.frontier((0.25, 0.25))
    d = np.hypot(f[:, 0] - 0.25, f[:, 1] - 0.25)
    g = n.sample((0.25, 0.25), None)
    assert np.isclose(np.hypot(g[0] - 0.25, g[1] - 0.25), d.min())


def test_goals_stay_inside_bounds_when_extent_is_not_a_cell_multiple():
    from envs.spot_mapless import GoalSampler

    for p_revisit in (0.0, 1.0):
        s = GoalSampler((0, 0, 1.2, 1.2), p_revisit=p_revisit, min_goal_dist=0.0)
        for x in (0.25, 0.75, 1.1):
            s.observe(x, 0.25)
        f = s.frontier((1.2, 1.2))
        assert len(f) > 0 and np.all((f >= 0) & (f <= 1.2))
        for _ in range(50):
            g = s.sample((1.2, 1.2), lambda goals: np.zeros(len(goals)))
            assert np.all((g >= 0) & (g <= 1.2))
    r = GoalSampler((0, 0, 1.2, 1.2), p_revisit=0.0)
    assert all(np.all((r.sample((1.2, 1.2), None) >= 0) & (r.sample((1.2, 1.2), None) <= 1.2)) for _ in range(50))


def test_frontier_uses_the_8_neighbourhood():
    from envs.spot_mapless import GoalSampler

    s = GoalSampler((0, 0, 2.5, 2.5), min_goal_dist=0.0)
    s.observe(1.25, 1.25)
    assert len(s.frontier((0.0, 0.0))) == 8


# ---------- mapless env ----------------------------------------------------------------------


def _mapless_env(start=(3.0, 2.0, 1.0), **kwargs):
    from envs.spot_mapless import MaplessSpotEnv
    from tests.test_spot_online import FakeSpotClient

    c = FakeSpotClient(start=start)
    return MaplessSpotEnv(c, sleep_fn=lambda _: None, **kwargs), c


def test_mapless_reset_does_not_navigate_and_starts_at_origin():
    env, c = _mapless_env()
    ob, info = env.reset(options={'goal': np.array([1.0, 0.0])})
    assert c.navigations == [] and np.allclose(ob[:2], 0.0, atol=1e-6)
    assert env.task_infos == [] and info['outcome'] == 'running'


def test_second_reset_keeps_the_frame_and_continues_from_current_pose():
    env, c = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})
    for _ in range(5):
        ob, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert info['x'] > 0.2 and info['step_metres'] > 0
    ob2, info2 = env.reset(options={'goal': np.array([0.0, 3.0])})
    assert np.allclose(ob2[:2], ob[:2], atol=1e-6) and c.navigations == []
    assert np.allclose(info2['goal'], (0.0, 3.0))


def test_stuck_ends_episode_after_window_without_progress():
    env, _ = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})
    for _ in range(29):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert not (terminated or truncated) and info['outcome'] == 'running'
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not terminated and info['outcome'] == 'stuck' and info['episode_step'] == 30


def test_reached_terminates_inside_goal_tol():
    env, _ = _mapless_env()
    env.reset(options={'goal': np.array([1.0, 0.0])})
    for _ in range(50):
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0]))
        if terminated or truncated:
            break
    assert terminated and not truncated and info['outcome'] == 'reached' and reward == 0.0
    assert info['distance_to_goal'] <= 0.5


def test_timeout_truncates_at_max_episode_steps():
    env, _ = _mapless_env(max_episode_steps=10)
    env.reset(options={'goal': np.array([50.0, 0.0])})
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert truncated and not terminated and info['outcome'] == 'timeout'


def test_pose_jump_ends_episode_as_intervention():
    env, c = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})
    env.step(np.zeros(3))
    c.pose[0] += 2.0
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not terminated and info['outcome'] == 'intervention' and info['step_metres'] == 0.0


def test_lease_loss_ends_episode_as_intervention_and_other_errors_propagate():
    env, c = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})

    class LeaseUseError(Exception):
        pass

    def lose_lease(*a, **k):
        raise LeaseUseError('lease taken by the tablet')

    c.send_velocity = lose_lease
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not terminated and info['outcome'] == 'intervention'

    def other(*a, **k):
        raise ValueError('not an intervention')

    c.send_velocity = other
    with pytest.raises(ValueError):
        env.step(np.zeros(3))


def test_power_cut_by_estop_ends_episode_as_intervention():
    env, c = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})

    class NotPoweredOnError(Exception):
        pass

    def power_cut(*a, **k):
        raise NotPoweredOnError('motors off after hardware e-stop')

    c.send_velocity = power_cut
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not terminated and info['outcome'] == 'intervention'


def test_blocked_proxy_thresholds():
    from envs.spot_mapless import is_blocked

    assert is_blocked((0.3, 0.0), (0.01, 0.0)) and not is_blocked((0.1, 0.0), (0.0, 0.0))
    assert not is_blocked((0.3, 0.0), (0.2, 0.0))


def test_mapless_sim_reads_no_map_but_the_world_grid(monkeypatch):
    import deploy.graphnav_map as gm
    import envs.spot_maze as sm
    from envs.spot_mapless import make_spot_mapless_env

    monkeypatch.setattr(gm, 'open_map', lambda *a, **k: pytest.fail('opened a GraphNav map'))
    monkeypatch.setattr(sm, 'load_tasks', lambda *a, **k: pytest.fail('read tasks.json'))
    monkeypatch.chdir('/')
    env = make_spot_mapless_env('sim', start_pose=(1.0, 1.0, 0.0), control_hz=10.0, max_episode_steps=50,
                                goal_tol=0.5, vel_limits=(0.6, 0.4, 0.8), reverse_limit=None, seed=0)
    ob, info = env.reset(options={'goal': np.array([1.0, 0.0])})
    assert np.allclose(ob[:2], 0.0) and info['localization_waypoint'] == ''
    _, _, _, _, info = env.step(np.zeros(3))
    assert info['blocked'] is False and np.isclose(env.client.pose[0], 1.0)


def test_vision_pose_source_reads_odometry_not_graphnav(monkeypatch):
    import sys
    import types

    from deploy.spot_client import SpotClient

    ns = types.SimpleNamespace
    pose = ns(x=1.5, y=-0.5, rot=ns(to_yaw=lambda: 0.25))
    frame_helpers = types.ModuleType('bosdyn.client.frame_helpers')
    frame_helpers.get_vision_tform_body = lambda snapshot: pose
    math_helpers = types.ModuleType('bosdyn.client.math_helpers')
    math_helpers.Quat = math_helpers.SE3Pose = None
    monkeypatch.setitem(sys.modules, 'bosdyn', types.ModuleType('bosdyn'))
    monkeypatch.setitem(sys.modules, 'bosdyn.client', types.ModuleType('bosdyn.client'))
    monkeypatch.setitem(sys.modules, 'bosdyn.client.frame_helpers', frame_helpers)
    monkeypatch.setitem(sys.modules, 'bosdyn.client.math_helpers', math_helpers)

    vel = ns(linear=ns(x=0.1, y=0.0, z=0.0), angular=ns(z=0.2))
    kinematic = ns(transforms_snapshot=ns(child_to_parent_edge_map={}), velocity_of_body_in_odom=vel)
    robot_state = ns(kinematic_state=kinematic, battery_states=[ns(charge_percentage=ns(value=80.0))])
    c = SpotClient('robot', pose_source='vision')
    c.state_client = ns(get_robot_state=lambda: robot_state)
    c.graph_nav_client = ns(get_localization_state=lambda: pytest.fail('called GraphNav'))
    s = c.get_state()
    assert (s['x'], s['y'], s['yaw'], s['vx'], s['wz']) == (1.5, -0.5, 0.25, 0.1, 0.2)
    assert s['localization_waypoint'] == '' and s['localization_age'] == 0.0 and s['battery'] == 80.0


# ---------- batched V(s, g) and the --mapless training branch --------------------------------

PHYS = {'w_fk': 1.0, 'viscous_scale': 0.01, 'num_walks': 10, 'fk_kappa': 15.0}


@pytest.mark.parametrize('agent_file,overrides', [(f, None) for f in AGENT_FILES] + [('agents/sharsa.py', PHYS)])
def test_goal_values_shape_and_finite(agent_file, overrides):
    from deploy.time_jetson import build_agent
    from utils.value_query import goal_values

    agent, _ = build_agent(agent_file, 8, overrides)
    ob = np.zeros(7, dtype=np.float32)
    goals = np.random.default_rng(0).uniform(0, 5, size=(5, 2)).astype(np.float32)
    v = goal_values(agent, ob, goals)
    assert v.shape == (5,) and np.all(np.isfinite(v))
    # padding to a power of two must not change the values of the real goals
    assert np.allclose(goal_values(agent, ob, goals[:3]), v[:3], atol=1e-5)


def test_goal_values_matches_high_value_and_dual_goes_through_rep_value():
    from deploy.time_jetson import build_agent
    from utils.value_query import goal_values

    ob = np.zeros(7, dtype=np.float32)
    goals = np.array([[1.0, 2.0], [3.0, 0.5]], dtype=np.float32)
    obs = np.repeat(ob[None], 2, 0)
    agent, _ = build_agent('agents/sharsa.py', 8)
    assert np.allclose(goal_values(agent, ob, goals), agent.network.select('high_value')(obs, goals), atol=1e-5)
    dual, _ = build_agent('agents/sharsa_dual.py', 8)
    reps = dual.network.select('rep_value')(goals)
    assert np.allclose(goal_values(dual, ob, goals), dual.network.select('high_value')(obs, reps), atol=1e-5)


def test_pose_jump_is_flagged_in_info():
    env, c = _mapless_env()
    env.reset(options={'goal': np.array([5.0, 0.0])})
    _, _, _, _, info = env.step(np.zeros(3))
    assert info['jump'] is False
    c.pose[0] += 2.0
    _, _, _, _, info = env.step(np.zeros(3))
    assert info['jump'] is True and info['outcome'] == 'intervention'


def test_wait_for_control_acquires_only_once_the_operator_creates_the_resume_file(tmp_path):
    from envs.spot_mapless import wait_for_control

    class KeepAlive:
        shutdowns = 0

        def shutdown(self):
            KeepAlive.shutdowns += 1

    class Client:
        acquires = 0
        _lease_keepalive, _estop_keepalive = KeepAlive(), KeepAlive()

        def get_state(self):  # the robot answers the whole time: that alone is not consent
            return {}

        def acquire(self):
            self.acquires += 1

    c, lines, resume = Client(), [], tmp_path / 'RESUME'
    polls = []

    def sleep(dt):
        polls.append(dt)
        assert c.acquires == 0  # never while the file is absent
        if len(polls) == 25:
            resume.touch()

    wait_for_control(c, str(resume), sleep_fn=sleep, print_fn=lines.append)
    assert polls == [1.0] * 25 and c.acquires == 1 and not resume.exists()
    assert KeepAlive.shutdowns == 2 and c._lease_keepalive is None and c._estop_keepalive is None
    assert len(lines) == 3  # one line per 10 s of waiting


def _main(tmp_path, *extra, check=True):
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, 'main.py', '--mapless', '--robot_hostname=sim', '--arena_bounds=0,0,8,6',
        '--warmup_steps=100', '--save_interval=100', '--log_interval=50', '--agent.batch_size=16',
        '--agent.gc_negative=True', '--agent.value_loss_type=squared', '--agent.subgoal_steps=10',
        f'--save_dir={tmp_path}', *extra,
    ]
    # `setup_wandb` passes mode='online' explicitly, which beats WANDB_MODE. With no API key and a HOME
    # without ~/.netrc it falls back to offline, so these runs never reach the wandb server.
    env = {k: v for k, v in os.environ.items() if k != 'WANDB_API_KEY'}
    env.update(WANDB_MODE='disabled', JAX_PLATFORMS='cpu', HOME=str(tmp_path))
    r = subprocess.run(cmd, check=False, cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    if check:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def _run_dir(tmp_path, step):
    import glob

    runs = sorted(os.path.dirname(p) for p in glob.glob(f'{tmp_path}/**/params_{step}.pkl', recursive=True))
    assert len(runs) == 1, runs
    return runs[0]


ONLINE_METRIC_KEYS = [
    'progress/env_steps', 'progress/wall_time', 'progress/metres', 'progress/updates',
    'time/update_ms', 'time/update_p95_ms', 'time/act_ms', 'time/act_p95_ms', 'time/overrun_frac',
    'safety/blocked_steps', 'safety/interventions',
    'explore/coverage', 'explore/frontier_size', 'explore/failed_cells',
    'episode/reached', 'episode/timeout', 'episode/stuck',
]  # fmt: skip


def test_online_metrics_summary():
    from utils.online_metrics import OnlineMetrics

    m = OnlineMetrics(control_period=0.1)
    m.on_step(dict(dt=0.1, step_metres=0.05, blocked=False, outcome='running'), act_ms=3.0)
    m.on_step(dict(dt=0.2, step_metres=0.05, blocked=True, outcome='stuck'), act_ms=5.0)
    m.on_update(20.0)
    m.on_episode('stuck')
    s = m.summary(dict(frontier_size=4, failed_cells=0, visited_cells=3), coverage=0.25)
    assert set(s) == set(ONLINE_METRIC_KEYS)
    assert s['progress/env_steps'] == 2 and s['progress/metres'] == pytest.approx(0.1)
    assert s['time/overrun_frac'] == 0.5 and s['safety/blocked_steps'] == 1 and s['episode/stuck'] == 1
    assert s['progress/updates'] == 1 and s['time/act_ms'] == pytest.approx(4.0)
    s = m.summary(dict(frontier_size=4, failed_cells=0, visited_cells=3), coverage=0.25)
    assert math.isnan(s['time/act_ms']) and s['progress/env_steps'] == 2  # windows reset, counters do not


def test_online_metrics_untimed_compile_call_counts_but_is_not_averaged():
    from utils.online_metrics import OnlineMetrics

    m = OnlineMetrics(control_period=0.1)
    m.on_step(dict(dt=0.1, step_metres=0.0, blocked=False, outcome='running'), act_ms=None)
    m.on_update(None)
    s = m.summary(dict(frontier_size=0, failed_cells=0, visited_cells=0), coverage=0.0)
    assert s['progress/updates'] == 1 and s['progress/env_steps'] == 1
    assert math.isnan(s['time/act_ms']) and math.isnan(s['time/update_ms'])


def test_mapless_train_online_runs_in_sim(tmp_path):
    import csv

    _main(tmp_path, '--start_pose=1,1,0', '--online_steps=300')
    run = _run_dir(tmp_path, 300)
    for name in ('buffer_300.npz', 'sampler_300.json', 'sampler_200.json', 'session_log.csv'):
        assert os.path.exists(os.path.join(run, name)), name
    with open(os.path.join(run, 'episodes.csv')) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0 and {'goal_x', 'goal_y', 'outcome', 'metres', 'success'} <= set(rows[0])
    assert 'goal_waypoint' not in rows[0] and {r['task_name'] for r in rows} == {'mapless'}
    with open(os.path.join(run, 'session_log.csv')) as f:
        session = list(csv.DictReader(f))
    assert [s['event'] for s in session] == ['start', 'end'] and session[-1]['drift_x'] != ''
    with open(os.path.join(run, 'train.csv')) as f:
        header = next(csv.DictReader(f))
    assert 'evaluation/overall_success' in header and set(ONLINE_METRIC_KEYS) <= set(header)
    assert os.path.exists(os.path.join(run, 'warmup.csv'))


def test_resume_restores_sampler(tmp_path):
    _main(tmp_path, '--start_pose=1,1,0', '--online_steps=300')
    first = _run_dir(tmp_path, 300)
    r = _main(tmp_path, '--start_pose=1,1,0', '--online_steps=400', f'--restore_path={first}', '--restore_epoch=300')
    assert f'Restored sampler from {os.path.join(first, "sampler_300.json")}' in r.stdout
    second = _run_dir(tmp_path, 400)
    with open(os.path.join(first, 'sampler_300.json')) as f:
        before = np.array(json.load(f)['visited'])
    with open(os.path.join(second, 'sampler_400.json')) as f:
        after = np.array(json.load(f)['visited'])
    assert before.any() and np.all(after[before])


def test_mapless_sim_without_start_pose_or_mark_fails_clearly(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(repo, 'spot_data', 'start_mark.json')):
        pytest.skip('start_mark.json exists, so the sim has a start pose')
    r = _main(tmp_path, '--online_steps=10', check=False)
    assert r.returncode != 0 and 'start_mark.json' in r.stdout + r.stderr


def test_pose_jump_aborts_the_session_after_saving_and_resumes_through_warmup(tmp_path, monkeypatch, capsys):
    import csv

    import main
    from envs.spot_mapless import MaplessSpotEnv

    step, calls = MaplessSpotEnv.step, []

    def carried_at_step_20(self, action):
        calls.append(1)
        if len(calls) == 20:
            self.client.pose[0] += 2.0
        return step(self, action)

    def run(save_dir, *extra):
        os.makedirs(save_dir, exist_ok=True)
        main.FLAGS(['main.py', '--mapless', '--robot_hostname=sim', '--arena_bounds=0,0,8,6', '--start_pose=1,1,0',
                    '--warmup_steps=100', f'--save_dir={save_dir}', *extra])
        # `--agent.*` overrides are not registered on an in-process parse, so set them on the config.
        main.FLAGS.agent.update(batch_size=16, gc_negative=True, value_loss_type='squared', subgoal_steps=10)
        try:
            main.train_online(main.FLAGS.agent)
        finally:
            main.FLAGS.unparse_flags()

    monkeypatch.setattr(MaplessSpotEnv, 'step', carried_at_step_20)
    with pytest.raises(SystemExit) as exc:
        run(tmp_path, '--online_steps=300')
    assert exc.value.code == 3 and 'Pose jump at env step 20' in capsys.readouterr().out
    assert os.path.exists(tmp_path / 'sampler_20.json') and not os.path.exists(tmp_path / 'params_20.pkl')
    with np.load(tmp_path / 'buffer_20.npz') as f:
        obs, next_obs, terminals = f['observations'], f['next_observations'], f['terminals']
    assert len(obs) == 19 and terminals[-1] == 1.0  # the jumped step is not stored
    assert np.all(np.hypot(*(next_obs[:, :2] - obs[:, :2]).T) <= 1.0)
    with open(tmp_path / 'session_log.csv') as f:
        assert [r['event'] for r in csv.DictReader(f)] == ['start', 'end']

    # Resume from the jump: warmup finishes from the restored buffer, then training runs to the budget.
    run(tmp_path / 'resume', '--online_steps=150', f'--restore_path={tmp_path}', '--restore_epoch=20')
    assert os.path.exists(tmp_path / 'resume' / 'params_150.pkl')
    with np.load(tmp_path / 'resume' / 'buffer_150.npz') as f:
        assert len(f['observations']) == 150


def _world(tmp_path, gap=None, res=0.1):
    """40x40 grid, a wall at column 20; `gap` is a (row_lo, row_hi) opening, None is a full wall."""
    from deploy.sim_client import SimSpotClient

    free = np.ones((40, 40), dtype=bool)
    free[:, 20] = False
    if gap is not None:
        free[gap[0] : gap[1], 20] = True
    np.savez(tmp_path / 'occupancy.npz', free=free, lo=np.zeros(2), res=res)
    return SimSpotClient(str(tmp_path / 'occupancy.npz'), {})


def test_distance_field_refactor_matches_waypoint_version(tmp_path):
    client = _world(tmp_path, gap=(0, 4))
    client.waypoint_xy = {'a': (0.55, 3.05)}
    assert np.array_equal(client._distance_field('a'), client.distance_field_xy(0.55, 3.05))


def test_geodesic_detours_around_wall(tmp_path):
    from deploy.eval_pairs import geodesic

    client = _world(tmp_path, gap=(0, 4))
    start, goal = (1.05, 3.05), (2.95, 3.05)
    assert geodesic(client, start, goal) > math.dist(start, goal) + 1.0
    assert geodesic(client, start, (1.05, 1.05)) == pytest.approx(2.0, abs=0.01)


def test_fixed_pairs_respect_clearance_and_min_geodesic(tmp_path):
    from scipy.ndimage import distance_transform_edt

    from deploy.eval_pairs import geodesic, make_fixed_pairs

    client = _world(tmp_path, gap=(0, 4))
    pairs = make_fixed_pairs(client, (0, 0, 4, 4), n=20, clearance=0.3, min_geo=1.5, seed=1)
    assert pairs.shape == (20, 4)
    edt = distance_transform_edt(client.free) * client.res
    for sx, sy, gx, gy in pairs:
        for x, y in ((sx, sy), (gx, gy)):
            assert edt[client._cell(x, y)] >= 0.3
        assert geodesic(client, (sx, sy), (gx, gy)) >= 1.5
    assert np.array_equal(pairs, make_fixed_pairs(client, (0, 0, 4, 4), n=20, clearance=0.3, min_geo=1.5, seed=1))


def test_fixed_pairs_exclude_disconnected(tmp_path):
    from deploy.eval_pairs import make_fixed_pairs

    client = _world(tmp_path)
    pairs = make_fixed_pairs(client, (0, 0, 4, 4), n=30, clearance=0.3, min_geo=1.5, seed=0)
    assert np.all((pairs[:, 0] < 2.0) == (pairs[:, 2] < 2.0))
    with pytest.raises(RuntimeError):
        make_fixed_pairs(client, (0, 0, 4, 4), n=30, clearance=0.3, min_geo=50.0, seed=0)


def test_explored_pairs_only_use_visited_cells(tmp_path):
    from deploy.eval_pairs import explored_pairs
    from envs.spot_mapless import RunFrame

    client = _world(tmp_path, gap=(0, 4))
    frame = RunFrame((1.0, 1.0, math.pi / 2))
    seed_xy = np.array([[1.05, 1.05], [1.05, 2.05], [1.05, 3.05], [3.05, 1.05], [3.05, 3.05]])
    # Buffer observations are run-frame, so convert the seed-frame cells we want visited.
    obs = np.array([frame.to_run(x, y, 0.0)[:2] for x, y in seed_xy])
    np.savez(tmp_path / 'buffer_1.npz', observations=np.hstack([obs, np.zeros((len(obs), 5))]))
    pairs = explored_pairs(str(tmp_path / 'buffer_1.npz'), frame, client, n=5, clearance=0.3, min_geo=1.0, seed=0)
    cells = {client._cell(x, y) for x, y in seed_xy}
    assert pairs.shape == (5, 4)
    for sx, sy, gx, gy in pairs:
        assert client._cell(sx, sy) in cells and client._cell(gx, gy) in cells

    np.savez(tmp_path / 'buffer_2.npz', observations=np.hstack([obs[:1], np.zeros((1, 5))]))
    assert explored_pairs(str(tmp_path / 'buffer_2.npz'), frame, client, n=5, seed=0) is None


def test_value_goals_and_start_mark(tmp_path):
    from deploy.eval_pairs import load_start_mark, value_goals

    client = _world(tmp_path, gap=(0, 4))
    goals = value_goals(client, k=8, seed=0)
    assert goals.shape == (8, 2) and len({tuple(g) for g in goals}) == 8
    (tmp_path / 'm.json').write_text(json.dumps({'x': 1.0, 'y': 2.0, 'yaw': 0.5}))
    assert load_start_mark(str(tmp_path / 'm.json')).origin == (1.0, 2.0, 0.5)


# ---------- checkpoint scorer ----------------------------------------------------------------


@pytest.fixture(scope='module')
def scored_run(tmp_path_factory):
    """One sim run shared by the scorer tests: the Task 4 config, with short episodes to score quickly."""
    tmp = tmp_path_factory.mktemp('scored_run')
    _main(tmp, '--start_pose=1,1,0', '--online_steps=300', '--max_episode_steps=50')
    return _run_dir(tmp, 300)


def _score(run_dir, tmp_path, *extra):
    import csv
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, 'deploy/score_checkpoints.py', f'--run_dir={run_dir}', '--steps=300', '--n_pairs=3',
        f'--pairs_path={tmp_path / "pairs.json"}', *extra,
    ]  # fmt: skip
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    r = subprocess.run(cmd, check=False, cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with open(os.path.join(run_dir, 'scores.csv')) as f:
        return list(csv.DictReader(f))


SCORE_METRICS = ['SR', 'reach_rate', 'steps', 'length', 'length_ratio', 'collisions', 'plan_SR', 'plan_length', 'plan_ms']


def test_score_checkpoint_columns(scored_run, tmp_path):
    rows = _score(scored_run, tmp_path)
    assert len(rows) == 1 and rows[0]['step'] == '300'
    columns = {f'{s}_{m}' for s in ('fixed', 'explored') for m in SCORE_METRICS}
    assert columns | {'value_spearman', 'value_spearman_h0', 'explored_skipped_reason', 'step'} == set(rows[0])
    assert 0.0 <= float(rows[0]['fixed_SR']) <= 1.0 and 0.0 <= float(rows[0]['fixed_reach_rate']) <= 1.0
    assert -1.0 <= float(rows[0]['value_spearman']) <= 1.0 and -1.0 <= float(rows[0]['value_spearman_h0']) <= 1.0
    # The buffer exists, so the explored set is scored or infeasible, never skipped for a missing buffer.
    assert rows[0]['explored_skipped_reason'] in ('', 'infeasible')
    if rows[0]['explored_skipped_reason'] == '':
        assert all(rows[0][f'explored_{m}'] != '' for m in ('SR', 'reach_rate', 'collisions', 'plan_SR'))
    if float(rows[0]['fixed_SR']) == 0.0:
        assert rows[0]['fixed_steps'] == rows[0]['fixed_length'] == rows[0]['fixed_length_ratio'] == ''


def test_score_skips_explored_without_buffer(scored_run, tmp_path):
    import shutil

    run = shutil.copytree(scored_run, tmp_path / 'run')
    os.remove(run / 'buffer_300.npz')
    row = _score(run, tmp_path)[0]
    assert row['explored_skipped_reason'] == 'no buffer'
    assert all(row[f'explored_{m}'] == '' for m in SCORE_METRICS)
    assert row['fixed_SR'] != '' and row['fixed_collisions'] != '' and row['fixed_plan_SR'] != ''


def test_scoring_env_starts_pairs_in_the_run_frame(tmp_path):
    from deploy.score_checkpoints import place, scoring_env
    from envs.spot_mapless import RunFrame

    client = _world(tmp_path, gap=(0, 4))
    frame = RunFrame((1.05, 1.05, math.pi / 2))
    env = scoring_env(client, frame, {'goal_tol': 0.5, 'max_episode_steps': 20, 'control_hz': 10.0})
    start, goal = (3.05, 1.05), (3.05, 3.05)
    ob = place(env, client, frame, start, goal)
    x, y, yaw = frame.to_run(*start, math.pi / 2)  # facing the goal
    assert np.allclose(ob[:4], (x, y, math.cos(yaw), math.sin(yaw)), atol=1e-5)
    assert np.allclose(env.goal_xy, frame.to_run(*goal, 0.0)[:2])


def test_chain_crossing_wall_fails(tmp_path):
    from deploy.score_checkpoints import planned_chain
    from envs.spot_mapless import RunFrame

    client = _world(tmp_path)  # a full wall at x = 2.0..2.1
    frame = RunFrame((0.5, 0.5, 0.3))
    start, goal = (1.05, 2.05), (3.05, 2.05)
    goal_run = np.array(frame.to_run(*goal, 0.0)[:2])
    jump = lambda ob, g: goal_run  # noqa: E731  one subgoal straight onto the goal, through the wall
    assert planned_chain(jump, client, frame, start, goal, goal_tol=0.5)['plan_SR'] == 0.0
    # The same jump on the near side of the wall succeeds, so the failure is the crossing.
    near = (1.05, 3.05)
    near_run = np.array(frame.to_run(*near, 0.0)[:2])
    r = planned_chain(lambda ob, g: near_run, client, frame, start, near, goal_tol=0.5)
    assert r['plan_SR'] == 1.0 and r['plan_length'] == pytest.approx(1.0, abs=1e-6) and r['plan_ms'] >= 0.0


@pytest.mark.parametrize('agent_file', AGENT_FILES)
def test_select_subgoal_is_the_subgoal_sample_actions_uses(agent_file):
    import jax

    from deploy.time_jetson import build_agent
    from utils.value_query import select_subgoal

    agent, config = build_agent(agent_file, 8)
    ob, goal, seed = np.full(7, 0.3, dtype=np.float32), np.array([2.0, -1.0], dtype=np.float32), jax.random.PRNGKey(3)
    high_seed, low_seed = jax.random.split(seed)
    subgoal = select_subgoal(agent, ob, goal, high_seed)
    assert subgoal.shape == (2,)
    # Finish with the low-level flow exactly as `sample_actions` does; the actions must match.
    actions = jax.random.normal(low_seed, (config['action_dim'],))
    for i in range(config['flow_steps']):
        t = np.full((1,), i / config['flow_steps'], dtype=np.float32)
        actions = actions + agent.network.select('low_actor_flow')(ob, subgoal, actions, t) / config['flow_steps']
    assert np.allclose(np.clip(actions, -1, 1), agent.sample_actions(ob, goal, seed), atol=1e-4)


def test_value_spearman_averages_v_over_headings(tmp_path, monkeypatch):
    from scipy.stats import spearmanr

    import deploy.score_checkpoints as sc
    from deploy.time_jetson import build_agent
    from envs.spot_mapless import RunFrame
    from utils.value_query import goal_values

    monkeypatch.setattr(sc, 'MAX_CELLS', 30)
    client = _world(tmp_path, gap=(0, 4))
    frame = RunFrame((0.5, 0.5, 0.7))
    agent, _ = build_agent('agents/sharsa.py', 8)
    goals = np.array([[3.05, 3.05], [0.55, 3.55]])
    mean, h0 = sc.value_spearman(agent, client, frame, goals)

    # The same cells, valued one at a time through goal_values.
    rows, cols = np.nonzero(client.free)
    keep = np.random.default_rng(sc.SEED).choice(len(rows), 30, replace=False)
    rows, cols = rows[keep], cols[keep]
    goals_run = np.array([frame.to_run(x, y, 0.0)[:2] for x, y in goals])
    v = np.zeros((30, 8, 2))
    for i, (r, c) in enumerate(zip(rows, cols)):
        x, y, _ = frame.to_run(0.05 + c * 0.1, 0.05 + r * 0.1, 0.0)
        for j, h in enumerate(sc.HEADINGS):
            v[i, j] = goal_values(agent, [x, y, math.cos(h), math.sin(h), 0.0, 0.0, 0.0], goals_run)
    rhos, rhos_h0 = [], []
    for k, (gx, gy) in enumerate(goals):
        geo = client.distance_field_xy(gx, gy)[rows, cols]
        ok = np.isfinite(geo)
        rhos.append(spearmanr(v[ok, :, k].mean(1), -geo[ok])[0])
        rhos_h0.append(spearmanr(v[ok, 0, k], -geo[ok])[0])
    assert mean == pytest.approx(np.mean(rhos), abs=1e-6) and h0 == pytest.approx(np.mean(rhos_h0), abs=1e-6)
    assert mean != h0


def test_fixed_pairs_stay_inside_a_rotated_arena(tmp_path):
    from deploy.eval_pairs import geodesic
    from deploy.score_checkpoints import fixed_pairs
    from envs.spot_mapless import RunFrame

    client = _world(tmp_path, gap=(0, 40))  # no wall
    frame = RunFrame((0.5, 0.5, math.pi / 4))  # a diagonal arena: its seed-frame bounding box is far larger
    bounds = (0.0, -0.6, 3.5, 0.6)
    pairs = fixed_pairs(client, frame, bounds, n=20)
    assert pairs.shape == (20, 4)
    for sx, sy, gx, gy in pairs:
        for x, y in ((sx, sy), (gx, gy)):
            rx, ry, _ = frame.to_run(x, y, 0.0)
            assert bounds[0] <= rx <= bounds[2] and bounds[1] <= ry <= bounds[3]
        assert geodesic(client, (sx, sy), (gx, gy)) >= 1.5


# ---------- figures and the robot pair mode --------------------------------------------------


def _write_csv(path, header, rows):
    import csv

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _synthetic_run(tmp_path, sr, coverage):
    """Scores at steps 1000..; progress rows every 500 env steps (the first two in warmup.csv), 0.1 s and 0.05 m each."""
    steps = [1000 * (i + 1) for i in range(len(sr))]
    _write_csv(tmp_path / 'scores.csv', ['step', 'fixed_SR'], [[s, v] for s, v in zip(steps, sr)])
    header = ['progress/env_steps', 'progress/wall_time', 'progress/metres', 'explore/coverage']
    rows = [[s, s * 0.1, s * 0.05, c] for s, c in zip(range(500, 500 * (len(coverage) + 1), 500), coverage)]
    _write_csv(tmp_path / 'warmup.csv', header, rows[:2])
    _write_csv(tmp_path / 'train.csv', header, rows[2:])
    return str(tmp_path)


def test_mapping_time_absent_when_never_crossed(tmp_path):
    from deploy.plot_mapless import mapping_time

    # SR clears 0.8 only after coverage has, and coverage never clears 0.9 while SR is high.
    run = _synthetic_run(tmp_path, sr=[0.5, 0.7, 0.79, 0.8], coverage=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85])
    assert mapping_time(run) is None


def test_mapping_time_first_crossing(tmp_path):
    from deploy.plot_mapless import mapping_time

    # SR >= 0.8 from step 2000, coverage >= 0.9 from env step 3000: the first step with both is 3000.
    run = _synthetic_run(tmp_path, sr=[0.5, 0.85, 0.9, 0.95], coverage=[0.2, 0.5, 0.7, 0.8, 0.85, 0.92, 0.95, 0.95])
    m = mapping_time(run)
    assert m['step'] == 3000 and m['wall_time'] == pytest.approx(300.0) and m['metres'] == pytest.approx(150.0)


def test_mapping_time_interpolates_between_progress_rows(tmp_path):
    from deploy.plot_mapless import mapping_time

    run = _synthetic_run(tmp_path, sr=[0.9], coverage=[0.95, 0.95])  # score at 1000, progress rows at 500 and 1000
    assert mapping_time(run)['step'] == 1000
    _write_csv(tmp_path / 'scores.csv', ['step', 'fixed_SR'], [[750, 0.9]])
    m = mapping_time(run)
    assert m['wall_time'] == pytest.approx(75.0) and m['metres'] == pytest.approx(37.5)


def test_plot_mapless_writes_every_figure(scored_run, tmp_path):
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _score(scored_run, tmp_path)  # scores.csv with one row
    cmd = [sys.executable, 'deploy/plot_mapless.py', f'--runs=sharsa={scored_run},geo={scored_run}', f'--out={tmp_path / "figs"}']
    r = subprocess.run(cmd, check=False, cwd=repo, env=dict(os.environ, JAX_PLATFORMS='cpu'), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    names = ['sr_vs_steps', 'sr_vs_time', 'sr_vs_metres', 'length_ratio', 'coverage', 'value_spearman', 'value_snapshots', 'timing']
    for n in names:
        assert os.path.getsize(tmp_path / 'figs' / f'{n}.png') > 0, n
    assert 'mapping time' in r.stdout


def test_run_task_pairs_mode_in_sim(scored_run, tmp_path):
    import csv
    import subprocess
    import sys

    from deploy.eval_pairs import save_pairs
    from deploy.score_checkpoints import fixed_pairs
    from deploy.sim_client import SimSpotClient
    from envs.spot_mapless import SIM_OCCUPANCY_PATH, RunFrame

    client = SimSpotClient(SIM_OCCUPANCY_PATH, {})
    pairs = fixed_pairs(client, RunFrame((1.0, 1.0, 0.0)), [0, 0, 8, 6], 2)
    save_pairs(str(tmp_path / 'pairs.json'), pairs)
    (tmp_path / 'mark.json').write_text(json.dumps({'x': 1.0, 'y': 1.0, 'yaw': 0.0}))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, 'deploy/run_task.py', '--mapless', '--robot_hostname=sim', f'--restore_path={scored_run}',
        '--restore_epoch=300', '--start_pose=1,1,0', f'--start_mark={tmp_path / "mark.json"}',
        f'--pairs_path={tmp_path / "pairs.json"}', f'--log_path={tmp_path / "pairs_log.csv"}', '--max_episode_steps=30',
    ]  # fmt: skip
    r = subprocess.run(cmd, check=False, cwd=repo, env=dict(os.environ, JAX_PLATFORMS='cpu'), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with open(tmp_path / 'pairs_log.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and {'outcome', 'steps', 'metres', 'interventions', 'act_ms'} <= set(rows[0])
    assert all(r['outcome'] in ('reached', 'timeout', 'stuck', 'intervention') for r in rows)


# ---------- guided session (spec section 11) ---------------------------------------------------


class _SessionClient:
    def __init__(self, misses=2, pose=(0.0, 0.0, 0.0)):
        self.misses, self.calls, self.poses = misses, [], []
        self.tag = (7, (2.0, 1.0, math.pi / 2))

    def detect_fiducial(self, tag_id=None):
        self.calls.append(tag_id)
        if self.misses > 0:
            self.misses -= 1
            return None
        return self.tag

    def get_state(self):
        x, y, yaw = self.poses.pop(0)
        return dict(x=x, y=y, yaw=yaw)


def _scripted(answers):
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        return answers.pop(0) if answers else ''

    return input_fn, prompts


def test_session_retries_tag_saves_tag_frame_poses_and_queues_arms(tmp_path):
    from deploy.spot_session import ARMS, NOT_DETECTED, load_session, run_session

    client = _SessionClient(misses=2)
    client.poses = [(2.0, 4.0, 0.0), (2.0, 1.5, 0.0)]  # end, start (vision frame)
    input_fn, prompts = _scripted([])
    launched = []
    session = run_session(str(tmp_path), 'robot', lambda: client, input_fn=input_fn, print_fn=lambda *a, **k: None,
                          launch_fn=lambda root: launched.append(root) or 123)
    assert prompts.count(NOT_DETECTED) == 2 and len(client.calls) == 3
    assert 'END' in prompts[3] and 'START' in prompts[4] and 'Release control' in prompts[5]
    saved = load_session(os.path.join(tmp_path, 'session.json'))
    # tag at (2, 1) facing +y: a point 3 m along +y is 3 m along the tag's x axis
    assert np.allclose(saved['end_tag'][:2], (3.0, 0.0), atol=1e-9)
    assert np.allclose(saved['start_tag'][:2], (0.5, 0.0), atol=1e-9)
    assert saved['tag_id'] == 7 and saved['tag_pose'] == [2.0, 1.0, math.pi / 2] and session['arms'] == list(ARMS)
    assert launched == [str(tmp_path)] and list(ARMS) == ['sharsa', 'dual', 'phys', 'geo']


def _write_checkpoint(run_dir, step, params=True):
    import pickle

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / f'buffer_{step}.npz', a=np.arange(10))
    if params:
        (run_dir / f'params_{step}.pkl').write_bytes(pickle.dumps(dict(agent={})))
    for kind in ('sampler', 'progress'):
        (run_dir / f'{kind}_{step}.json').write_text('{}')


def test_latest_checkpoint_skips_corrupt_set_and_reports_it(tmp_path):
    from deploy.spot_session import latest_checkpoint

    run = tmp_path / 'runs' / 'dual' / 'e1'
    for n in (1000, 2000):
        _write_checkpoint(run, n)
    (run / 'buffer_2000.npz').write_bytes((run / 'buffer_2000.npz').read_bytes()[:40])
    said = []
    assert latest_checkpoint(str(tmp_path), 'dual', said.append) == (str(run), 1000)
    assert len(said) == 1 and 'skipping checkpoint 2000' in said[0]
    (run / 'sampler_1000.json').write_text('{"a"')
    assert latest_checkpoint(str(tmp_path), 'dual', said.append) is None
    assert len(said) == 3


def test_latest_checkpoint_needs_params_unless_warmup(tmp_path):
    from deploy.spot_session import latest_checkpoint

    run = tmp_path / 'runs' / 'dual' / 'e1'
    _write_checkpoint(run, 500, params=False)  # warmup: the agent did not exist yet
    assert latest_checkpoint(str(tmp_path), 'dual', print) == (str(run), 500)
    _write_checkpoint(run, 1000)
    _write_checkpoint(run, 2000, params=False)  # power cut between the buffer and the params
    assert latest_checkpoint(str(tmp_path), 'dual', print) == (str(run), 1000)


def test_tmp_leftovers_are_ignored_and_cleaned(tmp_path):
    from deploy.spot_session import clean_tmp, latest_checkpoint

    run = tmp_path / 'runs' / 'dual' / 'e1'
    _write_checkpoint(run, 1000)
    (run / 'buffer_2000.npz.tmp').write_bytes(b'half')
    assert latest_checkpoint(str(tmp_path), 'dual', print) == (str(run), 1000)
    clean_tmp(str(tmp_path), 'dual', lambda *a: None)
    assert not (run / 'buffer_2000.npz.tmp').exists()


def test_failed_atomic_save_leaves_no_final_or_tmp_file(tmp_path, monkeypatch):
    from utils.datasets import ReplayBuffer

    data = dict(observations=np.zeros((2, 3), np.float32), terminals=np.zeros(2))
    buffer = ReplayBuffer.create_from_initial_dataset(data, 4)
    path = str(tmp_path / 'buffer_1.npz')
    buffer.save(path)
    before = open(path, 'rb').read()

    def boom(f, **data):
        f.write(b'partial')
        raise OSError('power cut')

    monkeypatch.setattr(np, 'savez', boom)
    with pytest.raises(OSError):
        buffer.save(path)
    assert open(path, 'rb').read() == before and os.listdir(tmp_path) == ['buffer_1.npz']
    with pytest.raises(OSError):
        buffer.save(str(tmp_path / 'buffer_2.npz'))
    assert os.listdir(tmp_path) == ['buffer_1.npz']


def test_runner_resumes_a_session_left_running(tmp_path, monkeypatch):
    from deploy.spot_session import run_arms, save_session

    monkeypatch.setenv('SPOT_ONLINE_STEPS', '3000')
    runs = {a: dict(status='pending', run_dirs=[]) for a in ('sharsa', 'dual', 'phys', 'geo')}
    runs['sharsa']['status'] = 'running'  # the container was removed before the runner could write 'stopped'
    save_session(str(tmp_path / 'session.json'), dict(tag_id=0, hostname='sim', arms=list(runs), runs=runs))
    run = tmp_path / 'runs' / 'sharsa' / 'e1'
    _write_checkpoint(run, 1000)
    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)

        def wait(self):
            return 2

    run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None, sleep_fn=lambda s: None)
    assert f'--restore_path={run}' in cmds[0] and '--restore_epoch=1000' in cmds[0]


def test_runner_runs_arms_in_order_and_resumes_from_last_buffer(tmp_path, monkeypatch):
    from deploy.spot_session import load_session, run_arms, save_session

    monkeypatch.setenv('SPOT_ONLINE_STEPS', '3000')
    runs = {a: dict(status='pending', run_dirs=[]) for a in ('sharsa', 'dual', 'phys', 'geo')}
    runs['sharsa']['status'] = 'done'
    save_session(str(tmp_path / 'session.json'), dict(tag_id=0, hostname='sim', arms=list(runs), runs=runs))
    old = tmp_path / 'runs' / 'dual' / 'p' / 'g' / 'e1'
    old.mkdir(parents=True)
    for n in (1000, 2000):
        _write_checkpoint(old, n)
    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)

        def wait(self):
            return 0 if len(cmds) < 2 else 2

    assert run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None, sleep_fn=lambda s: None) == 2
    assert [c[c.index('--goal_mode=ab') - 1] for c in cmds] == ['--mapless'] * 9
    arms = [next(f for f in c if f.startswith('--save_dir=')).rsplit('/', 1)[1] for c in cmds]
    assert arms == ['dual'] + ['phys'] * 4 + ['geo'] * 4  # each failing arm is resumed 3 times, then the next runs
    assert f'--restore_path={old}' in cmds[0] and '--restore_epoch=2000' in cmds[0]
    assert '--agent.fk_kappa=15.0' in cmds[1] and not any(f.startswith('--restore') for f in cmds[1])
    for c in cmds:  # absl rejects --agent.x before --agent=
        first = next(i for i, f in enumerate(c) if f.startswith('--agent'))
        assert c[first].startswith('--agent=')
    s = load_session(str(tmp_path / 'session.json'))
    assert [s['runs'][a]['status'] for a in ('sharsa', 'dual', 'phys', 'geo')] == ['done', 'done', 'failed', 'failed']


def _runner_session(tmp_path, arms=('sharsa', 'dual')):
    from deploy.spot_session import save_session

    runs = {a: dict(status='pending', run_dirs=[]) for a in arms}
    save_session(str(tmp_path / 'session.json'), dict(tag_id=0, hostname='sim', arms=list(arms), runs=runs))


def test_runner_retries_a_failed_arm_from_its_checkpoint_then_moves_on(tmp_path, monkeypatch):
    from deploy.spot_session import load_session, run_arms

    monkeypatch.setenv('SPOT_ONLINE_STEPS', '3000')
    _runner_session(tmp_path)
    run = tmp_path / 'runs' / 'sharsa' / 'e1'
    codes, cmds, sleeps, lines = {'sharsa': [1, 1, 1, 1], 'dual': [1, 0]}, [], [], []

    class Proc:
        def __init__(self, cmd, stdout, **kwargs):
            self.arm = next(f for f in cmd if f.startswith('--save_dir=')).rsplit('/', 1)[1]
            cmds.append((self.arm, cmd))
            stdout.write(f'TimedOutError: {self.arm} lost wifi\n')
            stdout.flush()
            if self.arm == 'sharsa' and len(cmds) == 1:
                _write_checkpoint(run, 1000)  # the first attempt saved before it died

        def wait(self):
            return codes[self.arm].pop(0)

    code = run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: lines.append(a[0]), sleep_fn=sleeps.append)
    assert code == 1 and [a for a, _ in cmds] == ['sharsa'] * 4 + ['dual'] * 2 and sleeps == [10] * 4
    assert all(f'--restore_path={run}' in c and '--restore_epoch=1000' in c for _, c in cmds[1:4])
    assert sum('why: TimedOutError: sharsa lost wifi' in line for line in lines) == 4
    s = load_session(str(tmp_path / 'session.json'))
    assert [s['runs'][a]['status'] for a in ('sharsa', 'dual')] == ['failed', 'done']


def test_runner_never_retries_an_interrupted_arm(tmp_path):
    from deploy.spot_session import load_session, run_arms

    _runner_session(tmp_path)
    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)
            self.calls = 0

        def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt  # SIGINT reached the runner; the arm saves and exits non-zero
            return 1

    sleep = lambda s: pytest.fail('retried an interrupted arm')  # noqa: E731
    assert run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None, sleep_fn=sleep) == 1
    s = load_session(str(tmp_path / 'session.json'))
    assert len(cmds) == 1 and [s['runs'][a]['status'] for a in ('sharsa', 'dual')] == ['stopped', 'pending']


def test_runner_stops_on_a_pose_jump_without_retrying(tmp_path):
    from deploy.spot_session import POSE_JUMP_EXIT, load_session, run_arms

    _runner_session(tmp_path)
    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)

        def wait(self):
            return POSE_JUMP_EXIT

    sleep = lambda s: pytest.fail('retried after a pose jump')  # noqa: E731
    assert run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None, sleep_fn=sleep) == POSE_JUMP_EXIT
    s = load_session(str(tmp_path / 'session.json'))
    assert len(cmds) == 1 and [s['runs'][a]['status'] for a in ('sharsa', 'dual')] == ['stopped', 'pending']


def test_runner_clears_its_pid_on_exit_and_on_exception(tmp_path):
    from deploy.spot_session import load_session, run_arms

    _runner_session(tmp_path)

    class Proc:
        def __init__(self, cmd, **kwargs):
            assert load_session(str(tmp_path / 'session.json'))['runner_pid'] == os.getpid()

        def wait(self):
            return 0

    assert run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None) == 0
    assert load_session(str(tmp_path / 'session.json'))['runner_pid'] is None

    class Crash(Proc):
        def wait(self):
            raise SystemExit(143)  # SIGTERM, through the runner's handler

    _runner_session(tmp_path)
    with pytest.raises(SystemExit):
        run_arms(str(tmp_path), popen=Crash, print_fn=lambda *a, **k: None)
    assert load_session(str(tmp_path / 'session.json'))['runner_pid'] is None


def test_runner_resumes_an_arm_stopped_during_its_drain(tmp_path, monkeypatch):
    from deploy.spot_session import WARMUP_STEPS, run_arms

    monkeypatch.setenv('SPOT_ONLINE_STEPS', '3000')
    _runner_session(tmp_path, arms=('sharsa',))
    run = tmp_path / 'runs' / 'sharsa' / 'e1'
    _write_checkpoint(run, 3000)
    (run / 'progress_3000.json').write_text(json.dumps(dict(updates=3000 - WARMUP_STEPS - 1)))
    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)
            (run / 'progress_3000.json').write_text(json.dumps(dict(updates=3000 - WARMUP_STEPS)))

        def wait(self):
            return 1  # e.g. env.close failed after the drain: the next attempt finds the arm complete

    assert run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None, sleep_fn=lambda s: None) == 0
    assert len(cmds) == 1 and '--restore_epoch=3000' in cmds[0]


def test_session_resume_keeps_poses_and_new_moves_old_dir(tmp_path):
    from deploy.spot_session import load_session, run_session, save_session

    root = str(tmp_path)
    prev = dict(tag_id=7, end_tag=[3.0, 0.0, 0.0], start_tag=[0.5, 0.0, 0.0], tag_pose=[0, 0, 0], hostname='robot',
                arms=['sharsa'], runs={'sharsa': dict(status='running', run_dirs=[])}, runner_pid=None)
    save_session(os.path.join(root, 'session.json'), prev)
    os.makedirs(os.path.join(root, 'runs', 'sharsa'))
    client = _SessionClient(misses=0)
    input_fn, prompts = _scripted(['x', 'r'])
    run_session(root, 'robot', lambda: client, input_fn=input_fn, print_fn=lambda *a, **k: None, launch_fn=lambda r: 1)
    s = load_session(os.path.join(root, 'session.json'))
    assert client.calls == [7] and s['end_tag'] == [3.0, 0.0, 0.0] and s['tag_pose'] == [2.0, 1.0, math.pi / 2]
    assert not any('END' in p for p in prompts) and os.path.isdir(os.path.join(root, 'runs', 'sharsa'))

    client = _SessionClient(misses=0)
    client.poses = [(2.0, 4.0, 0.0), (2.0, 1.5, 0.0)]
    input_fn, prompts = _scripted(['n'])
    run_session(root, 'robot', lambda: client, input_fn=input_fn, print_fn=lambda *a, **k: None, launch_fn=lambda r: 1)
    olds = [d for d in os.listdir(root) if d.startswith('old_')]
    assert len(olds) == 1 and os.path.exists(os.path.join(root, olds[0], 'session.json'))
    assert os.path.isdir(os.path.join(root, olds[0], 'runs', 'sharsa'))
    assert not os.path.exists(os.path.join(root, 'runs')) and client.calls == [None]
    assert np.allclose(load_session(os.path.join(root, 'session.json'))['end_tag'][:2], (3, 0))


def test_ab_goals_alternate_on_reached_and_hold_otherwise():
    from envs.spot_mapless import ABGoals

    g = ABGoals((3.0, 0.0, 0.1), (0.5, 0.0, 0.0))
    seen = []
    for outcome in ('stuck', 'timeout', 'reached', 'intervention', 'reached', 'reached'):
        seen.append(ABGoals.names[g.idx])
        g.report(g.sample((0, 0), None), outcome, False, False)
    assert seen == ['end', 'end', 'end', 'start', 'start', 'end'] and g.idx == 1
    assert np.allclose(g.sample((0, 0), None), (0.5, 0.0))
    h = ABGoals((3.0, 0.0), (0.5, 0.0))
    h.load_state_dict(g.state_dict())
    assert h.idx == 1


def test_main_without_arguments_starts_the_session(tmp_path):
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k not in ('SPOT_IP', 'SPOT_USERNAME', 'SPOT_PASSWORD')}
    env.update(JAX_PLATFORMS='cpu', SPOT_SESSION_ROOT=str(tmp_path))
    r = subprocess.run([sys.executable, 'main.py'], cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and 'Set SPOT_IP, SPOT_USERNAME, SPOT_PASSWORD' in r.stderr


def test_detect_fiducial_picks_nearest_tag_and_normal_heading(monkeypatch):
    import sys
    import types

    from deploy.spot_client import SpotClient

    ns = types.SimpleNamespace

    def rot_z_to(nx, ny):  # rotation whose z axis is (nx, ny, 0)
        return ns(transform_point=lambda x, y, z: (nx * z, ny * z, 0.0) if z else (0.0, 0.0, 1.0))

    poses = {'filtered_5': ns(x=4.0, y=0.0, rot=rot_z_to(-1.0, 0.0)), 'raw_9': ns(x=1.0, y=1.0, rot=rot_z_to(0.0, 1.0))}
    world_object_pb2 = ns(WORLD_OBJECT_APRILTAG=2)
    frame_helpers = types.ModuleType('bosdyn.client.frame_helpers')
    frame_helpers.VISION_FRAME_NAME = 'vision'
    frame_helpers.get_a_tform_b = lambda snap, a, b: poses.get(b)
    frame_helpers.get_vision_tform_body = lambda snap: ns(x=0.0, y=0.0)
    api = types.ModuleType('bosdyn.api')
    api.world_object_pb2 = world_object_pb2
    for name, mod in [('bosdyn', types.ModuleType('bosdyn')), ('bosdyn.api', api),
                      ('bosdyn.client', types.ModuleType('bosdyn.client')),
                      ('bosdyn.client.frame_helpers', frame_helpers)]:
        monkeypatch.setitem(sys.modules, name, mod)

    def tag(i, filtered, raw):
        return ns(apriltag_properties=ns(tag_id=i, frame_name_fiducial_filtered=filtered, frame_name_fiducial=raw),
                  transforms_snapshot=None)

    objects = [tag(5, 'filtered_5', 'raw_5'), tag(9, '', 'raw_9')]
    c = SpotClient('robot', pose_source='vision')
    requested = []
    c.world_object_client = ns(
        list_world_objects=lambda object_type: requested.append(object_type) or ns(world_objects=objects)
    )
    c.state_client = ns(get_robot_state=lambda: ns(kinematic_state=ns(transforms_snapshot=None)))
    tid, (x, y, yaw) = c.detect_fiducial()
    assert requested[0] == [2] and tid == 9 and (x, y) == (1.0, 1.0) and math.isclose(yaw, math.pi / 2)
    tid, (x, y, yaw) = c.detect_fiducial(tag_id=5)
    assert tid == 5 and (x, y) == (4.0, 0.0) and math.isclose(abs(yaw), math.pi)
    objects[:] = []
    assert c.detect_fiducial() is None


def test_walk_data_reaches_targets_and_loads(tmp_path):
    from deploy.gen_walk_data import MAX_EPISODE_STEPS, generate_walk_data
    from utils.datasets import load_spot_datasets

    paths = generate_walk_data(2000, str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ['buffer_2000.npz'] and os.listdir(tmp_path) == ['buffer_2000.npz']
    train, val = load_spot_datasets(paths)
    assert len(train['observations']) + len(val['observations']) == 2000
    with np.load(paths[0]) as f:
        data = {k: f[k] for k in f.files}
    assert np.all(np.abs(data['actions']) <= 1.0) and np.array_equal(data['oracle_reps'], data['observations'][:, :2])
    # Stuck detection is off, so an episode shorter than the cap ended at its target.
    lengths = np.diff(np.r_[-1, np.nonzero(data['terminals'])[0]])[:-1]  # the last one is cut by n_steps
    assert len(lengths) > 10 and np.mean(lengths < MAX_EPISODE_STEPS) > 0.8


def _two_agents(agent_file='agents/sharsa.py'):
    import runpy

    from agents import agents
    from deploy.time_jetson import SPOT_OVERRIDES, synthetic_batch

    config = runpy.run_path(agent_file)['get_config']()
    config.update(SPOT_OVERRIDES)
    make = agents[config['agent_name']].create
    return make(0, synthetic_batch(8), config), make(1, synthetic_batch(8), config)


@pytest.mark.parametrize('agent_file', AGENT_FILES)
def test_restore_modules_copies_only_named_modules(tmp_path, agent_file):
    import jax

    from deploy.time_jetson import synthetic_batch
    from utils.flax_utils import restore_modules, save_agent

    a, b = _two_agents(agent_file)
    save_agent(a, str(tmp_path), 7)
    c = restore_modules(b, str(tmp_path), 7, ['low_actor_flow'])
    leaves = jax.tree_util.tree_leaves
    same = lambda x, y: all((u == v).all() for u, v in zip(leaves(x), leaves(y)))  # noqa: E731
    p, pa, pb = c.network.params, a.network.params, b.network.params
    assert 'modules_target_low_actor_flow' not in p  # SHARSA keeps no target for the low-level actor
    assert same(p['modules_low_actor_flow'], pa['modules_low_actor_flow'])
    for key in ('modules_high_value', 'modules_high_critic', 'modules_target_high_value', 'modules_high_actor_flow'):
        assert same(p[key], pb[key]) and not same(p[key], pa[key]), key
    c.update(synthetic_batch(8))  # the copied params keep the fresh optimizer state's structure


def test_restore_modules_copies_target_params_and_rejects_unknown_module(tmp_path):
    import jax

    from utils.flax_utils import restore_modules, save_agent

    a, b = _two_agents()
    save_agent(a, str(tmp_path), 7)
    c = restore_modules(b, str(tmp_path), 7, ['high_value'])
    leaves = lambda t: jax.tree_util.tree_leaves(t)  # noqa: E731
    for key in ('modules_high_value', 'modules_target_high_value'):
        assert all((u == v).all() for u, v in zip(leaves(c.network.params[key]), leaves(a.network.params[key])))
    with pytest.raises(AssertionError, match='not a module'):
        restore_modules(b, str(tmp_path), 7, ['low_actor'])


def test_mapless_warm_start_runs_in_sim(tmp_path):
    import glob
    import subprocess
    import sys

    from deploy.gen_walk_data import generate_walk_data

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = tmp_path / 'walk'
    generate_walk_data(2000, str(data))
    env = {k: v for k, v in os.environ.items() if k != 'WANDB_API_KEY'}
    env.update(WANDB_MODE='disabled', JAX_PLATFORMS='cpu', HOME=str(tmp_path))
    cmd = [
        sys.executable, 'main.py', '--env_name=spot-walk-v0', f'--dataset_dir={data}', f'--save_dir={tmp_path}/pre',
        '--offline_steps=200', '--save_interval=200', '--log_interval=100', '--eval_interval=0',
        '--agent.batch_size=16', '--agent.gc_negative=True', '--agent.value_loss_type=squared',
        '--agent.subgoal_steps=10',
    ]  # fmt: skip
    r = subprocess.run(cmd, check=False, cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (pre,) = glob.glob(f'{tmp_path}/pre/**/params_200.pkl', recursive=True)
    warm = [f'--warm_start_path={os.path.dirname(pre)}', '--warm_start_epoch=200',
            '--warm_start_modules=low_actor_flow']
    r = _main(tmp_path / 'online', '--start_pose=1,1,0', '--online_steps=200', *warm)
    assert f'Warm start: restored low_actor_flow from {pre}' in r.stdout


def test_walking_warmup_and_frozen_low_actor_act_from_the_warm_start(tmp_path):
    """A low actor whose only nonzero weight is the output bias (50, -50, 50) always acts (1, -1, 1) after clipping.

    Uniform random warmup never produces that exactly, and an unfrozen actor drifts off it once updates start.
    """
    import csv
    import glob
    import pickle

    import jax

    from deploy.time_jetson import build_agent
    from utils.flax_utils import save_agent

    agent, _ = build_agent('agents/sharsa.py', 8)
    params = dict(agent.network.params)
    params['modules_low_actor_flow'] = jax.tree_util.tree_map(
        lambda x: np.full(x.shape, [50.0, -50.0, 50.0], np.float32) if x.shape == (3,) else np.zeros_like(x),
        params['modules_low_actor_flow'],
    )
    assert sum(x.shape == (3,) for x in jax.tree_util.tree_leaves(params['modules_low_actor_flow'])) == 1  # the bias
    save_agent(agent.replace(network=agent.network.replace(params=params)), str(tmp_path), 0)
    _main(tmp_path / 'online', '--start_pose=1,1,0', '--online_steps=300', f'--warm_start_path={tmp_path}',
          '--warm_start_epoch=0', '--agent.freeze_low_actor=True', '--max_episode_steps=100')
    (run,) = {os.path.dirname(p) for p in glob.glob(f'{tmp_path}/online/**/steps.csv', recursive=True)}
    with open(os.path.join(run, 'steps.csv')) as f:
        actions = np.array([[float(r[k]) for k in ('action_x', 'action_y', 'action_w')] for r in csv.DictReader(f)])
    bad = np.nonzero(~np.all(actions == [1.0, -1.0, 1.0], axis=1))[0]
    assert len(actions) == 300 and len(bad) == 0, (bad, actions[bad])
    with open(os.path.join(run, 'params_300.pkl'), 'rb') as f:
        final = pickle.load(f)['agent']['network']['params']['modules_low_actor_flow']
    pairs = zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(params['modules_low_actor_flow']))
    assert all(np.array_equal(u, v) for u, v in pairs)  # 200 updates later, still the warm start


@pytest.mark.parametrize('agent_file', AGENT_FILES)
def test_frozen_low_actor_is_bit_identical_after_updates(agent_file):
    import jax

    from deploy.time_jetson import build_agent, synthetic_batch

    agent, _ = build_agent(agent_file, 8, {'freeze_low_actor': True})
    before = agent.network.params
    for _ in range(3):
        agent, info = agent.update(synthetic_batch(8))
    after = agent.network.params

    def same(key):
        pairs = zip(jax.tree_util.tree_leaves(before[key]), jax.tree_util.tree_leaves(after[key]))
        return all(np.array_equal(u, v) for u, v in pairs)

    assert same('modules_low_actor_flow') and not same('modules_high_value')
    low = [k for k in info if k.startswith('low_actor/')]
    assert low and all(np.isfinite(info[k]) for k in low)  # still logged


def test_session_offers_walk_pretrain_and_arms_get_the_warm_start(tmp_path):
    from deploy.spot_session import WALK_DIR, arm_command, load_session, offer_walk_pretrain, run_session

    root = str(tmp_path)
    assert not any(f.startswith(('--warm_start', '--agent.freeze')) for f in arm_command(root, 'sharsa', 'sim', 100))
    made = []

    def pretrain(r, print_fn):
        made.append(r)
        for step in (500, 10000):
            run = tmp_path / WALK_DIR / 'runs' / 'p' / 'g' / 'e'
            run.mkdir(parents=True, exist_ok=True)
            (run / f'params_{step}.pkl').write_bytes(b'')

    input_fn, prompts = _scripted(['x', 'c'])
    offer_walk_pretrain(root, input_fn, lambda *a, **k: None, pretrain)
    assert made == [root] and len(prompts) == 2 and '[c]reate' in prompts[0]
    run = str(tmp_path / WALK_DIR / 'runs' / 'p' / 'g' / 'e')
    for arm in ('sharsa', 'dual', 'phys', 'geo'):
        cmd = arm_command(root, arm, 'sim', 100, restore=('/r', 1000), warm_start=f'{run}/params_10000.pkl')
        assert cmd[-4:] == [f'--warm_start_path={run}', '--warm_start_epoch=10000',
                            '--warm_start_modules=low_actor_flow', '--agent.freeze_low_actor=True']
    input_fn, prompts = _scripted([])
    offer_walk_pretrain(root, input_fn, lambda *a, **k: None, pretrain)  # exists: nothing asked, nothing made
    assert made == [root] and prompts == []

    # A fresh session records the highest checkpoint and keeps it, since it knows nothing of the room.
    client = _SessionClient(misses=0)
    client.poses = [(2.0, 4.0, 0.0), (2.0, 1.5, 0.0)]
    run_session(root, 'robot', lambda: client, input_fn=_scripted([])[0], print_fn=lambda *a, **k: None,
                launch_fn=lambda r: 1)
    assert load_session(os.path.join(root, 'session.json'))['warm_start'] == f'{run}/params_10000.pkl'
    client.poses = [(2.0, 4.0, 0.0), (2.0, 1.5, 0.0)]
    input_fn, prompts = _scripted(['n'])
    run_session(root, 'robot', lambda: client, input_fn=input_fn, print_fn=lambda *a, **k: None, launch_fn=lambda r: 1,
                pretrain_fn=pretrain)
    assert os.path.isdir(tmp_path / WALK_DIR) and made == [root] and not any('[c]reate' in p for p in prompts)


def _warm_session(tmp_path, warm_start, legacy=False):
    from deploy.spot_session import save_session

    session = dict(tag_id=7, end_tag=[3.0, 0.0, 0.0], start_tag=[0.5, 0.0, 0.0], tag_pose=[0, 0, 0], hostname='sim',
                   arms=['sharsa'], runs={'sharsa': dict(status='pending', run_dirs=[])}, runner_pid=None)
    if not legacy:
        session['warm_start'] = warm_start
    save_session(str(tmp_path / 'session.json'), session)


def _runner_cmds(tmp_path):
    from deploy.spot_session import run_arms

    cmds = []

    class Proc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)

        def wait(self):
            return 0

    code = run_arms(str(tmp_path), popen=Proc, print_fn=lambda *a, **k: None)
    return code, cmds


def test_resume_uses_the_recorded_warm_start_and_never_offers_pretrain(tmp_path):
    from deploy.spot_session import WALK_DIR, run_session

    recorded = tmp_path / 'pre' / 'params_10000.pkl'
    recorded.parent.mkdir()
    recorded.write_bytes(b'')
    newer = tmp_path / WALK_DIR / 'runs' / 'e' / 'params_20000.pkl'  # created after the session: not used
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b'')
    _warm_session(tmp_path, str(recorded))
    input_fn, prompts = _scripted(['r'])
    launched = []
    offer = lambda *a, **k: pytest.fail('offered the pretrain on resume')  # noqa: E731
    run_session(str(tmp_path), 'sim', lambda: _SessionClient(misses=0), input_fn=input_fn,
                print_fn=lambda *a, **k: None, launch_fn=launched.append, pretrain_fn=offer)
    assert launched and not any('[c]reate' in p for p in prompts)
    code, cmds = _runner_cmds(tmp_path)
    assert code == 0 and f'--warm_start_path={recorded.parent}' in cmds[0] and '--warm_start_epoch=10000' in cmds[0]
    assert '--agent.freeze_low_actor=True' in cmds[0]


def test_resume_with_a_missing_warm_start_stops_instead_of_unfreezing(tmp_path):
    from deploy.spot_session import run_session

    _warm_session(tmp_path, str(tmp_path / 'gone' / 'params_10000.pkl'))
    lines, launched = [], []
    session = run_session(str(tmp_path), 'sim', lambda: pytest.fail('connected'), input_fn=_scripted(['r'])[0],
                          print_fn=lambda *a, **k: lines.append(a[0]), launch_fn=launched.append)
    assert session is None and not launched and any('no longer exists' in line for line in lines)
    code, cmds = _runner_cmds(tmp_path)
    assert code == 1 and cmds == []


def test_legacy_session_without_the_key_runs_cold(tmp_path):
    from deploy.spot_session import WALK_DIR

    ckpt = tmp_path / WALK_DIR / 'runs' / 'e' / 'params_10000.pkl'  # on disk, but not recorded
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'')
    _warm_session(tmp_path, None, legacy=True)
    code, cmds = _runner_cmds(tmp_path)
    assert code == 0 and not any(f.startswith(('--warm_start', '--agent.freeze')) for f in cmds[0])


def test_skipped_walk_pretrain_starts_the_arms_cold(tmp_path):
    from deploy.spot_session import offer_walk_pretrain

    input_fn, prompts = _scripted(['s'])
    offer_walk_pretrain(str(tmp_path), input_fn, lambda *a, **k: None, lambda *a, **k: pytest.fail('pretrained'))
    assert len(prompts) == 1


@pytest.mark.parametrize('async_updates', [True, False])
def test_async_updates_do_not_stretch_dt(tmp_path, monkeypatch, async_updates):
    import csv
    import time

    import envs.spot_mapless
    import main
    from agents import agents

    agent_class = agents['sharsa']
    update = agent_class.update

    def slow_update(self, batch):
        out = update(self, batch)
        time.sleep(0.15)  # a 150 ms update: longer than the 100 ms control period
        return out

    def pace(seconds):  # 2 ms naps: macOS stretches one long sleep by up to 10%, which alone reads as an overrun
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            time.sleep(max(0.0, min(0.002, end - time.perf_counter())))

    monkeypatch.setattr(agent_class, 'update', slow_update)
    monkeypatch.setattr(envs.spot_mapless, '_no_sleep', pace)  # pace the sim at 10 Hz, as on the robot
    monkeypatch.setattr(main.wandb, 'log', lambda *a, **k: None)  # in-process runs never call wandb.init
    main.FLAGS(['main.py', '--mapless', '--robot_hostname=sim', '--arena_bounds=0,0,8,6', '--start_pose=1,1,0',
                '--warmup_steps=50', '--online_steps=250', '--log_interval=250', f'--save_dir={tmp_path}',
                f'--async_updates={async_updates}'])
    main.FLAGS.agent.update(batch_size=16, gc_negative=True, value_loss_type='squared', subgoal_steps=10)
    try:
        main.train_online(main.FLAGS.agent)
    finally:
        main.FLAGS.unparse_flags()
    with open(tmp_path / 'train.csv') as f:
        row = list(csv.DictReader(f))[-1]
    with open(tmp_path / 'progress_250.json') as f:
        updates = json.load(f)['updates']
    # Async drains its update debt at the end, so it lands exactly on the env steps since warmup; inline never exceeds it.
    assert updates == 200 if async_updates else 0 < updates <= 200
    overrun = float(row['time/overrun_frac'])
    assert overrun < 0.05 if async_updates else overrun > 0.5


def test_async_drain_ends_on_one_update_per_step_and_a_resume_keeps_the_debt(tmp_path, monkeypatch):
    import time

    import main
    from agents import agents

    agent_class = agents['sharsa']
    update = agent_class.update

    def slow_update(self, batch):  # the unpaced sim steps far faster than this learner
        out = update(self, batch)
        time.sleep(0.03)
        return out

    advance, phases, interrupt = main.advance_bar, [], [True]

    def interrupt_drain(bar, env_step, phase):
        phases.append(phase.split()[0])
        if interrupt[0] and phase.startswith('drain') and not phase.startswith('drain 0/'):  # Ctrl+C mid-drain
            raise KeyboardInterrupt
        advance(bar, env_step, phase)

    monkeypatch.setattr(agent_class, 'update', slow_update)
    monkeypatch.setattr(main, 'advance_bar', interrupt_drain)
    monkeypatch.setattr(main.wandb, 'log', lambda *a, **k: None)  # in-process runs never call wandb.init

    def run(save_dir, *extra):
        main.FLAGS(['main.py', '--mapless', '--robot_hostname=sim', '--arena_bounds=0,0,8,6', '--start_pose=1,1,0',
                    '--warmup_steps=50', '--online_steps=150', '--log_interval=50', '--save_interval=1000',
                    f'--save_dir={save_dir}', '--async_updates=True', *extra])
        main.FLAGS.agent.update(batch_size=16, gc_negative=True, value_loss_type='squared', subgoal_steps=10)
        try:
            main.train_online(main.FLAGS.agent)
        finally:
            main.FLAGS.unparse_flags()

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    with open(tmp_path / 'progress_150.json') as f:
        first = json.load(f)['updates']
    assert 0 < first < 100 and os.path.exists(tmp_path / 'params_150.pkl')  # saved with debt left

    interrupt[0] = False
    del phases[:]
    os.makedirs(tmp_path / 'resume')
    run(tmp_path / 'resume', f'--restore_path={tmp_path}', '--restore_epoch=150')
    with open(tmp_path / 'resume' / 'progress_150.json') as f:
        assert json.load(f)['updates'] == 100  # 150 env steps - 50 warmup, the first run's debt included
    assert set(phases) == {'drain'}  # the resume takes no env step, it only finishes the drain
