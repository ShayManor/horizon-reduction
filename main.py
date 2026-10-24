import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from functools import partial

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets, make_online_env
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer, get_size, load_spot_datasets
from utils.evaluation import evaluate, supply_rng
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

GOAL_SLICE = 2  # observation dims carried into `oracle_reps` as the goal: (x, y)

dataset_class_dict = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
}

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 5000000, 'Number of offline steps.')

flags.DEFINE_integer('online_steps', 0, 'Number of online (on-robot) env steps. 0 runs the offline path.')
flags.DEFINE_integer('warmup_steps', 10000, 'Transitions collected before the first gradient step.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient steps per env step. Time one update against the control period.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer capacity. Must exceed the run so the ring never wraps.')
flags.DEFINE_float('control_hz', 10.0, 'Control rate in Hz.')
flags.DEFINE_integer('max_episode_steps', 1000, 'Truncation, and therefore which episodes the steps metric excludes.')
flags.DEFINE_list('task_ids', None, 'Task ids (1-based) from the map task list. None uses all of them.')
flags.DEFINE_string('graphnav_map_path', None, 'Directory holding the downloaded GraphNav map.')
flags.DEFINE_string('robot_hostname', None, 'Spot address. Credentials come from SPOT_USERNAME/SPOT_PASSWORD.')
flags.DEFINE_integer('seed_episodes', 0, 'GraphNav-driven episodes collected before anything else.')
flags.DEFINE_float('goal_tol', 0.5, 'Goal tolerance in metres.')
flags.DEFINE_list('vel_limits', ['0.6', '0.4', '0.8'], 'Forward, lateral and yaw caps. Measure these in your space.')
flags.DEFINE_float('reverse_limit', None, 'Backward cap in m/s. Defaults to the forward cap.')
flags.DEFINE_bool('mapless', False, 'Train with no map: run-frame odometry, frontier goals, reset-free episodes.')
flags.DEFINE_list('arena_bounds', None, 'Mapless: xmin,ymin,xmax,ymax in the run frame, in metres. Required.')
flags.DEFINE_enum('goal_mode', 'frontier', ['frontier', 'ab'], 'Mapless: frontier sampler, or end/start (session).')
flags.DEFINE_string('session_path', None, 'Mapless ab: session.json from deploy/spot_session.py (tag, start, end).')
flags.DEFINE_enum('goal_rule', 'value', ['value', 'nearest'], 'Mapless: frontier cell with the highest V, or nearest.')
flags.DEFINE_float('p_revisit', 0.3, 'Mapless: probability of a goal at a visited cell instead of the frontier.')
flags.DEFINE_integer('stuck_window', 30, 'Mapless: steps over which the robot must move stuck_dist.')
flags.DEFINE_float('stuck_dist', 0.15, 'Mapless: net displacement in metres under which an episode is stuck.')
flags.DEFINE_list('start_pose', None, 'Mapless sim only: seed-frame x,y,yaw. Defaults to spot_data/start_mark.json.')
flags.DEFINE_bool('async_updates', False, 'Run updates in a learner thread. Not implemented yet.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 250000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 5000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 15, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('verbose', 0, 'Verbosity level.')

config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)


def make_transition(ob, action, next_ob, terminal):
    """One replay-buffer row.

    `oracle_reps` is the (x, y) slice of the observation. `GCDataset` routes goals through that key
    when it is present, so a relabeled goal is a position and not a full 7-dim pose, which is what
    the env hands back as `info['goal']` and what the high-level policy emits as a subgoal.

    `terminals` marks trajectory boundaries and nothing else, so truncation sets it too. Rewards
    and masks come from relabeling in `GCDataset.sample`, never from this field.
    """
    ob = np.asarray(ob, dtype=np.float32)
    return dict(
        observations=ob,
        next_observations=np.asarray(next_ob, dtype=np.float32),
        actions=np.asarray(action, dtype=np.float32),
        oracle_reps=ob[:GOAL_SLICE],
        terminals=np.float32(terminal),
    )


def episode_metrics(records, num_tasks, window_episodes):
    """Completion rate and steps to completion, per task, over the last `window_episodes`.

    Steps are averaged over completed episodes only: a truncated episode ends at
    `max_episode_steps`, which describes the timeout and not the policy. A task with no completions
    in the window therefore has no steps value and is left out rather than imputed.
    """
    window = records[-window_episodes * max(num_tasks, 1):]
    by_task = defaultdict(list)
    for record in window:
        by_task[record['task_name']].append(record)

    metrics = {}
    overall = defaultdict(list)
    for task_name, rows in by_task.items():
        success = np.mean([row['success'] for row in rows])
        metrics[f'evaluation/{task_name}_success'] = success
        overall['success'].append(success)

        completed = [row['steps'] for row in rows if row['success'] > 0]
        if len(completed) > 0:
            metrics[f'evaluation/{task_name}_steps'] = np.mean(completed)
            overall['steps'].append(np.mean(completed))

    for k, v in overall.items():
        metrics[f'evaluation/overall_{k}'] = np.mean(v)
    metrics['evaluation/num_episodes'] = len(records)
    return metrics


def online_step_range(env_step, seeded_steps, online_steps):
    """The env steps the online loop will take.

    `env_step` indexes the buffer, so a buffer seeded from an offline dataset starts it well past
    zero. The budget counts online steps, so those seeded transitions are added back: without it an
    `online_steps` smaller than the offline dataset gives an empty range, and the run trains on
    nothing and reports no error.
    """
    steps = range(env_step + 1, seeded_steps + online_steps + 1)
    assert len(steps) > 0, (
        f'online_steps={online_steps} is already spent at env_step={env_step}. It counts steps '
        f'taken online, and {seeded_steps} of the buffer was seeded from the offline dataset.'
    )
    return steps


def mapless_start_pose():
    """The sim's seed-frame start pose: `--start_pose`, else the start mark. The robot ignores it."""
    from envs.spot_maze import SIM_HOSTNAME

    if FLAGS.robot_hostname != SIM_HOSTNAME:
        return None
    if FLAGS.start_pose is not None:
        return [float(v) for v in FLAGS.start_pose]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spot_data', 'start_mark.json')
    assert os.path.exists(path), f'the mapless sim needs --start_pose=x,y,yaw or {path}'
    with open(path) as f:
        mark = json.load(f)
    return [mark['x'], mark['y'], mark['yaw']]


def train_online(config):
    """Train on the robot: step it, fill the buffer, update, repeat.

    Nothing here touches a file the agents read. They consume a batch dict from `GCDataset.sample`,
    so a growing replay buffer substitutes for a static dataset without any agent change.
    """
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    dataset_class = dataset_class_dict[config['dataset_class']]
    agent_class = agents[config['agent_name']]

    task_ids = None if FLAGS.task_ids is None else [int(t) for t in FLAGS.task_ids]
    ab = FLAGS.mapless and FLAGS.goal_mode == 'ab'
    if FLAGS.mapless:
        from envs.spot_mapless import ABGoals, GoalSampler, RunFrame, retake_control
        from utils.online_metrics import OnlineMetrics
        from utils.value_query import goal_values

        assert not FLAGS.async_updates, '--async_updates is not implemented yet'
        FLAGS.seed_episodes = 0  # GraphNav is never touched, so warmup is random actions only
        warm_rng = np.random.default_rng(FLAGS.seed)  # untrained V: a uniformly random frontier cell
        if ab:
            # The run frame is the AprilTag's. The sim's tag pose is stored; the robot's is detected below.
            from deploy.spot_session import load_session, tag_frame
            from envs.spot_maze import SIM_HOSTNAME

            session = load_session(FLAGS.session_path)
            sampler = ABGoals(session['end_tag'], session['start_tag'])
            frame = RunFrame(session['tag_pose']) if FLAGS.robot_hostname == SIM_HOSTNAME else None
            start_pose = None if frame is None else list(frame.from_run(*session['start_tag']))
        else:
            assert FLAGS.arena_bounds is not None, '--mapless needs --arena_bounds=xmin,ymin,xmax,ymax (run frame, m)'
            sampler = GoalSampler(
                [float(v) for v in FLAGS.arena_bounds], p_revisit=FLAGS.p_revisit, rule=FLAGS.goal_rule, seed=FLAGS.seed
            )
            frame, start_pose = None, mapless_start_pose()
        online = OnlineMetrics(1.0 / FLAGS.control_hz)
        map_kwargs = dict(
            mapless=True, start_pose=start_pose, frame=frame, stuck_window=FLAGS.stuck_window,
            stuck_dist=FLAGS.stuck_dist,
        )
    else:
        map_kwargs = dict(graphnav_map_path=FLAGS.graphnav_map_path, task_ids=task_ids)
    env = make_online_env(
        FLAGS.env_name,
        robot_hostname=FLAGS.robot_hostname,
        **map_kwargs,
        control_hz=FLAGS.control_hz,
        max_episode_steps=FLAGS.max_episode_steps,
        goal_tol=FLAGS.goal_tol,
        vel_limits=tuple(float(v) for v in FLAGS.vel_limits),
        reverse_limit=FLAGS.reverse_limit,
        seed=FLAGS.seed,
    )
    if ab and env.frame is None:
        env.frame = tag_frame(env.client, session)
    # Mapless episodes have no task: one name keeps `episode_metrics` working per task as before.
    # In ab mode the goal's name is the task, so evaluation/* is per direction.
    tasks = [dict(task_name='mapless')] if FLAGS.mapless else env.task_infos
    if ab:
        tasks = [dict(task_name=name) for name in ABGoals.names]
    num_tasks = len(tasks)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    step_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'steps.csv'))
    episode_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'episodes.csv'))
    session_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'session_log.csv')) if FLAGS.mapless else None
    # Warmup rows carry no training keys and CsvLogger fixes its header on the first row, so they get their own file.
    warmup_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'warmup.csv')) if FLAGS.mapless else None

    episode_records = []
    episode_idx = 0
    env_step = 0
    seeded_steps = 0  # offline transitions, which do not count against the online step budget
    # Mapless bookkeeping: where the robot is, the last step's info, and the current episode's totals.
    robot_xy, last_info, ep_metres, ep_blocked, session_started = (0.0, 0.0), None, 0.0, False, False

    def save_sampler(step):
        with open(os.path.join(FLAGS.save_dir, f'sampler_{step}.json'), 'w') as f:
            json.dump(sampler.state_dict(), f)
        with open(os.path.join(FLAGS.save_dir, f'progress_{step}.json'), 'w') as f:
            json.dump(online.state_dict(), f)

    def mapless_metrics():
        return online.summary(sampler.stats(), sampler.coverage())

    def log_episode(task_idx, steps, terminated, truncated, success, start_step):
        nonlocal episode_idx
        task_idx = sampler.idx if ab else task_idx  # read before `report` switches the goal
        task = tasks[task_idx]
        if FLAGS.mapless:
            goal, outcome = last_info['goal'], last_info['outcome']
            sampler.report(goal, outcome, ep_blocked, warmup=buffer.size < FLAGS.warmup_steps, end_xy=robot_xy)
            online.on_episode(outcome)
            where = dict(goal_x=float(goal[0]), goal_y=float(goal[1]), outcome=outcome, metres=ep_metres)
        else:
            where = dict(start_waypoint=task['start'], goal_waypoint=task['goal'])
        record = dict(
            episode=episode_idx,
            task_id=task_idx + 1,
            task_name=task['task_name'],
            **where,
            steps=steps,
            terminated=int(terminated),
            truncated=int(truncated),
            success=float(success),
            start_env_step=start_step,
        )
        episode_records.append(record)
        episode_logger.log(dict(record), step=env_step)
        episode_idx += 1
        if FLAGS.mapless and outcome == 'intervention':
            if last_info['jump']:
                # The run frame no longer matches the floor, so nothing after this could share it.
                # Raised inside the try below, whose finally saves the buffer, sampler and agent.
                raise SystemExit(
                    f'Pose jump at env step {env_step}: odometry reset or the robot was carried. Return Spot '
                    f'to the start mark and resume with --restore_epoch={env_step}.'
                )
            retake_control(env.client)

    def store_transition(ob, action, next_ob, terminal, info, ep_steps):
        if FLAGS.mapless and info['jump']:
            # Odometry reset or the robot was carried: not a transition. End the episode at the last real one.
            if ep_steps > 1:
                buffer['terminals'][buffer.size - 1] = 1.0
            return
        buffer.add_transition(make_transition(ob, action, next_ob, terminal))

    def log_step(info, terminated, truncated, act_ms=None):
        nonlocal robot_xy, last_info, ep_metres, ep_blocked
        row = {k: v for k, v in info.items() if k != 'goal'}
        row['episode'] = episode_idx
        row['terminated'] = int(terminated)
        row['truncated'] = int(truncated)
        step_logger.log(row, step=env_step)
        if FLAGS.mapless:
            online.on_step(info, act_ms)
            if not info['jump']:
                sampler.observe(info['x'], info['y'])
            robot_xy, last_info = (info['x'], info['y']), info
            ep_metres += info['step_metres']
            ep_blocked = ep_blocked or info['blocked']

    def mapless_reset(agent=None, ob=None):
        """Pick the next goal and attempt it from wherever the robot stands. V scores it once trained."""
        nonlocal ep_metres, ep_blocked, session_started
        warm = agent is None or buffer.size < FLAGS.warmup_steps
        value_fn = (lambda goals: warm_rng.random(len(goals))) if warm else partial(goal_values, agent, ob)
        ob, info = env.reset(options=dict(goal=sampler.sample(robot_xy, value_fn)))
        sampler.observe(info['x'], info['y'])
        ep_metres, ep_blocked = 0.0, False
        if not session_started:
            session_logger.log(
                dict(
                    event='start',
                    wall_time=time.time(),
                    x=info['x'],
                    y=info['y'],
                    yaw=info['yaw'],
                    drift_x='',
                    drift_y='',
                    drift_yaw='',
                ),
                step=env_step,
            )
            session_started = True
        return ob, info

    # -- buffer: restored, or created and filled by warmup episodes ---------------------------
    if FLAGS.restore_path is not None:
        candidates = glob.glob(FLAGS.restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
        buffer_path = os.path.join(candidates[0], f'buffer_{FLAGS.restore_epoch}.npz')
        if os.path.exists(buffer_path):
            buffer = ReplayBuffer.load(buffer_path, FLAGS.buffer_size)
            print(f'Restored {buffer.size} transitions from {candidates[0]}')
        else:
            # Offline-to-online. The offline path saves weights and no buffer, so the warm start
            # seeds the buffer with the dataset those weights were trained on. Starting online from
            # an empty buffer would have the restored weights learn from the first few episodes
            # alone, and the first thing online updates would do is unlearn the offline stage.
            assert FLAGS.dataset_dir is not None, (
                f'{buffer_path} does not exist, so this is a warm start from an offline run. '
                'Pass --dataset_dir so the buffer starts from the offline dataset.'
            )
            paths = [f for f in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in f]
            offline_data, _ = load_spot_datasets(paths)
            assert get_size(offline_data) < FLAGS.buffer_size, (
                f'buffer_size={FLAGS.buffer_size} does not leave room for {get_size(offline_data)} '
                'offline transitions plus the online ones. Raise it.'
            )
            buffer = ReplayBuffer.create_from_initial_dataset(offline_data, FLAGS.buffer_size)
            seeded_steps = buffer.size
            print(f'Seeded {buffer.size} offline transitions from {FLAGS.dataset_dir}')
        env_step = buffer.size
        # A crash lands mid-episode. The run stopped at that index, so it is a real trajectory
        # boundary; marking it keeps the partial episode instead of discarding its transitions.
        buffer['terminals'][buffer.size - 1] = 1.0
        if FLAGS.mapless:
            sampler_path = os.path.join(candidates[0], f'sampler_{FLAGS.restore_epoch}.json')
            if os.path.exists(sampler_path):
                with open(sampler_path) as f:
                    sampler.load_state_dict(json.load(f))
                print(f'Restored sampler from {sampler_path}')
            progress_path = os.path.join(candidates[0], f'progress_{FLAGS.restore_epoch}.json')
            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    online.load_state_dict(json.load(f))
            else:
                print(f'No {progress_path}: metres, updates and safety counters restart at 0.')
            online.env_steps = env_step  # continuous across sessions even when only the buffer was restored
    else:
        # `ReplayBuffer.create` reads only shape and dtype off the example, so this costs no robot
        # time and the first reset below is the first real one.
        example_ob = np.zeros(env.observation_space.shape, dtype=np.float32)
        example_action = np.zeros(env.action_space.shape, dtype=np.float32)
        buffer = ReplayBuffer.create(
            make_transition(example_ob, example_action, example_ob, 0.0), FLAGS.buffer_size
        )

    agent = None
    try:
        # Warmup runs without an agent: `GCDataset` asserts a terminal at its last index, so the
        # buffer needs at least one complete episode before the agent can be built from a batch.
        # GraphNav drives `seed_episodes` of them, which gives better state coverage than random
        # actions and costs nothing, then uniform random actions fill the rest.
        # A restored run skips it, except a mapless one that a pose jump stopped short of warmup.
        do_warmup = FLAGS.restore_path is None or FLAGS.mapless
        while do_warmup and (
            (FLAGS.restore_path is None and episode_idx == 0)
            or episode_idx < FLAGS.seed_episodes
            or buffer.size < FLAGS.warmup_steps
        ):
            use_graphnav = episode_idx < FLAGS.seed_episodes
            task_idx = episode_idx % num_tasks
            ob, info = mapless_reset() if FLAGS.mapless else env.reset(options=dict(task_id=task_idx + 1))
            start_step, ep_steps = env_step, 0
            while True:
                if use_graphnav:
                    next_ob, _, terminated, truncated, info = env.step_graphnav()
                    action = np.array(
                        [info['action_x'], info['action_y'], info['action_w']], dtype=np.float32
                    )
                else:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
                    next_ob, _, terminated, truncated, info = env.step(action)
                env_step += 1
                ep_steps += 1
                store_transition(ob, action, next_ob, float(terminated or truncated), info, ep_steps)
                log_step(info, terminated, truncated)
                ob = next_ob
                if FLAGS.mapless and env_step % FLAGS.log_interval == 0:
                    warmup_metrics = dict(mapless_metrics(), **{'buffer/size': buffer.size})
                    wandb.log(warmup_metrics, step=env_step)
                    warmup_logger.log(warmup_metrics, step=env_step)
                if terminated or truncated:
                    break
            log_episode(task_idx, ep_steps, terminated, truncated, terminated, start_step)

        # -- agent ----------------------------------------------------------------------------
        gc_dataset = dataset_class(buffer, config)
        gc_dataset.rebuild_boundaries()
        agent = agent_class.create(FLAGS.seed, gc_dataset.sample(1, idxs=np.array([0])), config)
        # A pose jump during mapless warmup aborts before any agent exists, so there is nothing to restore.
        if FLAGS.restore_path is not None and (
            not FLAGS.mapless or os.path.exists(os.path.join(candidates[0], f'params_{FLAGS.restore_epoch}.pkl'))
        ):
            agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed))

        # -- training loop ----------------------------------------------------------------------
        first_time = time.time()
        first_act = first_update = True
        last_time = time.time()
        update_info = {}
        task_idx = episode_idx % num_tasks
        # The loop's first goal is picked without V: a resumed session has no observation to value
        # it from.
        ob, info = mapless_reset() if FLAGS.mapless else env.reset(options=dict(task_id=task_idx + 1))
        goal = info['goal']
        start_step, ep_steps = env_step, 0

        remaining = online_step_range(env_step, seeded_steps, FLAGS.online_steps)
        for env_step in tqdm.tqdm(remaining, smoothing=0.1, dynamic_ncols=True):
            # The acting policy is the agent's own sampling. SHARSA's flow actor ignores
            # `temperature`; exploration is the noise the flow starts from, reseeded every step.
            if FLAGS.mapless:
                act_start = time.perf_counter()
                action = np.array(jax.block_until_ready(actor_fn(observations=ob, goals=goal)))
                # The process's first call includes XLA compile time, which is not per-decision latency.
                act_ms = None if first_act else 1000 * (time.perf_counter() - act_start)
                first_act = False
            else:
                action = np.array(actor_fn(observations=ob, goals=goal))
                act_ms = None
            next_ob, _, terminated, truncated, info = env.step(action)
            ep_steps += 1
            store_transition(ob, action, next_ob, float(terminated or truncated), info, ep_steps)
            log_step(info, terminated, truncated, act_ms)
            ob = next_ob

            if buffer.size >= FLAGS.warmup_steps:
                for _ in range(FLAGS.updates_per_step):
                    idxs = np.random.randint(gc_dataset.size, size=config['batch_size'])
                    update_start = time.perf_counter()
                    agent, update_info = agent.update(gc_dataset.sample(config['batch_size'], idxs=idxs))
                    if FLAGS.mapless:
                        jax.block_until_ready((agent, update_info))
                        online.on_update(None if first_update else 1000 * (time.perf_counter() - update_start))
                        first_update = False
                actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(env_step))

            if terminated or truncated:
                log_episode(task_idx, ep_steps, terminated, truncated, terminated, start_step)
                gc_dataset.rebuild_boundaries()
                task_idx = episode_idx % num_tasks
                if FLAGS.mapless:
                    ob, info = mapless_reset(agent, ob)
                else:
                    ob, info = env.reset(options=dict(task_id=task_idx + 1))
                goal = info['goal']
                start_step, ep_steps = env_step, 0

            if env_step % FLAGS.log_interval == 0:
                metrics = {f'training/{k}': v for k, v in update_info.items()}
                metrics.update(episode_metrics(episode_records, num_tasks, FLAGS.eval_episodes))
                metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                metrics['time/total_time'] = time.time() - first_time
                metrics['buffer/size'] = buffer.size
                if FLAGS.mapless:
                    metrics.update(mapless_metrics())
                last_time = time.time()
                wandb.log(metrics, step=env_step)
                train_logger.log(metrics, step=env_step)

            if env_step % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, env_step)
                buffer.save(os.path.join(FLAGS.save_dir, f'buffer_{env_step}.npz'))
                if FLAGS.mapless:
                    save_sampler(env_step)
    finally:
        # The transitions the robot produced cannot be regenerated, so they get written whatever
        # ends the run: the step budget, a crash, or a keyboard interrupt.
        buffer.save(os.path.join(FLAGS.save_dir, f'buffer_{env_step}.npz'))
        if agent is not None:  # a crash or pose jump during warmup comes before the agent exists
            save_agent(agent, FLAGS.save_dir, env_step)
        if FLAGS.mapless:
            save_sampler(env_step)
            # Drift: the run-frame pose read at session end, with Spot back on the start mark.
            try:
                x, y, yaw = env.frame.to_run(*(env.client.get_state()[k] for k in ('x', 'y', 'yaw')))
                session_logger.log(
                    dict(event='end', wall_time=time.time(), x=x, y=y, yaw=yaw, drift_x=x, drift_y=y, drift_yaw=yaw),
                    step=env_step,
                )
            except Exception as exc:  # noqa: BLE001 - the saves above matter more than this row
                print(f'session end pose not logged: {type(exc).__name__}: {exc}')
            session_logger.close()
        train_logger.close()
        step_logger.close()
        episode_logger.close()
        env.close()


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setting = str(FLAGS.agent['agent_name']).split('/')[-1].split('.')[0]
    enable_fk = FLAGS.agent.get('enable_fk_regularization', False)
    fk_key = 'fk_' if enable_fk else 'no_fk_'
    use_viscous_metric = FLAGS.agent.get('enable_viscous_metric', False)
    num_walks = FLAGS.agent.get('num_walks', 10)
    viscous_scale = FLAGS.agent.get('viscous_scale', 0.001)
    if FLAGS.verbose == 1:
        fk_key += f'viscous_{use_viscous_metric}_nwalks_{num_walks}_nuscale_{viscous_scale}_'
        setting = 'verbose_' + setting

    setup_wandb(project='goal_representation', group=FLAGS.run_group, name=setting + '_' + fk_key + FLAGS.env_name + '_' + exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    if FLAGS.online_steps > 0:
        train_online(config)
        return

    # Set up environment and datasets.
    if FLAGS.dataset_dir is None:
        datasets = [None]
    else:
        # Dataset directory.
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    if FLAGS.num_datasets is not None:
        datasets = datasets[: FLAGS.num_datasets]
    dataset_idx = 0
    is_spot = FLAGS.env_name.startswith('spot-')
    if is_spot:
        # A whole on-robot run fits in memory, so the buffers are concatenated once here and the
        # replacement path below is skipped. Evaluating means driving the robot, which the offline
        # loop has no env for: run `--online_steps` against the trained checkpoint instead.
        assert FLAGS.eval_interval == 0, 'offline Spot training has no env to evaluate in; pass --eval_interval=0'
        env = None
        train_dataset, val_dataset = load_spot_datasets(datasets)
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx])

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            val_batch = val_dataset.sample(config['batch_size'])
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders, _ = evaluate(
                    agent=agent,
                    env=env,
                    env_name=FLAGS.env_name,
                    goal_conditioned=True,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success', 'steps']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                # A task with no completions contributes no steps value. Absent, not imputed.
                present = [x for x in v if not np.isnan(x)]
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(present) if len(present) > 0 else np.nan

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=5)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if (
            not is_spot
            and FLAGS.dataset_replace_interval != 0
            and i % FLAGS.dataset_replace_interval == 0
            and len(datasets) > 1
        ):
            dataset_idx = (dataset_idx + 1) % len(datasets)
            train_dataset, val_dataset = make_env_and_datasets(
                FLAGS.env_name, dataset_path=datasets[dataset_idx], dataset_only=True, cur_env=env
            )
            train_dataset = dataset_class(Dataset.create(**train_dataset), config)
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    # No arguments, or --spot_sim alone: the guided Spot session, which launches the arms itself.
    if sys.argv[1:] in ([], ['--spot_sim']):
        from deploy.spot_session import main as session_main

        session_main(sim=sys.argv[1:] == ['--spot_sim'])
    else:
        app.run(main)
