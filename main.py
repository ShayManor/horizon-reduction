import glob
import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets, make_online_env
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer
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
    env = make_online_env(
        FLAGS.env_name,
        robot_hostname=FLAGS.robot_hostname,
        graphnav_map_path=FLAGS.graphnav_map_path,
        task_ids=task_ids,
        control_hz=FLAGS.control_hz,
        max_episode_steps=FLAGS.max_episode_steps,
        goal_tol=FLAGS.goal_tol,
        vel_limits=tuple(float(v) for v in FLAGS.vel_limits),
        reverse_limit=FLAGS.reverse_limit,
    )
    tasks = env.task_infos
    num_tasks = len(tasks)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    step_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'steps.csv'))
    episode_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'episodes.csv'))

    episode_records = []
    episode_idx = 0
    env_step = 0

    def log_episode(task_idx, steps, terminated, truncated, success, start_step):
        nonlocal episode_idx
        task = tasks[task_idx]
        record = dict(
            episode=episode_idx,
            task_id=task_idx + 1,
            task_name=task['task_name'],
            start_waypoint=task['start'],
            goal_waypoint=task['goal'],
            steps=steps,
            terminated=int(terminated),
            truncated=int(truncated),
            success=float(success),
            start_env_step=start_step,
        )
        episode_records.append(record)
        episode_logger.log(dict(record), step=env_step)
        episode_idx += 1

    def log_step(info, terminated, truncated):
        row = {k: v for k, v in info.items() if k != 'goal'}
        row['episode'] = episode_idx
        row['terminated'] = int(terminated)
        row['truncated'] = int(truncated)
        step_logger.log(row, step=env_step)

    # -- buffer: restored, or created and filled by warmup episodes ---------------------------
    if FLAGS.restore_path is not None:
        candidates = glob.glob(FLAGS.restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
        buffer = ReplayBuffer.load(
            os.path.join(candidates[0], f'buffer_{FLAGS.restore_epoch}.npz'), FLAGS.buffer_size
        )
        env_step = buffer.size
        # A crash lands mid-episode. The run stopped at that index, so it is a real trajectory
        # boundary; marking it keeps the partial episode instead of discarding its transitions.
        buffer['terminals'][buffer.size - 1] = 1.0
        print(f'Restored {buffer.size} transitions from {candidates[0]}')
    else:
        # `ReplayBuffer.create` reads only shape and dtype off the example, so this costs no robot
        # time and the first reset below is the first real one.
        example_ob = np.zeros(env.observation_space.shape, dtype=np.float32)
        example_action = np.zeros(env.action_space.shape, dtype=np.float32)
        buffer = ReplayBuffer.create(
            make_transition(example_ob, example_action, example_ob, 0.0), FLAGS.buffer_size
        )

        # Warmup runs without an agent: `GCDataset` asserts a terminal at its last index, so the
        # buffer needs at least one complete episode before the agent can be built from a batch.
        # GraphNav drives `seed_episodes` of them, which gives better state coverage than random
        # actions and costs nothing, then uniform random actions fill the rest.
        while episode_idx == 0 or episode_idx < FLAGS.seed_episodes or buffer.size < FLAGS.warmup_steps:
            use_graphnav = episode_idx < FLAGS.seed_episodes
            task_idx = episode_idx % num_tasks
            ob, info = env.reset(options=dict(task_id=task_idx + 1))
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
                buffer.add_transition(
                    make_transition(ob, action, next_ob, float(terminated or truncated))
                )
                log_step(info, terminated, truncated)
                ob = next_ob
                if terminated or truncated:
                    break
            log_episode(task_idx, ep_steps, terminated, truncated, terminated, start_step)

    # -- agent --------------------------------------------------------------------------------
    gc_dataset = dataset_class(buffer, config)
    gc_dataset.rebuild_boundaries()
    agent = agent_class.create(FLAGS.seed, gc_dataset.sample(1, idxs=np.array([0])), config)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed))

    # -- training loop --------------------------------------------------------------------------
    first_time = time.time()
    last_time = time.time()
    update_info = {}
    task_idx = episode_idx % num_tasks
    ob, info = env.reset(options=dict(task_id=task_idx + 1))
    goal = info['goal']
    start_step, ep_steps = env_step, 0

    remaining = range(env_step + 1, FLAGS.online_steps + 1)
    try:
        for env_step in tqdm.tqdm(remaining, smoothing=0.1, dynamic_ncols=True):
            # The acting policy is the agent's own sampling. SHARSA's flow actor ignores
            # `temperature`; exploration is the noise the flow starts from, reseeded every step.
            action = np.array(actor_fn(observations=ob, goals=goal))
            next_ob, _, terminated, truncated, info = env.step(action)
            ep_steps += 1
            buffer.add_transition(make_transition(ob, action, next_ob, float(terminated or truncated)))
            log_step(info, terminated, truncated)
            ob = next_ob

            if buffer.size >= FLAGS.warmup_steps:
                for _ in range(FLAGS.updates_per_step):
                    idxs = np.random.randint(gc_dataset.size, size=config['batch_size'])
                    agent, update_info = agent.update(gc_dataset.sample(config['batch_size'], idxs=idxs))
                actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(env_step))

            if terminated or truncated:
                log_episode(task_idx, ep_steps, terminated, truncated, terminated, start_step)
                gc_dataset.rebuild_boundaries()
                task_idx = episode_idx % num_tasks
                ob, info = env.reset(options=dict(task_id=task_idx + 1))
                goal = info['goal']
                start_step, ep_steps = env_step, 0

            if env_step % FLAGS.log_interval == 0:
                metrics = {f'training/{k}': v for k, v in update_info.items()}
                metrics.update(episode_metrics(episode_records, num_tasks, FLAGS.eval_episodes))
                metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                metrics['time/total_time'] = time.time() - first_time
                metrics['buffer/size'] = buffer.size
                last_time = time.time()
                wandb.log(metrics, step=env_step)
                train_logger.log(metrics, step=env_step)

            if env_step % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, env_step)
                buffer.save(os.path.join(FLAGS.save_dir, f'buffer_{env_step}.npz'))
    finally:
        # The transitions the robot produced cannot be regenerated, so they get written whatever
        # ends the run: the step budget, a crash, or a keyboard interrupt.
        buffer.save(os.path.join(FLAGS.save_dir, f'buffer_{env_step}.npz'))
        save_agent(agent, FLAGS.save_dir, env_step)
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

        if FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0 and len(datasets) > 1:
            dataset_idx = (dataset_idx + 1) % len(datasets)
            train_dataset, val_dataset = make_env_and_datasets(
                FLAGS.env_name, dataset_path=datasets[dataset_idx], dataset_only=True, cur_env=env
            )
            train_dataset = dataset_class(Dataset.create(**train_dataset), config)
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
