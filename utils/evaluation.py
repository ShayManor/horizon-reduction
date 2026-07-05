import atexit
import os
from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def _eval_worker(env_name, conn):
    """Worker process: owns one bare env and steps it on command.

    Kept off the GPU/JAX so N of these only contend for CPU cores (the eval
    bottleneck is MuJoCo env.step, not policy inference).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import ogbench

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == 'reset':
                obs, info = env.reset(options=data)
                conn.send((obs, info.get('goal')))
            elif cmd == 'step':
                obs, _, term, trunc, info = env.step(data)
                conn.send((obs, bool(term or trunc), info))
            else:  # 'close'
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()
        env.close()


# Module-level worker pool, created lazily on first parallel eval and reused for
# the whole run (spawn cost is paid once). Keyed by env_name.
_POOL = {'env_name': None, 'workers': None}
_PARALLEL_DISABLED = os.environ.get('EVAL_PARALLEL', '1') == '0'
# A stalled worker (e.g. GL init hang) must not block the whole job: poll with a
# timeout and raise (-> serial fallback) instead of blocking forever on recv.
_RECV_TIMEOUT = float(os.environ.get('EVAL_RECV_TIMEOUT', '300'))


def _recv(conn):
    if not conn.poll(_RECV_TIMEOUT):
        raise TimeoutError(f'eval worker silent for >{_RECV_TIMEOUT:.0f}s')
    return conn.recv()


def _close_pool():
    workers = _POOL.get('workers')
    if not workers:
        return
    for p, conn in workers:
        try:
            conn.send(('close', None))
        except Exception:
            pass
    for p, conn in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
        conn.close()
    _POOL['workers'] = None


atexit.register(_close_pool)


def _get_pool(env_name, num_workers):
    if _POOL['workers'] is not None and _POOL['env_name'] == env_name:
        return _POOL['workers']
    _close_pool()
    import multiprocessing as mp

    ctx = mp.get_context('spawn')
    workers = []
    for _ in range(num_workers):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_eval_worker, args=(env_name, child), daemon=True)
        p.start()
        child.close()
        workers.append((p, parent))
    _POOL['env_name'] = env_name
    _POOL['workers'] = workers
    return workers


def _evaluate_parallel(
    actor_fn, env_name, task_id, num_eval_episodes, eval_temperature, eval_gaussian
):
    """Run num_eval_episodes concurrently across worker envs.

    Policy inference stays per-episode (single observation) to match the serial
    numerics exactly; only the CPU-bound env.step is parallelized. Raises on any
    failure so the caller can fall back to the serial path.
    """
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    num_workers = min(num_eval_episodes, n_cpus)
    workers = _get_pool(env_name, num_workers)
    stats = defaultdict(list)

    remaining = num_eval_episodes
    while remaining > 0:
        batch = min(num_workers, remaining)
        conns = [workers[j][1] for j in range(batch)]
        for c in conns:
            c.send(('reset', dict(task_id=task_id, render_goal=False)))
        obs = [None] * batch
        goals = [None] * batch
        for k, c in enumerate(conns):
            obs[k], goals[k] = _recv(c)

        active = [True] * batch
        final_info = [None] * batch
        while any(active):
            # Dispatch one step for every still-running episode (envs step in
            # parallel across workers); inference is single-obs per episode.
            for k in range(batch):
                if not active[k]:
                    continue
                action = np.array(actor_fn(observations=obs[k], goals=goals[k], temperature=eval_temperature))
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
                conns[k].send(('step', action))
            for k in range(batch):
                if not active[k]:
                    continue
                obs[k], done, info = _recv(conns[k])
                if done:
                    active[k] = False
                    final_info[k] = info

        for info in final_info:
            add_to(stats, flatten(info))
        remaining -= batch

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    env_name=None,
    goal_conditioned=True,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        env_name: Environment name.
        goal_conditioned: Whether to do goal-conditioned evaluation.
        task_id: Task ID to be passed to the environment (only used when goal_conditioned is True).
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    # Fast path: parallelize the (non-rendered) rollouts across worker envs. Only
    # env.step is parallelized; inference stays per-episode so numerics are
    # identical to the serial loop. Any failure falls back to serial below.
    global _PARALLEL_DISABLED
    if (
        not _PARALLEL_DISABLED
        and goal_conditioned
        and num_video_episodes == 0
        and num_eval_episodes > 1
        and env_name is not None
    ):
        try:
            stats = _evaluate_parallel(
                actor_fn, env_name, task_id, num_eval_episodes, eval_temperature, eval_gaussian
            )
            return stats, [], []
        except Exception as e:
            _PARALLEL_DISABLED = True
            _close_pool()
            print(f'[eval] parallel eval failed ({type(e).__name__}: {e}); falling back to serial for the rest of the run.')
            stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if goal_conditioned:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
        else:
            observation, info = env.reset()
            goal = None
            goal_frame = None
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
