from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


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
        A tuple containing the statistics, trajectories, rendered videos, and per-episode records.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    episodes = []
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
            episodes.append(
                dict(
                    success=float(flatten(info).get('success', 0.0)),
                    steps=step,
                    terminated=bool(terminated),
                )
            )
        else:
            renders.append(np.array(render))

    # The Spot env reports waypoint ids and SDK error strings alongside the numeric fields, so
    # only the numeric ones average. OGBench envs report numbers throughout and are unaffected.
    stats = {k: np.mean(v) for k, v in stats.items() if np.asarray(v).dtype.kind in 'biufc'}

    # Steps to completion over completed episodes only: a truncated episode ends at
    # `max_episode_steps`, which describes the timeout and not the policy. Timeouts are excluded
    # rather than censored, so a task with no completions has no steps value.
    completed_steps = [ep['steps'] for ep in episodes if ep['success'] > 0]
    stats['steps'] = np.mean(completed_steps) if len(completed_steps) > 0 else np.nan

    return stats, trajs, renders, episodes
