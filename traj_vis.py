"""Render end-effector trajectory overlays for the OGBench manipulation envs.

Rolls the restored policy out on each task and draws the gripper path as green
line geoms inside the MuJoCo scene, then renders one frame per task. One image
per (method, task); the 2-D grid is assembled afterwards.
"""

import glob
import os
import random
from pathlib import Path

import imageio
import mujoco
import numpy as np

import jax
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, HGCDataset
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'puzzle-4x6-play-oraclerep-v0', 'Environment name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets.')
flags.DEFINE_string('save_dir', 'plots/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch (None = auto-detect latest).')
flags.DEFINE_string('title', None, 'Filename prefix (default: agent_name from config).')
flags.DEFINE_string('tasks', None, 'Comma-separated task ids (default: every task).')
flags.DEFINE_integer('num_rollouts', 1, 'Rollouts overlaid on one panel.')
flags.DEFINE_string('ee_body', 'ur5e/robotiq/base', 'Body whose path is traced.')
flags.DEFINE_float('ee_offset', 0.1, 'Offset along the body z-axis, in meters.')
flags.DEFINE_float('min_step', 0.005, 'Minimum movement before a point is kept.')
flags.DEFINE_integer('width', 640, 'Render width.')
flags.DEFINE_integer('height', 480, 'Render height.')
flags.DEFINE_float('line_width', 2.0, 'Trajectory line width, in pixels.')
flags.DEFINE_float('cam_azimuth', None, 'Free-camera azimuth (None = model default).')
flags.DEFINE_float('cam_elevation', None, 'Free-camera elevation.')
flags.DEFINE_float('cam_distance', None, 'Free-camera distance.')
flags.DEFINE_string('cam_lookat', None, 'Free-camera lookat as "x,y,z".')

# Dummy flags expected by main.py-style configs
flags.DEFINE_integer('offline_steps', 0, '')
flags.DEFINE_integer('log_interval', 10000, '')
flags.DEFINE_integer('eval_interval', 0, '')
flags.DEFINE_integer('save_interval', 0, '')
flags.DEFINE_integer('eval_episodes', 0, '')
flags.DEFINE_float('eval_temperature', 0, '')
flags.DEFINE_float('eval_gaussian', None, '')
flags.DEFINE_integer('video_episodes', 0, '')
flags.DEFINE_integer('video_frame_skip', 3, '')

config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)

TRAJ_RGBA = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)


# ──────────────────────────────────────────────
# Rollout + overlay
# ──────────────────────────────────────────────
def rollout(env, actor_fn, data, ee_id, task_id):
    """Run one episode, returning the end-effector path and the final info dict."""
    observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get('goal')
    ee_traj = []
    done = False
    while not done:
        action = np.array(actor_fn(observations=observation, goals=goal, temperature=0))
        action = np.clip(action, -1, 1)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Push the gripper base out along its approach axis so the trace sits at
        # the fingertips rather than inside the wrist.
        rot = data.body(ee_id).xmat.copy().reshape(3, 3)
        point = data.body(ee_id).xpos.copy() + FLAGS.ee_offset * rot[:, 2]
        if not ee_traj or np.linalg.norm(point - ee_traj[-1]) > FLAGS.min_step:
            ee_traj.append(point)

    return ee_traj, info


def draw_traj(scene, ee_traj):
    """Push one line geom per trajectory segment into the render scene."""
    for i in range(len(ee_traj) - 1):
        if scene.ngeom >= scene.maxgeom:
            print('Scene geom buffer full; trajectory truncated.')
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), TRAJ_RGBA
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, FLAGS.line_width, ee_traj[i], ee_traj[i + 1]
        )
        geom.rgba = TRAJ_RGBA
        scene.ngeom += 1


def make_camera(model, data):
    """Free camera framed on the scene, with optional per-axis overrides."""
    if all(f is None for f in (FLAGS.cam_azimuth, FLAGS.cam_elevation,
                               FLAGS.cam_distance, FLAGS.cam_lookat)):
        return None
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    if FLAGS.cam_azimuth is not None:
        camera.azimuth = FLAGS.cam_azimuth
    if FLAGS.cam_elevation is not None:
        camera.elevation = FLAGS.cam_elevation
    if FLAGS.cam_distance is not None:
        camera.distance = FLAGS.cam_distance
    if FLAGS.cam_lookat is not None:
        camera.lookat[:] = [float(v) for v in FLAGS.cam_lookat.split(',')]
    return camera


def vis_trajectories(env, agent, config):
    model = env.unwrapped.model
    data = env.unwrapped.data

    # The manipulation XMLs ship a 200x200 offscreen framebuffer, which is
    # smaller than anything worth putting in a figure.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, FLAGS.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, FLAGS.height)
    renderer = mujoco.Renderer(model, height=FLAGS.height, width=FLAGS.width)

    ee_id = model.body(FLAGS.ee_body).id
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed))

    task_infos = env.unwrapped.task_infos
    if FLAGS.tasks:
        task_ids = [int(t) for t in FLAGS.tasks.split(',')]
    else:
        task_ids = list(range(1, len(task_infos) + 1))

    title = FLAGS.title if FLAGS.title else config.get('agent_name', 'sharsa')
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        trajs = []
        for _ in range(FLAGS.num_rollouts):
            ee_traj, info = rollout(env, actor_fn, data, ee_id, task_id)
            trajs.append(ee_traj)

        # The backdrop is the physics state the last rollout ended in.
        camera = make_camera(model, data)
        renderer.update_scene(data) if camera is None else renderer.update_scene(data, camera)
        for ee_traj in trajs:
            draw_traj(renderer.scene, ee_traj)
        image = renderer.render()

        success = int(info.get('success', 0))
        task_name = task_infos[task_id - 1]['task_name']
        save_path = save_dir / f'{title}_task{task_id}_success{success}.png'
        imageio.imwrite(save_path, image)
        print(f'{task_name}: {sum(len(t) for t in trajs)} points, success={success} -> {save_path}')


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main(_):
    config = FLAGS.agent

    # ── Load env + dataset (one shard) ──
    if FLAGS.dataset_dir is None:
        datasets = [None]
    else:
        datasets = [f for f in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz'))
                    if '-val.npz' not in f]
    dataset_path = datasets[0]

    env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path,
    )

    # ── Build HGCDataset for agent creation ──
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset = HGCDataset(Dataset.create(**train_dataset_raw), config)
    example_batch = train_dataset.sample(1)

    # ── Create + restore agent ──
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch, config)

    # Resolve restore path: pick the run with the highest checkpoint epoch.
    restore_epoch = FLAGS.restore_epoch
    candidates = glob.glob(FLAGS.restore_path)
    assert candidates, f'No candidates found for {FLAGS.restore_path}'

    best_path = None
    best_epoch = -1
    for cand in candidates:
        pkl_files = glob.glob(os.path.join(cand, 'params_*.pkl'))
        if not pkl_files:
            continue
        epochs = [int(os.path.basename(f).replace('params_', '').replace('.pkl', '')) for f in pkl_files]
        if restore_epoch is not None and restore_epoch not in epochs:
            continue
        top = max(epochs)
        if top > best_epoch:
            best_epoch = top
            best_path = cand

    if restore_epoch is not None:
        assert best_path is not None, (
            f'No run under {FLAGS.restore_path} contains params_{restore_epoch}.pkl')
    else:
        assert best_path is not None, f'No params_*.pkl found in any candidate: {candidates}'
    print(f'Selected run: {best_path} (max epoch {best_epoch})')

    if restore_epoch is None:
        restore_epoch = best_epoch
        print(f'Auto-detected latest checkpoint: epoch {restore_epoch}')

    agent = restore_agent(agent, best_path, restore_epoch)

    # ── Visualize ──
    vis_trajectories(env, agent, config)
    print('Done.')


if __name__ == '__main__':
    app.run(main)
