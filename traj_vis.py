import os
import pickle
import json
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use headless backend for saving plots on a cluster
import matplotlib.pyplot as plt
import flax
from absl import app, flags
from ml_collections import ConfigDict

# Set headless rendering for Mujoco to avoid X11/GLFW errors
os.environ['MUJOCO_GL'] = 'egl' 

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.evaluation import supply_rng
from utils.datasets import Dataset, GCDataset, HGCDataset

# Define paths to your specific job
EXP_DIR = "exp/long-horizon-results-replication/puzzle_experiments_4seed_sharsa_fk/sd003_s_11149735.0.20260629_072415" # UPDATE THIS TO YOUR RUN DIRECTORY
CHECKPOINT_PATH = os.path.join(EXP_DIR, "params_5000000.pkl") # UPDATE EPOCH IF NEEDED
FLAGS_PATH = os.path.join(EXP_DIR, "flags.json")

def load_agent_and_env():
    with open(FLAGS_PATH, 'r') as f:
        config_dict = json.load(f)
    
    config = ConfigDict(config_dict['agent'])
    env_name = config_dict['env_name']
    
    env, train_dataset_dict, _ = make_env_and_datasets(env_name, dataset_path=None)
    
    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }
    dataset_class = dataset_class_dict.get(config.get('dataset_class', 'GCDataset'), GCDataset)
    
    train_dataset = dataset_class(Dataset.create(**train_dataset_dict), config)
    example_batch = train_dataset.sample(1)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(seed=0, example_batch=example_batch, config=config)
    
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH, 'rb') as f:
        save_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, save_dict['agent'])
    
    return agent, env

def collect_trajectory_until_success(agent, env, task_id=1, max_attempts=15):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(42))

    for attempt in range(max_attempts):
        observation, info = env.reset(options=dict(task_id=task_id))
        goal = info.get('goal')
        
        ee_coords = []
        state_coords = []
        done = False
        
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=0.0)
            action = np.clip(np.array(action), -1, 1)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 1. Arm trajectory (first 3 elements are XYZ spatial coordinates)
            ee_coords.append(observation[:3])
            
            # 2. Puzzle state (the last N elements match the N elements of the goal)
            if goal is not None:
                state_coords.append(observation[-len(goal):])
            
            observation = next_observation
            
        success = info.get('success', False)
        
        if success:
            print(f"Success found on attempt {attempt + 1}!")
            return np.array(ee_coords), np.array(state_coords), goal
            
        print(f"Attempt {attempt + 1} failed. Retrying...")

    print("Warning: Could not find a successful trajectory in the given attempts.")
    return np.array(ee_coords), np.array(state_coords), goal

def plot_combined_trajectories(ee_traj, obs_traj, goal_obs, save_path="trajectory_plot_combined.png"):
    fig = plt.figure(figsize=(16, 8))
    
    # --- SUBPLOT 1: End-Effector (3D) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 
             label='Arm Trajectory', color='purple', linewidth=3, marker='o', markersize=3)
    ax1.scatter(ee_traj[0, 0], ee_traj[0, 1], ee_traj[0, 2], 
                color='black', s=100, label='Start Position', zorder=5)
    
    ax1.set_title('End-Effector Trajectory', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # --- SUBPLOT 2: Puzzle State Convergence (2D) ---
    ax2 = fig.add_subplot(122)
    
    if goal_obs is not None:
        # Now both obs_traj and goal_obs have 20 dimensions and can be safely compared!
        distances = np.mean((obs_traj - goal_obs)**2, axis=1)
        
        ax2.plot(range(len(distances)), distances, 
                 label='State Distance to Goal', color='teal', linewidth=3)
        ax2.scatter(0, distances[0], color='black', s=100, label='Start State', zorder=5)
        ax2.scatter(len(distances)-1, distances[-1], color='gold', s=200, marker='*', label='Final State', zorder=5)
            
    ax2.set_title('Puzzle State Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Mean Squared Error (vs Target)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    agent, env = load_agent_and_env()
    ee_traj, obs_traj, goal_obs = collect_trajectory_until_success(agent, env)
    plot_combined_trajectories(ee_traj, obs_traj, goal_obs)