import ogbench
env, train_dataset, _ = ogbench.make_env_and_datasets('puzzle-4x5-play-oraclerep-v0', dataset_only=False)

D = env.observation_space.shape[0]
A = env.action_space.shape[0]
G = D # In standard goal-conditioned RL, the goal space matches the observation space

print(f"B = 256  # Standard batch size")
print(f"D = {D}")
print(f"A = {A}")
print(f"G = {G}")