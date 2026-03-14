import gymnasium as gym
from stable_baselines3 import PPO
from env import InvertedPendulumEnv

# 1. Load the environment with "human" rendering so we can see it
env = InvertedPendulumEnv(render_mode="human")

# 2. Load the trained agent
#    Note: We don't need to pass the env to load(), but we need the env for prediction
model = PPO.load("ppo_inverted_pendulum",env=env)

# 3. Play a few episodes
obs, info = env.reset()
print("Running trained agent...")

while True:
    # Ask the agent: "Given this observation, what action should I take?"
    # deterministic=True means "give me your best move", no randomness.
    action, _states = model.predict(obs, deterministic=True)
    
    # Execute the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # If the episode ends, reset
    if terminated or truncated:
        obs, info = env.reset()