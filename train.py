import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_checker import check_env
from env import InvertedPendulumEnv
env = InvertedPendulumEnv()
env = TimeLimit(env, max_episode_steps=1000)
print("Checking custom environment...")
check_env(env)
print("Environment check passed!")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_inverted_pendulum_tensorboard/")

print("staring tarining...")
model.learn(total_timesteps=300000, tb_log_name="first_run")
print('Training completed')
model.save("ppo_inverted_pendulum")
print("Model saved as ppo_inverted_pendulum.zip")