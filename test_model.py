import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import imageio
import numpy as np

env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
#model = PPO.load("./models/ppo_lunar_lander")
#model = DQN.load("./models/dqn_lunar_lander")

#model = PPO.load("./PPO_TRAINING/logs/best_model")
model = DQN.load("./DQN_TRAINING/logs/best_model")

mean, std = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
print(f"Performance: {mean:.1f} ± {std:.1f}")

# Watch one episode
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
env.close()
