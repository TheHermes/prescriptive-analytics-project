import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Load environment (no rendering during evaluation)
env = Monitor(gym.make("LunarLander-v3"))

# Load your trained model (use correct type)
#ppo_model = PPO.load("./models/ppo_lunar_lander")
#dqn_model = DQN.load("./models/dqn_lunar_lander")

ppo_model = PPO.load("./PPO_TRAINING/logs/best_model")
dqn_model = DQN.load("./DQN_TRAINING/logs/best_model")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    dqn_model,
    env,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

print(f"DQN-Model evaluation over 20 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    ppo_model,
    env,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

print(f"PPO-model evaluation over 20 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()