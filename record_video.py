import gymnasium as gym
import imageio
from stable_baselines3 import DQN, PPO

env = gym.make("LunarLander-v3", render_mode="rgb_array")
model = PPO.load(".\models\ppo_lunar_lander")
#model = DQN.load(".\models\dqn_lunar_lander")


num_episodes = 3
all_frames = []

for i in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        all_frames.append(env.render())

    # Add short pause between episodes
    for _ in range(10):
        all_frames.append(all_frames[-1])

#path = "./assets/dqn_lunarlander_run.gif"
path = "./assets/ppo_lunarlander_run.gif"

# Save as combined video
imageio.mimsave(path, all_frames, fps=30)

env.close()
print(f"Saved combined gif to: {path}")