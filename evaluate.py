import fleetreplacement_env
import gymnasium as gym
from stable_baselines3 import PPO

# Load the best saved model
model = PPO.load("./models/ppo_fleet/best_model")

env = gym.make("FleetReplacement-v0", render_mode="human")
# obs, info = env.reset(seed=0)                                 # uncomment if fleet should stay the same
obs, info = env.reset()                                         # random fleet composition
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"\nEpisode total reward: €{total_reward:,.0f}")
env.close()