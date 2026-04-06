"""
Single-episode visual evaluation of a saved MaskablePPO model.

Loads the best saved model for the Status Quo scenario and runs one episode with human rendering so the fleet decisions can be watched step by step.

Usage:
    python evaluate.py
"""

import fleetreplacement_env
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Load the best saved model
model = MaskablePPO.load("./models/ppo_fleet_Status_Quo/best_model")

env = gym.make("FleetReplacement-v0", render_mode="human")
env = ActionMasker(env, lambda e: e.unwrapped.action_masks())

# obs, info = env.reset(seed=0)                                 # uncomment if fleet should stay the same
obs, info = env.reset()                                         # random fleet composition
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"\nEpisode total reward: EUR {total_reward:,.0f}")
env.close()