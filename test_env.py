from gymnasium.utils.env_checker import check_env
from fleetreplacement_env.envs.fleet_replacement import FleetReplacementEnv

# Validation without rendering
env = FleetReplacementEnv()
check_env(env, warn=True)
env.close()
print("✓ check_env passed\n")

# Demonstrate training episode
env = FleetReplacementEnv(render_mode="human")

obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")
print(f"Initial fleet state:\n{obs}")

total_reward = 0.0
done = False

while not done:
    action = env.action_space.sample()      # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    print(f"Step {info['step']:>2} | reward: €{reward:>12,.0f} | mean age: {info['mean_age']:.1f}")

print(f"\nEpisode finished. Total reward: €{total_reward:,.0f}")
env.close()
