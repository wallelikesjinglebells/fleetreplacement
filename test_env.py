from gymnasium.utils.env_checker import check_env
from fleetreplacement_env.envs.fleet_replacement import FleetReplacementEnv

env = FleetReplacementEnv(render_mode="human")
check_env(env)
print("✓ check_env passed")

obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")
print(f"Initial fleet state:\n{obs}")

total_reward = 0.0
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    print(f"Step {info['step']:>2} | reward: €{reward:>12,.0f} | mean age: {info['mean_age']:.1f}")

print(f"\nEpisode finished. Total reward: €{total_reward:,.0f}")
env.close()
