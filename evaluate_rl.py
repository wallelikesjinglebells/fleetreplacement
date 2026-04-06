"""
RL model evaluation over multiple episodes

Loads the best saved MaskablePPO model for a given scenario and evaluates it over N episodes using the same seed sequence as evaluate_heuristics.py → results are directly comparable

Usage:
    python evaluate_rl.py SQ              # Status Quo (default)
    python evaluate_rl.py S1              # Scenario 1, etc.
    python evaluate_rl.py SQ --trace      # also print step-by-step for one episode
    python evaluate_rl.py SQ --episodes 50 --seed 42
"""

import argparse
import numpy as np
import gymnasium as gym
import fleetreplacement_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_SCENARIO_MAP = {
    "SQ": "Status Quo",
    "S1": "Scenario 1: Tech Stalemate",
    "S2": "Scenario 2: Tech without Mandate",
    "S3": "Scenario 3: Ambition meets Reality",
    "S4": "Scenario 4: Autonomous Green Logistics",
}

parser = argparse.ArgumentParser()
parser.add_argument("scenario", choices=_SCENARIO_MAP, nargs="?", default="SQ")
parser.add_argument("--episodes", type=int, default=50,
                    help="Number of episodes (default: 50)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed (default: 42)")
parser.add_argument("--trace", action="store_true",
                    help="Print step-by-step output for the first episode")
args = parser.parse_args()

SCENARIO_NAME  = _SCENARIO_MAP[args.scenario]
N_EPISODES     = args.episodes
BASE_SEED      = args.seed
_scenario_tag  = args.scenario
MODEL_PATH     = f"./models/ppo_fleet_{_scenario_tag}/best_model"

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(render_mode=None):
    cfg = FleetEnvConfig(mdp=MDPConfig(), cost=load_cost_config(scenario_name=SCENARIO_NAME))
    env = gym.make("FleetReplacement-v0", config=cfg, render_mode=render_mode)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"\nScenario : {SCENARIO_NAME}")
print(f"Model    : {MODEL_PATH}")
print(f"Episodes : {N_EPISODES}  |  Base seed : {BASE_SEED}\n")

model = MaskablePPO.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
env = make_env()
rewards = []

for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=BASE_SEED + ep)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    rewards.append(total_reward)

env.close()
rewards = np.array(rewards)

# ---------------------------------------------------------------------------
# Results — same format as evaluate_heuristics.py
# ---------------------------------------------------------------------------
print(f"{'Model':<22} {'Mean (EUR)':>15} {'Std':>12} {'Best':>15} {'Worst':>15}")
print("-" * 82)
print(
    f"{'RL best_model':<22}"
    f"  {np.mean(rewards):>14,.0f}"
    f"  {np.std(rewards):>11,.0f}"
    f"  {np.max(rewards):>14,.0f}"
    f"  {np.min(rewards):>14,.0f}"
)
print("-" * 82)

# ---------------------------------------------------------------------------
# Optional: step-by-step trace for the first episode
# ---------------------------------------------------------------------------
if args.trace:
    print(f"\n{'='*60}")
    print(f"Step-by-step trace  (seed={BASE_SEED})")
    print(f"{'='*60}")

    env = make_env(render_mode="human")
    obs, _ = env.reset(seed=BASE_SEED)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"\nEpisode total reward: EUR {total_reward:,.0f}")
    env.close()
