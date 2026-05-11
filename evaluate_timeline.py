"""
Single-episode visual evaluation of a saved MaskablePPO model.

Loads the best saved model for a given scenario and runs one episode with human rendering so the fleet decisions can be watched step by step.

Usage:
    python evaluate_timeline.py SQ --v0
    python evaluate_timeline.py S1 --v1
    python evaluate_timeline.py S1 --v2
    python evaluate_timeline.py S1 --v2_rt1
"""

import argparse
import re as _re
import fleetreplacement_env
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

_SCENARIO_MAP = {
    "SQ": "Status_Quo",
    "S1": "Scenario_1",
    "S2": "Scenario_2",
    "S3": "Scenario_3",
    "S4": "Scenario_4",
}

parser = argparse.ArgumentParser()
parser.add_argument("scenario", choices=_SCENARIO_MAP, nargs="?", default="SQ")
args, _extra = parser.parse_known_args()

# Detect --vN flag dynamically (e.g. --v0, --v1, --v2, --v2_rt1, ...)
_version_flags = [a for a in _extra if _re.fullmatch(r"--v\d+\w*", a)]
if len(_version_flags) == 0:
    parser.error("A version flag is required (e.g. --v0, --v1, --v2, --v2_rt1, ...)")
if len(_version_flags) > 1:
    parser.error(f"Only one version flag allowed, got: {' '.join(_version_flags)}")
_version = _version_flags[0].lstrip("-")   # "v0", "v1", "v2_rt1", ...

_scenario_tag = args.scenario
MODEL_PATH = f"./models/{_version}/ppo_fleet_{_scenario_tag}/best_model"
print(f"Scenario : {args.scenario}  |  Model : {MODEL_PATH}\n")

# Load the best saved model
model = MaskablePPO.load(MODEL_PATH)

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