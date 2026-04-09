"""
MaskablePPO training for fleet replacement.

Trains a MaskablePPO agent on the FleetReplacement-v0 environment for a given scenario, periodically evaluates the current policy, and saves the best and final models.

Usage:
    python train.py SQ                  # Status Quo (default)
    python train.py S1                  # Scenario 1, etc.
    python train.py Sx --writetooold    # overwrite previous training logs/models
"""

import argparse
import os
import re
import fleetreplacement_env  # triggers register() in __init__.py, is important for gym.make() although flagged as not accessed
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker                        
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env import sync_envs_normalization
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config


_SCENARIO_MAP = {
    "SQ": "Status Quo",
    "S1": "Scenario 1: Tech Stalemate",
    "S2": "Scenario 2: Tech without Mandate",
    "S3": "Scenario 3: Ambition meets Reality",
    "S4": "Scenario 4: Autonomous Green Logistics",
}

# S1 has no BET adoption (break-even > 80 yrs); its only learnable signal is DT lifecycle
# timing — a narrow decision space that converges cleanly with low entropy.
# S2/S3/S4 need higher entropy to explore the BET timing / subsidy expiry trade-offs.
_ENTROPY_COEF = {
    "SQ": 0.05,
    "S1": 0.01,
    "S2": 0.05,
    "S3": 0.05,
    "S4": 0.05,
}
parser = argparse.ArgumentParser()
parser.add_argument("scenario", choices=_SCENARIO_MAP, help="Scenario key: SQ, S1, S2, S3, S4")
parser.add_argument("--writetooold", action="store_true",
                    help="Write into the latest existing version folder (e.g. v1) instead of creating a new one (e.g. v2)")
args = parser.parse_args()

# Detect next (or current) version by scanning logs/ and models/ for v<N> folders
def _detect_version(logs_root, models_root, write_to_old):
    existing = set()
    for root in (logs_root, models_root):
        if os.path.isdir(root):
            for name in os.listdir(root):
                if re.fullmatch(r"v\d+", name):
                    existing.add(int(name[1:]))
    if not existing:
        return 0
    return max(existing) if write_to_old else max(existing) + 1

# Configuration variables, tunable settings
SCENARIO_NAME = _SCENARIO_MAP[args.scenario]
ENV_ID        = "FleetReplacement-v0"
TOTAL_STEPS   = 5_000_000             # total number of environment steps to train
N_ENVS        = 4                     # parallel environments for faster data collection
EVAL_FREQ     = 10_000                # pause training every EVAL_FREQ steps to evaluate current policy on eval_env
_scenario_tag = args.scenario
_version      = _detect_version("./logs", "./models", args.writetooold)
LOG_DIR       = f"./logs/v{_version}/{_scenario_tag}/"
SAVE_PATH     = f"./models/v{_version}/ppo_fleet_{_scenario_tag}"
print(f"[train] Using version v{_version} ({'overwriting' if args.writetooold else 'new'}): logs → {LOG_DIR}, models → {SAVE_PATH}")

# Wrap env with ActionMasker, loading the correct scenario config
def make_masked_env():
    cfg = FleetEnvConfig(mdp=MDPConfig(), cost=load_cost_config(scenario_name=SCENARIO_NAME))
    env = gym.make(ENV_ID, config=cfg)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env

# Create vectorized environment for all N_ENVS environments
vec_env = make_vec_env(make_masked_env, n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=1.0)       # gamma=1.0: env already applies economic discounting in reward

# Single evaluation environment
_monitor_env = Monitor(make_masked_env(), filename="./logs/eval_monitor")        # Monitor wrapper to fix warning if other wrappers are present
eval_env = DummyVecEnv([lambda: _monitor_env])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True,
                        training=False, gamma=1.0)
# Sync stats from training env
sync_envs_normalization(vec_env, eval_env)

# Callback: every EVAL_FREQ, run current policy on eval_env, record mean reward, if better than previous best 
eval_callback = MaskableEvalCallback(
    eval_env,
    best_model_save_path=SAVE_PATH,
    log_path=LOG_DIR,
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),      # SB3 counts steps per individual environment → divide EVAL_FREQ by no. of environments
    deterministic=True,                         # greedy, pick action with highest probability instead of sampling (like in training)
    render=False,
)

# PPO agent
model = MaskablePPO(
    policy="MlpPolicy",     # standard MLP for flat/matrix observations
    env=vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,           # no. of steps each env collects before policy update, results in total n_steps x n_envs number of transitions
    batch_size=64,
    n_epochs=10,
    gamma=1.0,              # env already applies economic discounting in reward, no additional discounting needed
    tensorboard_log=LOG_DIR,
    ent_coef=_ENTROPY_COEF[_scenario_tag],  # entropy coefficient for loss calculation (PPO is rewarded for exploring all actions)
    policy_kwargs=dict(net_arch=[256, 256]),  # larger network than default [64, 64] for 23-dim obs and 3^10 action space
)

# Train
model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model = MaskablePPO.load(f"{SAVE_PATH}/best_model")                                             # reload best model

# Save final model at end of training
model.save(f"{SAVE_PATH}_final")
print(f"Training complete. Model saved to {SAVE_PATH}_final")
