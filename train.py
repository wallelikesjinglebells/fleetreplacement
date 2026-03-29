import fleetreplacement_env  # triggers register() in __init__.py, is important for gym.make() although flagged as not accessed
import gymnasium as gym
from stable_baselines3 import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker                        
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env import sync_envs_normalization


# Configuration variables, tunable settings
ENV_ID        = "FleetReplacement-v0"
TOTAL_STEPS   = 5_000_000   # total number of environment steps to train (increase for real training)
N_ENVS        = 4         # parallel environments for faster data collection
EVAL_FREQ     = 10_000    # pause training every EVAL_FREQ steps to evaluate current policy on eval_env
LOG_DIR       = "./logs/"
SAVE_PATH     = "./models/ppo_fleet"

# Wrap env with ActionMasker
def make_masked_env():
    env = gym.make(ENV_ID)
    env = ActionMasker(env, lambda e: e.action_masks())  # tells SB3 where to find masks
    return env

# Create vectorized environment for all N_ENVS environments
vec_env = make_vec_env(make_masked_env, n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)       # normalize reward

# Single evaluation environment
_monitor_env = Monitor(make_masked_env(), filename="./logs/eval_monitor")        # Monitor wrapper to fix warning if other wrappers are present
eval_env = DummyVecEnv([lambda: _monitor_env])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True,
                        training=False, gamma=0.99)
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
    gamma=0.99,
    tensorboard_log=LOG_DIR,
    ent_coef=0.01,          # entropy coefficient for loss calculation (PPO is rewarded for exploring all actions)
)

# Train
model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model = MaskablePPO.load(f"{SAVE_PATH}/best_model")                                             # reload best model

# Save final model at end of training
model.save(f"{SAVE_PATH}_final")
print(f"Training complete. Model saved to {SAVE_PATH}_final")
