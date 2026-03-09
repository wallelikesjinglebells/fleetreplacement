import fleetreplacement_env  # triggers register() in __init__.py, is important for gym.make() although flagged as not accessed
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback

# Configuration variables, tunable settings
ENV_ID        = "FleetReplacement-v0"
TOTAL_STEPS   = 1_500_000   # total number of environment steps to train (increase for real training)
N_ENVS        = 4         # parallel environments for faster data collection
EVAL_FREQ     = 10_000    # pause training every EVAL_FREQ steps to evaluate current policy on eval_env
LOG_DIR       = "./logs/"
SAVE_PATH     = "./models/ppo_fleet"

# Create vectorized environment for all N_ENVS environments
vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)

# Single evaluation environment
eval_env = Monitor(gym.make(ENV_ID), filename="./logs/eval_monitor")        # Monitor wrapper to fix warning if other wrappers are present

# Callback: every EVAL_FREQ, run current policy on eval_env, record mean reward, if better than previous best 
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_PATH,
    log_path=LOG_DIR,
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),      # SB3 counts steps per individual environment → divide EVAL_FREQ by no. of environments
    deterministic=True,                         # greedy, pick action with highest probability instead of sampling (like in training)
    render=False,
)

# PPO agent
model = PPO(
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
model = PPO.load(f"{SAVE_PATH}/best_model")                                             # reload best model

# Save final model at end of training
model.save(f"{SAVE_PATH}_final")
print(f"Training complete. Model saved to {SAVE_PATH}_final")

# Sanity check :) run one full episode with trained policy
env = gym.make(ENV_ID, render_mode="human")
# obs, info = env.reset(seed=0)                                 # uncomment if fleet should stay the same
obs, info = env.reset()                                         # random fleet composition
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)              # pick greedy action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Evaluation episode total reward: €{total_reward:,.0f}")
env.close()
