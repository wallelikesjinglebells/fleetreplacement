"""
Modal.com training script for fleet replacement MaskablePPO.

Usage:
    modal run modal_train.py                    # train Status Quo (default)
    modal run modal_train.py --scenario S1      # train a specific scenario
    modal run modal_train.py --scenario all     # train all 5 scenarios in parallel

Prerequisites:
    pip install modal
    modal token new    # authenticate once in your browser

Outputs are saved to a Modal Volume named "fleet-models".
Download results after training:
    modal volume get fleet-models /outputs/models ./models
"""

import modal

app = modal.App("fleet-replacement-training")

# --- Image ---
# All pip dependencies + the local fleetreplacement_env package baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium",
        "pygame>=2.1.3",
        "stable-baselines3",
        "sb3-contrib",
        "pandas",
        "shimmy",           # gymnasium compatibility shim required by SB3
        "tensorboard",
        "rich",             # enables progress bar in model.learn()
        "tqdm",
    )
    .env({"SDL_VIDEODRIVER": "dummy",                 # prevents pygame display errors in headless container
          "PYGAME_HIDE_SUPPORT_PROMPT": "1",
          "PYTHONUNBUFFERED": "1"})                   # stream stdout immediately instead of buffering
    .add_local_python_source("fleetreplacement_env")  # copies local package into image (must be last)
    .add_local_dir("data", remote_path="/root/data")  # bakes local data/ CSVs into image (must be last)
)

# --- Volume for persisting outputs ---
volume = modal.Volume.from_name("fleet-models", create_if_missing=True)
VOLUME_PATH = "/outputs"

SCENARIO_MAP = {
    "SQ": "Status Quo",
    "S1": "Scenario 1: Tech Stalemate",
    "S2": "Scenario 2: Tech without Mandate",
    "S3": "Scenario 3: Ambition meets Reality",
    "S4": "Scenario 4: Autonomous Green Logistics",
}


@app.function(
    image=image,
    cpu=16.0,
    memory=32768,
    timeout=4 * 3600,  # 4 hours per scenario
    volumes={VOLUME_PATH: volume},
)
def train_scenario(scenario: str):
    """Train MaskablePPO for a single scenario inside a Modal container."""
    import os
    import gymnasium as gym
    import fleetreplacement_env  # registers FleetReplacement-v0
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.vec_env import sync_envs_normalization
    from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config

    os.chdir("/root")  # makes data/ relative paths in load_cost_config() work

    SCENARIO_NAME = SCENARIO_MAP[scenario]
    ENV_ID        = "FleetReplacement-v0"
    TOTAL_STEPS   = 5_000_000
    N_ENVS        = 16
    EVAL_FREQ     = 10_000
    LOG_DIR       = f"{VOLUME_PATH}/logs/scenarios/{scenario}/"
    SAVE_PATH     = f"{VOLUME_PATH}/models/scenarios/ppo_fleet_{scenario}"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/models/scenarios", exist_ok=True)

    def make_masked_env():
        cfg = FleetEnvConfig(mdp=MDPConfig(), cost=load_cost_config(scenario_name=SCENARIO_NAME))
        env = gym.make(ENV_ID, config=cfg)
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env

    vec_env = make_vec_env(make_masked_env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=1.0)

    _monitor_env = Monitor(make_masked_env(), filename=f"{LOG_DIR}/eval_monitor")
    eval_env = DummyVecEnv([lambda: _monitor_env])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False, gamma=1.0)
    sync_envs_normalization(vec_env, eval_env)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        render=False,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        tensorboard_log=LOG_DIR,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
    model = MaskablePPO.load(f"{SAVE_PATH}/best_model")
    model.save(f"{SAVE_PATH}_final")

    volume.commit()  # flush writes to the volume before container exits
    print(f"[{scenario}] Training complete. Model saved to {SAVE_PATH}_final")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    single_use_containers=True,
)
@modal.web_server(6006, startup_timeout=60)
def tensorboard():
    """
    Live TensorBoard server reading from the shared volume.
    Run in a separate terminal with: modal serve modal_train.py
    Modal will print a public URL you can open in your browser.
    """
    import subprocess
    subprocess.Popen([
        "tensorboard",
        "--logdir", f"{VOLUME_PATH}/logs",
        "--port", "6006",
        "--host", "0.0.0.0",
    ])


@app.local_entrypoint()
def main(scenario: str = "SQ"):
    """
    CLI entrypoint. Pass --scenario <key> or --scenario all.
    """
    if scenario == "all":
        # Spawn all 5 scenarios as separate containers in parallel
        for result in train_scenario.map(list(SCENARIO_MAP.keys())):
            pass  # .map() is lazy; iterating waits for all to finish
    elif scenario in SCENARIO_MAP:
        train_scenario.remote(scenario)
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIO_MAP.keys())} or 'all'")
