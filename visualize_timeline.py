"""
Timeline heatmap: replacement decision probabilities over the planning horizon.

For a given scenario, runs N episodes with the trained MaskablePPO model,
records the action taken per vehicle per year, sorts vehicles by starting age
(youngest = row 0), and plots BET and DT replacement probability heatmaps.

Usage:
    python visualize_timeline.py              # Status Quo (default)
    python visualize_timeline.py SQ
    python visualize_timeline.py S1
    python visualize_timeline.py S1 --episodes 100
    python visualize_timeline.py S1 --episodes 100 --seed 0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "nature", "grid"])
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "Arial"
from tum_colors import cmap_blue, cmap_orange
import gymnasium as gym
import fleetreplacement_env
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
                    help="Number of episodes to collect (default: 50)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed (default: 42)")
args = parser.parse_args()

SCENARIO_NAME = _SCENARIO_MAP[args.scenario]
N_EPISODES    = args.episodes
BASE_SEED     = args.seed
MODEL_PATH    = f"./models/scenarios/ppo_fleet_{args.scenario}/best_model"

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env_masked():
    from sb3_contrib.common.wrappers import ActionMasker
    cfg = FleetEnvConfig(mdp=MDPConfig(), cost=load_cost_config(scenario_name=SCENARIO_NAME))
    env = gym.make("FleetReplacement-v0", config=cfg)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_action_tensor() -> tuple[np.ndarray, int]:
    """
    Run N_EPISODES with the RL model.

    Returns
    -------
    tensor : np.ndarray, shape (n_episodes, n_vehicles, n_steps)
        Action taken per (episode, vehicle-rank, year).
        Rows are sorted by ascending starting age (rank 0 = youngest at t=0).
    start_year : int
        Calendar year of step 0, for axis labels.
    """
    from sb3_contrib import MaskablePPO

    model      = MaskablePPO.load(MODEL_PATH)
    env        = make_env_masked()
    n_vehicles = env.unwrapped.cfg.mdp.n_vehicles
    n_steps    = env.unwrapped.cfg.mdp.planning_horizon
    start_year = env.unwrapped.cfg.mdp.start_year

    tensor = np.empty((N_EPISODES, n_vehicles, n_steps), dtype=np.int32)

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=BASE_SEED + ep)

        # Age-rank: sort vehicles youngest → oldest at episode start (once per episode)
        starting_ages = env.unwrapped.fleet_state[:, 1].copy()
        age_rank = np.argsort(starting_ages)   # age_rank[0] = index of youngest vehicle

        actions_ep = np.empty((n_vehicles, n_steps), dtype=np.int32)
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, terminated, truncated, _ = env.step(action)
            actions_ep[:, step] = action       # action shape: (n_vehicles,)
            step += 1
            done = terminated or truncated

        # Reorder rows so row 0 = youngest vehicle at episode start
        tensor[ep] = actions_ep[age_rank, :]

    env.close()
    return tensor, start_year

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_heatmaps(tensor: np.ndarray, start_year: int):
    n_vehicles = tensor.shape[1]
    n_steps    = tensor.shape[2]

    bet_prob = np.mean(tensor == 2, axis=0)   # (n_vehicles, n_steps)
    dt_prob  = np.mean(tensor == 1, axis=0)

    year_labels    = [str(start_year + t) for t in range(n_steps)]
    vehicle_labels = [f"Rank {i} (youngest)" if i == 0
                      else f"Rank {i} (oldest)" if i == n_vehicles - 1
                      else f"Rank {i}"
                      for i in range(n_vehicles)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Replacement probability — {SCENARIO_NAME}\n"
        f"(RL policy, {N_EPISODES} episodes, rows sorted by starting age)",
        fontsize=12,
    )

    panels = [
        (axes[0], bet_prob, "BET replacement probability", cmap_blue),
        (axes[1], dt_prob,  "DT replacement probability",  cmap_orange),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Vehicle rank at episode start")
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels(year_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_vehicles))
        ax.set_yticklabels(vehicle_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Fraction of episodes")

    plt.tight_layout()
    os.makedirs("heatmaps/SVGs", exist_ok=True)
    os.makedirs("heatmaps/PNGs", exist_ok=True)
    svg_path = f"heatmaps/SVGs/timeline_heatmap_{args.scenario}.svg"
    png_path = f"heatmaps/PNGs/timeline_heatmap_{args.scenario}.png"
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH + ".zip"):
    raise FileNotFoundError(f"No model found at {MODEL_PATH}.zip")

print(f"\nScenario : {SCENARIO_NAME}")
print(f"Episodes : {N_EPISODES}  |  Base seed : {BASE_SEED}")
print(f"Model    : {MODEL_PATH}\n")

tensor, start_year = collect_action_tensor()
plot_heatmaps(tensor, start_year)
