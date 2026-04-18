"""
Timeline heatmap: replacement decision probabilities over the planning horizon.

For a given scenario, runs N episodes with the trained MaskablePPO model,
records the action taken per vehicle per year, sorts vehicles by starting age
(youngest = row 0), and plots BET and DT replacement probability heatmaps.

Usage:
    python visualize_timeline.py SQ --v0
    python visualize_timeline.py S1 --v1
    python visualize_timeline.py S1 --v2 --episodes 100
    python visualize_timeline.py S1 --v2 --episodes 100 --seed 0
    python visualize_timeline.py S1 --v2_rt1 --episodes 100
    python visualize_timeline.py S1 --v2 --separate                 # generates separate heatmaps for DT and BET and stores them in final folder
"""

import argparse
import os
import re as _re
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

CUTOFF_YEAR = 2046   # visualize up to (exclusive) this year to mitigate EOH effects

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
parser.add_argument("--final", action="store_true",
                    help="Use the final model (ppo_fleet_SX_final.zip) instead of best_model")
parser.add_argument("--separate", action="store_true",
                    help="Save separate SVG/PNG files for BET and DT heatmaps")
args, _extra = parser.parse_known_args()

# Detect --vN flag dynamically (e.g. --v0, --v1, --v2, --v2_rt1, ...)
_version_flags = [a for a in _extra if _re.fullmatch(r"--v\d+\w*", a)]
if len(_version_flags) == 0:
    parser.error("A version flag is required (e.g. --v0, --v1, --v2, --v2_rt1, ...)")
if len(_version_flags) > 1:
    parser.error(f"Only one version flag allowed, got: {' '.join(_version_flags)}")
_version = _version_flags[0].lstrip("-")   # "v0", "v1", "v2_rt1", ...

SCENARIO_NAME = _SCENARIO_MAP[args.scenario]
N_EPISODES    = args.episodes
BASE_SEED     = args.seed
_scenario_tag = args.scenario
_model_suffix = "_final" if args.final else ""
if args.final:
    MODEL_PATH = f"./models/{_version}/ppo_fleet_{_scenario_tag}_final"
else:
    MODEL_PATH = f"./models/{_version}/ppo_fleet_{_scenario_tag}/best_model"

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

    # Truncate to CUTOFF_YEAR to mitigate end-of-horizon effects
    n_eval  = min(n_steps, CUTOFF_YEAR - start_year)
    tensor  = tensor[:, :, :n_eval]
    n_steps = n_eval

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
        even_year_ticks = [t for t in range(n_steps) if (start_year + t) % 2 == 0]
        ax.set_xticks(even_year_ticks)
        # ax.set_xticklabels([year_labels[t] for t in even_year_ticks], rotation=45, ha="right", fontsize=8)
        ax.set_xticklabels([year_labels[t] for t in even_year_ticks], rotation=0, ha="center", fontsize=8)     
        ax.set_yticks(range(n_vehicles))
        ax.set_yticklabels(vehicle_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Fraction of episodes")

    plt.tight_layout()
    os.makedirs(f"heatmaps/{_version}/SVG", exist_ok=True)
    os.makedirs(f"heatmaps/{_version}/PNG", exist_ok=True)
    svg_path = f"heatmaps/{_version}/SVG/timeline_heatmap_{_scenario_tag}{_model_suffix}.svg"
    png_path = f"heatmaps/{_version}/PNG/timeline_heatmap_{_scenario_tag}{_model_suffix}.png"
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    plt.show()

    if args.separate:
        _save_separate_heatmaps(bet_prob, dt_prob, year_labels, vehicle_labels,
                                n_steps, n_vehicles, start_year)


def _save_separate_heatmaps(bet_prob, dt_prob, year_labels, vehicle_labels,
                             n_steps, n_vehicles, start_year):
    os.makedirs("heatmaps/final/SVG", exist_ok=True)
    os.makedirs("heatmaps/final/PNG", exist_ok=True)

    panels = [
        (bet_prob, "BET", cmap_blue),
        (dt_prob,  "DT",  cmap_orange),
    ]
    for data, tag, cmap in panels:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap, origin="upper")
        ax.set_xlabel("Year", fontsize=14)
        ax.set_xticks(range(n_steps))
        x_labels = [year_labels[t] if t % 4 == 0 else "" for t in range(n_steps)]
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
        ax.set_yticks(range(n_vehicles))
        ax.set_yticklabels([""] * n_vehicles)
        ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
        ax.set_ylabel("")
        # "Age of vehicle" label + downward arrow to the left of the y-axis
        # (origin="upper" means row 0 = youngest at top, age increases downward)
        ax.text(-0.07, 0.5, "DT age at episode start", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, rotation=90)
        ax.annotate("", xy=(-0.04, 0.02), xytext=(-0.04, 0.98),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                    annotation_clip=False)
        plt.colorbar(im, ax=ax).set_label("Replacement rate", fontsize=14)
        plt.tight_layout()
        stem = f"timeline_heatmap_{_scenario_tag}{_model_suffix}_{tag}"
        svg_path = f"heatmaps/final/SVG/{stem}.svg"
        png_path = f"heatmaps/final/PNG/{stem}.png"
        plt.savefig(svg_path, bbox_inches="tight")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {svg_path}")
        print(f"Saved: {png_path}")
        plt.close(fig)

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
