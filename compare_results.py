"""
Baseline + RL evaluation (direct comparison)

Runs several "manual" policies and the trained MaskablePPO model on the same
environment so that all results are directly comparable.

Usage:
    python compare_results.py SQ --v0
    python compare_results.py S1 --v1
    python compare_results.py SQ --v2 --trace
    python compare_results.py S1 --v2 --final
    python compare_results.py S1 --v2_rt1
    python compare_results.py S1 --v1 --allbaselines    # generates plot with all baselines pooled
    python compare_results.py SQ --v2 --allbaselinesdifference --versions SQ=v2 S1=v3 S2=v2 S3=v1 S4=v3
                                                         # per-scenario difference plot across all scenarios
                                                         # version pairs are optional; omitted scenarios fall back to global --vN
"""

import argparse
import os
import re as _re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import scienceplots
plt.style.use(["science", "nature", "grid"])
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "Arial"
from tum_colors import TUM_BLUE, TUM_ORANGE
import gymnasium as gym
import fleetreplacement_env
from fleetreplacement_env.envs.config import FleetEnvConfig, SDPConfig, load_cost_config

# Suppress TensorFlow oneDNN notifications before sb3_contrib imports TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CUTOFF_YEAR = 2046   # evaluate costs only up to (exclusive) this year to mitigate EOH effects

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
                    help="Number of episodes per policy (default: 50)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed (default: 42)")
parser.add_argument("--trace", action="store_true",
                    help="Print step-by-step output for the best baseline and the RL model")
parser.add_argument("--final", action="store_true",
                    help="Use the final model (ppo_fleet_SX_final.zip) instead of best_model")
parser.add_argument("--allbaselines", action="store_true",
                    help="Also produce a 2-box poster plot: pooled heuristic baselines vs RL")
parser.add_argument("--allbaselinesdifference", action="store_true",
                    help="Produce a per-scenario difference plot (baseline mean cost − RL cost) across all scenarios")
parser.add_argument("--versions", nargs="+", metavar="SCENARIO=version",
                    help="Per-scenario version overrides for --allbaselinesdifference, e.g. S1=v1 S2=v2_rt1 (falls back to global --vN for unspecified scenarios)")
args, _extra = parser.parse_known_args()

# Detect --vN flag dynamically (e.g. --v0, --v1, --v2, --v2_rt1, ...)
_version_flags = [a for a in _extra if _re.fullmatch(r"--v\d+\w*", a)]
if len(_version_flags) == 0:
    parser.error("A version flag is required (e.g. --v0, --v1, --v2, --v2_rt1, ...)")
if len(_version_flags) > 1:
    parser.error(f"Only one version flag allowed, got: {' '.join(_version_flags)}")
_version = _version_flags[0].lstrip("-")   # "v0", "v1", "v2_rt1", ...

# Per-scenario version overrides (used by --allbaselinesdifference)
_scenario_versions: dict[str, str] = {}
if args.versions:
    for entry in args.versions:
        if "=" not in entry:
            parser.error(f"--versions entries must be SCENARIO=version, got: {entry!r}")
        tag, ver = entry.split("=", 1)
        if tag not in _SCENARIO_MAP:
            parser.error(f"Unknown scenario in --versions: {tag!r}")
        _scenario_versions[tag] = ver

SCENARIO_NAME  = _SCENARIO_MAP[args.scenario]
N_EPISODES     = args.episodes
BASE_SEED      = args.seed
_scenario_tag  = args.scenario
_model_suffix  = "_final" if args.final else ""
if args.final:
    MODEL_PATH = f"./models/{_version}/ppo_fleet_{_scenario_tag}_final"
else:
    MODEL_PATH = f"./models/{_version}/ppo_fleet_{_scenario_tag}/best_model"

# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------
def make_env(render_mode=None):
    cfg = FleetEnvConfig(mdp=SDPConfig(), cost=load_cost_config(scenario_name=SCENARIO_NAME))
    return gym.make("FleetReplacement-v0", config=cfg, render_mode=render_mode)


def make_env_masked(render_mode=None):
    """Wrapped environment for the RL model (ActionMasker required by MaskablePPO)."""
    from sb3_contrib.common.wrappers import ActionMasker
    env = make_env(render_mode=render_mode)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def _masks(env) -> np.ndarray:
    """Returns masks as (n_vehicles, 3) bool array."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    return env.unwrapped.action_masks().reshape(n, 3)


def _best_valid(mask_row: np.ndarray, preference_order=(2, 0, 1)) -> int:
    """Return first valid action in preference order."""
    for a in preference_order:
        if mask_row[a]:
            return a
    return int(np.argmax(mask_row))  # safety fallback


# ---------------------------------------------------------------------------
# Baseline policies
# Each function takes an env, reads fleet state, and returns an action array.
# ---------------------------------------------------------------------------

def policy_eol_bet(env) -> np.ndarray:
    """
    End-of-life only: keep every vehicle until forced to replace, then choose BET.
    """
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not masks[i, 0]:                        # keep is blocked → must replace
            actions[i] = _best_valid(masks[i])     # prefer BET
        # else: keep (action=0, already set)
    return actions


def policy_age_threshold_bet(env, threshold: int = 5) -> np.ndarray:
    """
    Replace any vehicle at or above `threshold` years old with BET, keep younger vehicles unless forced to replace.
    """
    fleet = env.unwrapped.fleet_state              # (n, 3): tech, age, mileage
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        age = fleet[i, 1]
        if not masks[i, 0]:                        # forced replacement
            actions[i] = _best_valid(masks[i])
        elif age >= threshold and masks[i, 2]:     # above threshold and BET is valid
            actions[i] = 2
        # else: keep
    return actions


def policy_oldest_first_bet(env, k: int = 1) -> np.ndarray:
    """
    Each step, replace the k oldest vehicles with BET (staggered replacement).
    Also handles forced replacements.
    """
    fleet = env.unwrapped.fleet_state
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)

    # Pass 1: forced replacements
    for i in range(n):
        if not masks[i, 0]:
            actions[i] = _best_valid(masks[i])

    # Pass 2: voluntarily replace k oldest among replaceable vehicles
    ages = fleet[:, 1]
    candidates = [
        i for i in range(n)
        if actions[i] == 0          # not already forced
        and masks[i, 0]             # keep is valid (age > 1)
        and masks[i, 2]             # BET replacement is valid
    ]
    candidates_sorted = sorted(candidates, key=lambda i: -ages[i])  # oldest first
    for i in candidates_sorted[:k]:
        actions[i] = 2

    return actions


def policy_fixed_cycle_bet(env, cycle: int = 5) -> np.ndarray:
    """
    Replace a vehicle on a fixed cycle: vehicle i is replaced when current_step % cycle == i % cycle.
    Spreads replacements evenly across time (round-robin by vehicle index).
    """
    step = env.unwrapped.current_step
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not masks[i, 0]:                            # forced
            actions[i] = _best_valid(masks[i])
        elif (step % cycle) == (i % cycle) and masks[i, 2] and masks[i, 0]:
            actions[i] = 2                             # scheduled replacement
        # else: keep
    return actions


def policy_staggered_schedule_bet(env) -> np.ndarray:
    """
    Staggered schedule: pre-assigns each vehicle a target replacement step at episode start based on its current age, spreading all replacements evenly across the remaining planning horizon.

    Logic:
      - Sort vehicles by age (youngest first → replaced latest)
      - Divide the horizon into n_vehicles equally-spaced slots
      - Assign slot i to the i-th youngest vehicle
      - Replace a vehicle when current_step reaches its assigned slot

    Unlike fixed_cycle (which repeats forever on a fixed cadence), this schedule is computed once from the actual starting fleet state and aims for a single smooth wave of replacements across the horizon.
    Forced replacements are respected.
    """
    env_unwrapped = env.unwrapped
    fleet  = env_unwrapped.fleet_state
    n      = env_unwrapped.cfg.mdp.n_vehicles
    h      = env_unwrapped.cfg.mdp.planning_horizon
    step   = env_unwrapped.current_step
    masks  = _masks(env)
    actions = np.zeros(n, dtype=np.int32)

    # Build or retrieve the schedule (stored on the env so it persists per episode)
    if not hasattr(env_unwrapped, "_stagger_schedule") or env_unwrapped._stagger_schedule is None:
        ages = fleet[:, 1]
        # Sort vehicle indices youngest-first; youngest gets the latest slot
        order = np.argsort(ages)          # ascending age → latest replacement slot
        schedule = np.empty(n, dtype=int)
        for rank, vehicle_idx in enumerate(order):
            schedule[vehicle_idx] = round(rank * h / n)
        env_unwrapped._stagger_schedule = schedule

    schedule = env_unwrapped._stagger_schedule

    for i in range(n):
        if not masks[i, 0]:                                  # forced replacement
            actions[i] = _best_valid(masks[i])
        elif step == schedule[i] and masks[i, 2]:            # scheduled slot
            actions[i] = 2
        # else: keep

    return actions


def policy_greedy_bet(env) -> np.ndarray:
    """
    Replace with BET whenever the mask allows it.
    """
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if masks[i, 2]:            # BET replacement valid → do it
            actions[i] = 2
        elif not masks[i, 0]:      # must replace but BET blocked → DT
            actions[i] = _best_valid(masks[i])
        # else: keep
    return actions


# ---------------------------------------------------------------------------
# Episode runner (baselines)
# ---------------------------------------------------------------------------
def run_episode(policy_fn, env, seed=None, policy_kwargs=None, eval_steps=None) -> float:
    """Run one episode and return total (undiscounted by RL, discounted internally) reward.

    If eval_steps is set, only accumulate reward for the first eval_steps steps
    (the episode still runs to completion so policy decisions are unaffected).
    """
    obs, _ = env.reset(seed=seed)
    env.unwrapped._stagger_schedule = None   # clear staggered schedule for new episode
    done = False
    total_reward = 0.0
    step = 0
    kwargs = policy_kwargs or {}
    while not done:
        action = policy_fn(env, **kwargs)
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        if eval_steps is None or step <= eval_steps:
            total_reward += reward
        done = terminated or truncated
    return total_reward


def evaluate_policy(policy_fn, policy_kwargs=None, n_episodes=N_EPISODES) -> np.ndarray:
    env = make_env()
    cfg = env.unwrapped.cfg.mdp
    eval_steps = min(cfg.planning_horizon, CUTOFF_YEAR - cfg.start_year)
    rewards = np.array([
        run_episode(policy_fn, env, seed=BASE_SEED + ep, policy_kwargs=policy_kwargs,
                    eval_steps=eval_steps)
        for ep in range(n_episodes)
    ])
    env.close()
    return rewards


# ---------------------------------------------------------------------------
# RL evaluation
# ---------------------------------------------------------------------------
def evaluate_rl(n_episodes=N_EPISODES):
    """Load the best saved MaskablePPO model and evaluate it.

    The full 30-year trained policy is executed, but only the first
    eval_steps years of costs are accumulated (to mitigate EOH effects).
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(MODEL_PATH)
    env = make_env_masked()
    cfg = env.unwrapped.cfg.mdp
    eval_steps = min(cfg.planning_horizon, CUTOFF_YEAR - cfg.start_year)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=BASE_SEED + ep)
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if step <= eval_steps:
                total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return np.array(rewards), model


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_comparison(results: dict[str, np.ndarray]):
    os.makedirs(f"comparison_figures/{_version}/PNG", exist_ok=True)
    os.makedirs(f"comparison_figures/{_version}/SVG", exist_ok=True)
    stem    = f"comparison_{_scenario_tag}{_model_suffix}"
    png_dir = f"comparison_figures/{_version}/PNG"
    svg_dir = f"comparison_figures/{_version}/SVG"

    names  = list(results.keys())
    colors = [TUM_BLUE] * len(names)
    if "RL (PPO)" in names:
        colors[names.index("RL (PPO)")] = TUM_ORANGE

    # --- Box plot (broken y-axis: data panel + zero anchor) ---
    all_vals = np.concatenate([-results[n] for n in names])
    data_min, data_max = all_vals.min(), all_vals.max()
    span = data_max - data_min

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.18, hspace=0.04)

    def _draw_bp(ax):
        bp = ax.boxplot(
            [-results[n] for n in names],
            tick_labels=names,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")

    _draw_bp(ax_top)
    _draw_bp(ax_bot)

    # Shared tick interval (same granularity above and below the break)
    raw_step = span / 5
    magnitude = 10 ** np.floor(np.log10(raw_step))
    tick_step = round(raw_step / magnitude) * magnitude

    ax_top.set_ylim(data_min - span * 0.05, data_max + span * 0.1)
    ax_bot.set_ylim(0, data_min * 0.12)

    millions = FuncFormatter(lambda x, _: f"{x / 1e6:.1f}")
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_formatter(millions)
        ax.yaxis.get_offset_text().set_visible(False)

    # Otional: remove vertical grid lines
    # ax_top.xaxis.grid(False)
    # ax_bot.xaxis.grid(False)

    # Hide the touching spines to create the visual break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    # Slanted break markers (marker-based, style-independent)
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

    ax_top.set_ylabel("Costs (EUR millions)")
    ax_top.set_title(f"Policy comparison — {SCENARIO_NAME}")
    ax_bot.tick_params(axis="x", rotation=30)
    png_path = f"{png_dir}/{stem}_box.png"
    svg_path = f"{svg_dir}/{stem}_box.svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.show()


def plot_allbaselines_comparison(results: dict[str, np.ndarray]):
    """2-box poster plot: all baseline rewards pooled into one box vs RL (PPO)."""
    if "RL (PPO)" not in results:
        print("[--allbaselines] No RL results available — skipping poster plot.")
        return

    os.makedirs(f"comparison_figures/final/PNG", exist_ok=True)
    os.makedirs(f"comparison_figures/final/SVG", exist_ok=True)
    png_dir = f"comparison_figures/final/PNG"
    svg_dir = f"comparison_figures/final/SVG"
    stem    = f"comparison_{_scenario_tag}{_model_suffix}"

    baseline_pool = np.concatenate([-results[n] for n in results if n != "RL (PPO)"])
    rl_vals       = -results["RL (PPO)"]

    names  = ["Heuristic baselines", "RL"]
    data   = [baseline_pool, rl_vals]
    colors = [TUM_BLUE, TUM_ORANGE]

    all_vals = np.concatenate([baseline_pool, rl_vals])
    data_min, data_max = all_vals.min(), all_vals.max()
    span = data_max - data_min

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(5, 6),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.subplots_adjust(left=0.15, right=0.54, top=0.92, bottom=0.34, hspace=0.04, wspace=0.2)

    def _draw_bp(ax):
        bp = ax.boxplot(
            data,
            positions=[1, 1.15],
            widths=0.12,
            tick_labels=names,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")

    _draw_bp(ax_top)
    _draw_bp(ax_bot)

    raw_step = span / 5
    magnitude = 10 ** np.floor(np.log10(raw_step))
    tick_step = round(raw_step / magnitude) * magnitude

    for ax in (ax_top, ax_bot):
        ax.set_xlim(0.82, 1.33)
    ax_top.set_ylim(data_min - span * 0.05, data_max + span * 0.1)
    ax_bot.set_ylim(0, data_min * 0.12)

    millions = FuncFormatter(lambda x, _: f"{x / 1e6:.1f}")
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_formatter(millions)
        ax.yaxis.get_offset_text().set_visible(False)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.5
    bk = dict(marker=[(-1, -d), (1, d)], markersize=12,
               linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **bk)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **bk)

    ax_top.set_ylabel("Costs (EUR millions)")

    png_path = f"{png_dir}/{stem}_allbaselines_box.png"
    svg_path = f"{svg_dir}/{stem}_allbaselines_box.svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Helpers for multi-scenario evaluation (used by --allbaselinesdifference)
# ---------------------------------------------------------------------------
_SHORT_LABELS = {
    "SQ": "Status Quo",
    "S1": "Scenario 1",
    "S2": "Scenario 2",
    "S3": "Scenario 3",
    "S4": "Scenario 4",
}


def _make_env_for(scenario_name, masked=False):
    cfg = FleetEnvConfig(mdp=SDPConfig(), cost=load_cost_config(scenario_name=scenario_name))
    env = gym.make("FleetReplacement-v0", config=cfg)
    if masked:
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env


def _evaluate_policy_for(policy_fn, scenario_name, policy_kwargs=None) -> np.ndarray:
    env = _make_env_for(scenario_name)
    cfg = env.unwrapped.cfg.mdp
    eval_steps = min(cfg.planning_horizon, CUTOFF_YEAR - cfg.start_year)
    rewards = np.array([
        run_episode(policy_fn, env, seed=BASE_SEED + ep, policy_kwargs=policy_kwargs,
                    eval_steps=eval_steps)
        for ep in range(N_EPISODES)
    ])
    env.close()
    return rewards


def _evaluate_rl_for(scenario_tag) -> np.ndarray | None:
    from sb3_contrib import MaskablePPO
    ver = _scenario_versions.get(scenario_tag, _version)
    if args.final:
        path = f"./models/{ver}/ppo_fleet_{scenario_tag}_final"
    else:
        path = f"./models/{ver}/ppo_fleet_{scenario_tag}/best_model"
    if not os.path.exists(path + ".zip"):
        print(f"  [RL] No model found for {scenario_tag} at {path}.zip — skipping.")
        return None
    model = MaskablePPO.load(path)
    scenario_name = _SCENARIO_MAP[scenario_tag]
    env = _make_env_for(scenario_name, masked=True)
    cfg = env.unwrapped.cfg.mdp
    eval_steps = min(cfg.planning_horizon, CUTOFF_YEAR - cfg.start_year)
    rewards = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=BASE_SEED + ep)
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if step <= eval_steps:
                total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return np.array(rewards)


def compute_difference_data(baselines_list) -> dict[str, np.ndarray]:
    """For each scenario, compute per-episode cost savings: mean_baseline_cost − RL_cost.
    Positive = RL is cheaper."""
    diff_data = {}
    scenarios = {t: _SCENARIO_MAP[t] for t in _scenario_versions} if _scenario_versions else _SCENARIO_MAP
    for scenario_tag, scenario_name in scenarios.items():
        ver = _scenario_versions.get(scenario_tag, _version)
        print(f"\n  Scenario: {scenario_name}  (version: {ver})")
        baseline_rewards = []
        for name, fn, kwargs in baselines_list:
            r = _evaluate_policy_for(fn, scenario_name, policy_kwargs=kwargs)
            baseline_rewards.append(r)
            print(f"    {name:<22}  mean EUR {np.mean(r):>14,.0f}")
        baseline_rewards = np.array(baseline_rewards)   # (n_baselines, n_episodes)
        mean_baseline_cost = -np.mean(baseline_rewards, axis=0)  # (n_episodes,) in EUR

        rl_rewards = _evaluate_rl_for(scenario_tag)
        if rl_rewards is None:
            continue
        rl_cost = -rl_rewards  # (n_episodes,) in EUR
        print(f"    {'RL':<22}  mean EUR {np.mean(rl_rewards):>14,.0f}")

        diff_data[scenario_tag] = mean_baseline_cost - rl_cost  # positive = RL cheaper
    return diff_data


def plot_allbaselines_difference(diff_data: dict[str, np.ndarray]):
    """Box plot: per-episode cost savings (baseline mean − RL) per scenario."""
    out_png = "comparison_figures/final/allbaselinesdifference/PNG"
    out_svg = "comparison_figures/final/allbaselinesdifference/SVG"
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_svg, exist_ok=True)

    scenario_tags  = list(diff_data.keys())
    labels = [_SHORT_LABELS[t] for t in scenario_tags]
    data   = [diff_data[t] / 1e6 for t in scenario_tags]  # EUR → EUR millions

    suffix = "_final" if args.final else ""
    stem = f"comparison_allscenarios{suffix}_allbaselinesdifference"
    png_path = f"{out_png}/{stem}.png"
    svg_path = f"{out_svg}/{stem}.svg"

    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.subplots_adjust(left=0.12, right=0.73, top=0.95, bottom=0.22, hspace=0.2, wspace=0.2)

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(TUM_BLUE)
            patch.set_edgecolor("black")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("RL cost savings vs. baselines\n(EUR M, 20-year horizon)")
        ax.tick_params(axis="x", rotation=0)

        fig.savefig(png_path, dpi=150)
        fig.savefig(svg_path)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.show()


# ---------------------------------------------------------------------------
# All baselines to benchmark
# ---------------------------------------------------------------------------
BASELINES = [
    ("EOL to BET",          policy_eol_bet,               {}),
    ("Age>=4 to BET",       policy_age_threshold_bet,     {"threshold": 4}),
    ("Age>=5 to BET",       policy_age_threshold_bet,     {"threshold": 5}),
    ("Age>=6 to BET",       policy_age_threshold_bet,     {"threshold": 6}),
    ("Age>=7 to BET",       policy_age_threshold_bet,     {"threshold": 7}),
    ("Oldest-first k=1",    policy_oldest_first_bet,      {"k": 1}),
    ("Oldest-first k=2",    policy_oldest_first_bet,      {"k": 2}),
    ("Fixed cycle 5yr",     policy_fixed_cycle_bet,       {"cycle": 5}),
    ("Fixed cycle 7yr",     policy_fixed_cycle_bet,       {"cycle": 7}),
    ("Staggered schedule",  policy_staggered_schedule_bet, {}),
    ("Greedy BET",          policy_greedy_bet,            {}),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\nScenario : {SCENARIO_NAME}")
print(f"Episodes : {N_EPISODES}  |  Base seed : {BASE_SEED}\n")
print(f"{'Policy':<22} {'Mean (EUR)':>15} {'Std':>12} {'Best':>15} {'Worst':>15}")
print("-" * 82)

results: dict[str, np.ndarray] = {}
for name, fn, kwargs in BASELINES:
    rewards = evaluate_policy(fn, policy_kwargs=kwargs)
    results[name] = rewards
    print(
        f"{name:<22}"
        f"  {np.mean(rewards):>14,.0f}"
        f"  {np.std(rewards):>11,.0f}"
        f"  {np.max(rewards):>14,.0f}"
        f"  {np.min(rewards):>14,.0f}"
    )

# ---------------------------------------------------------------------------
# RL model row
# ---------------------------------------------------------------------------
rl_model = None
rl_rewards = None

if os.path.exists(MODEL_PATH + ".zip"):
    print("-" * 82)
    rl_rewards, rl_model = evaluate_rl()
    results["RL (PPO)"] = rl_rewards
    print(
        f"{'RL (PPO)':<22}"
        f"  {np.mean(rl_rewards):>14,.0f}"
        f"  {np.std(rl_rewards):>11,.0f}"
        f"  {np.max(rl_rewards):>14,.0f}"
        f"  {np.min(rl_rewards):>14,.0f}"
    )
else:
    print(f"\n[RL] No model found at {MODEL_PATH}.zip — skipping RL evaluation.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: np.mean(results[k]))
best_mean  = np.mean(results[best_name])
best_baseline_name = max(
    (k for k in results if k != "RL (PPO)"),
    key=lambda k: np.mean(results[k])
)
best_baseline_mean = np.mean(results[best_baseline_name])

print("-" * 82)
print(f"\nBest overall : {best_name}  (mean EUR {best_mean:,.0f})")
print(f"Best baseline: {best_baseline_name}  (mean EUR {best_baseline_mean:,.0f})")
if rl_rewards is not None:
    rl_mean = np.mean(rl_rewards)
    delta = rl_mean - best_baseline_mean
    sign  = "+" if delta >= 0 else ""
    print(f"RL vs best baseline: {sign}{delta:,.0f} EUR  "
          f"({'above' if delta >= 0 else 'below'} baseline)")
print()

plot_comparison(results)
if args.allbaselines:
    plot_allbaselines_comparison(results)
if args.allbaselinesdifference:
    print("\n--- Computing difference data across all scenarios ---")
    diff_data = compute_difference_data(BASELINES)
    plot_allbaselines_difference(diff_data)

# ---------------------------------------------------------------------------
# Optional: step-by-step trace for the best baseline and the RL model
# ---------------------------------------------------------------------------
if args.trace:
    fn_map = {name: (fn, kw) for name, fn, kw in BASELINES}
    best_fn, best_kw = fn_map[best_baseline_name]

    print(f"\n{'='*60}")
    print(f"Step-by-step trace (baseline): {best_baseline_name}  (seed={BASE_SEED})")
    print(f"{'='*60}")
    env = make_env(render_mode="human")
    total_reward = run_episode(best_fn, env, seed=BASE_SEED, policy_kwargs=best_kw)
    print(f"\nEpisode total reward: EUR {total_reward:,.0f}")
    env.close()

    if rl_model is not None:
        print(f"\n{'='*60}")
        print(f"Step-by-step trace (RL): PPO best_model  (seed={BASE_SEED})")
        print(f"{'='*60}")
        env = make_env_masked(render_mode="human")
        obs, _ = env.reset(seed=BASE_SEED)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"\nEpisode total reward: EUR {total_reward:,.0f}")
        env.close()
