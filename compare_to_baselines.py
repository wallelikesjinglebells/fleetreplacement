"""
Baseline evaluation with scenario-appropriate heuristics and RL comparison.

Runs the scenario-appropriate baseline set and the trained MaskablePPO model on the same environment so all results are comparable.

Usage:
    python compare_to_baselines.py SQ --v2
    python compare_to_baselines.py S1 --v3
    python compare_to_baselines.py S1 --v2 --no-plot
    python compare_to_baselines.py S1 --v2 --allbaselines
    python compare_to_baselines.py SQ --v2 --allbaselinesdifference --versions SQ=v2 S1=v3 S2=v2 S3=v1 S4=v3
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
from fleetreplacement_env.envs.costs import compute_step_cost

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CUTOFF_YEAR = 2046

_BAN_SCENARIOS    = {"SQ", "S3", "S4"}
_NO_BAN_SCENARIOS = {"S1", "S2"}

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
parser.add_argument("--final", action="store_true",
                    help="Use ppo_fleet_SX_final.zip instead of best_model")
parser.add_argument("--allbaselines", action="store_true",
                    help="Also produce a 2-box poster plot: pooled baselines vs RL")
parser.add_argument("--allbaselinesdifference", action="store_true",
                    help="Produce per-scenario cost-savings plot across all scenarios")
parser.add_argument("--no-plot", action="store_true",
                    help="Print terminal output only; skip all plot generation")
parser.add_argument("--versions", nargs="+", metavar="SCENARIO=version",
                    help="Per-scenario version overrides for --allbaselinesdifference")
args, _extra = parser.parse_known_args()

_version_flags = [a for a in _extra if _re.fullmatch(r"--v\d+\w*", a)]
if len(_version_flags) == 0:
    parser.error("A version flag is required (e.g. --v0, --v1, --v2, ...)")
if len(_version_flags) > 1:
    parser.error(f"Only one version flag allowed, got: {' '.join(_version_flags)}")
_version = _version_flags[0].lstrip("-")

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
    from sb3_contrib.common.wrappers import ActionMasker
    env = make_env(render_mode=render_mode)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def _masks(env) -> np.ndarray:
    """Returns action masks as (n_vehicles, 3) bool array."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    return env.unwrapped.action_masks().reshape(n, 3)


def _best_valid(mask_row: np.ndarray, preference_order=(2, 0, 1)) -> int:
    """Return first valid action in preference order."""
    for a in preference_order:
        if mask_row[a]:
            return a
    return int(np.argmax(mask_row))


# ---------------------------------------------------------------------------
# Baseline policies — no-ban scenarios (S1, S2)
# ---------------------------------------------------------------------------

def policy_eol_dt(env) -> np.ndarray:
    """Keep until forced to replace, then replace with DT."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not masks[i, 0]:  # keep is blocked → must replace
            actions[i] = _best_valid(masks[i], preference_order=(1, 2, 0))
    return actions


def policy_5yr_dt(env, threshold: int = 5) -> np.ndarray:
    """Replace any vehicle at or above `threshold` years old with DT."""
    fleet = env.unwrapped.fleet_state
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        age = fleet[i, 1]
        if not masks[i, 0]:  # forced replacement
            actions[i] = _best_valid(masks[i], preference_order=(1, 2, 0))
        elif age >= threshold and masks[i, 1]:  # voluntary, DT valid
            actions[i] = 1
    return actions


def policy_greedy_dt(env) -> np.ndarray:
    """Replace with DT whenever the mask allows it."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if masks[i, 1]:        # DT valid → replace
            actions[i] = 1
        elif not masks[i, 0]:  # must replace but DT blocked → BET fallback
            actions[i] = _best_valid(masks[i])
    return actions


# ---------------------------------------------------------------------------
# Baseline policies — ban scenarios (SQ, S3, S4)
# ---------------------------------------------------------------------------

def policy_eol_dtbet(env) -> np.ndarray:
    """Keep until forced: replace with DT pre-ban, BET post-ban."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    ban_step = env.unwrapped.dt_ban_step
    step = env.unwrapped.current_step
    for i in range(n):
        if not masks[i, 0]:  # must replace
            if step < ban_step and masks[i, 1]:  # pre-ban, DT available
                actions[i] = 1
            else:  # post-ban or DT unavailable → BET
                actions[i] = _best_valid(masks[i])
    return actions


def policy_5yr_dtbet(env, threshold: int = 5) -> np.ndarray:
    """5yr threshold: replace with DT pre-ban, BET post-ban."""
    fleet = env.unwrapped.fleet_state
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    ban_step = env.unwrapped.dt_ban_step
    step = env.unwrapped.current_step
    for i in range(n):
        age = fleet[i, 1]
        if not masks[i, 0]:  # forced replacement
            if step < ban_step and masks[i, 1]:
                actions[i] = 1
            else:
                actions[i] = _best_valid(masks[i])
        elif age >= threshold:  # voluntary replacement
            if step < ban_step and masks[i, 1]:
                actions[i] = 1
            elif masks[i, 2]:
                actions[i] = 2
    return actions


# ---------------------------------------------------------------------------
# Baseline policies — all scenarios
# ---------------------------------------------------------------------------

def policy_eol_bet(env) -> np.ndarray:
    """Keep until forced to replace, then replace with BET."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if not masks[i, 0]:  # must replace
            actions[i] = _best_valid(masks[i])
    return actions


def policy_5yr_bet(env, threshold: int = 5) -> np.ndarray:
    """Replace any vehicle at or above `threshold` years old with BET."""
    fleet = env.unwrapped.fleet_state
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        age = fleet[i, 1]
        if not masks[i, 0]:  # forced replacement
            actions[i] = _best_valid(masks[i])
        elif age >= threshold and masks[i, 2]:
            actions[i] = 2
    return actions


def policy_greedy_bet(env) -> np.ndarray:
    """Replace with BET whenever the mask allows it."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if masks[i, 2]:        # BET valid → replace
            actions[i] = 2
        elif not masks[i, 0]:  # must replace but BET blocked → DT
            actions[i] = _best_valid(masks[i])
    return actions


def policy_random(env) -> np.ndarray:
    """Sample a uniformly random valid action per vehicle (respects action masking)."""
    n = env.unwrapped.cfg.mdp.n_vehicles
    masks = _masks(env)
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        valid = np.where(masks[i])[0]
        actions[i] = int(env.unwrapped.np_random.choice(valid))
    return actions


def policy_cost_greedy(env) -> np.ndarray:
    """Myopic cost-greedy: pick the action minimising current-step cost per vehicle."""
    u = env.unwrapped
    n = u.cfg.mdp.n_vehicles
    masks = u.action_masks().reshape(n, 3)
    action = np.zeros(n, dtype=np.int32)
    n_charger_current = int(u.charger_slots.sum())  # frozen at step start, matching env.step()
    for i in range(n):
        tech, age, _ = u.fleet_state[i]
        best_action, best_cost = None, float("inf")
        for act in range(3):
            if not masks[i, act]:
                continue
            cost = compute_step_cost(
                tech=int(tech),
                age=age,
                action=act,
                annual_km=u.cfg.cost.akt_base,
                cfg=u.cfg.cost,
                current_year=u.cfg.mdp.start_year + u.current_step,
                has_charger=bool(u.charger_slots[i]),
                n_charger=n_charger_current,
            ).total
            if cost < best_cost:
                best_cost = cost
                best_action = act
        action[i] = best_action
    return action


# ---------------------------------------------------------------------------
# Scenario-appropriate baseline list
# ---------------------------------------------------------------------------
def get_baselines(scenario_tag: str) -> list:
    """Return ordered (name, fn, kwargs) list for the given scenario tag."""
    common = [
        ("EOL -> BET",    policy_eol_bet,     {}),
        ("5yr -> BET",    policy_5yr_bet,     {}),
        ("Greedy BET",    policy_greedy_bet,  {}),
        ("Random",        policy_random,      {}),
        ("Cost-Greedy",   policy_cost_greedy, {}),
    ]
    if scenario_tag in _NO_BAN_SCENARIOS:
        return [
            ("EOL -> DT",   policy_eol_dt,    {}),
            ("5yr -> DT",   policy_5yr_dt,    {}),
            ("Greedy DT",   policy_greedy_dt, {}),
        ] + common
    else:  # SQ, S3, S4 — ban scenarios
        return [
            ("EOL -> DT -> BET", policy_eol_dtbet, {}),
            ("5yr -> DT -> BET", policy_5yr_dtbet, {}),
        ] + common


# ---------------------------------------------------------------------------
# Episode runner (baselines)
# ---------------------------------------------------------------------------
def run_episode(policy_fn, env, seed=None, policy_kwargs=None, eval_steps=None) -> float:
    obs, _ = env.reset(seed=seed)
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
# Plots — same style as compare_results.py, output to baselinecomparison/
# ---------------------------------------------------------------------------
def plot_comparison(results: dict[str, np.ndarray]):
    os.makedirs(f"baselinecomparison/{_version}/PNG", exist_ok=True)
    os.makedirs(f"baselinecomparison/{_version}/SVG", exist_ok=True)
    stem    = f"baselinecomparison_{_scenario_tag}{_model_suffix}"
    png_dir = f"baselinecomparison/{_version}/PNG"
    svg_dir = f"baselinecomparison/{_version}/SVG"

    names  = list(results.keys())
    colors = [TUM_BLUE] * len(names)
    if "RL (PPO)" in names:
        colors[names.index("RL (PPO)")] = TUM_ORANGE

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
    ax_top.set_title(f"Baseline comparison — {SCENARIO_NAME}")
    ax_bot.tick_params(axis="x", rotation=30)

    png_path = f"{png_dir}/{stem}_box.png"
    svg_path = f"{svg_dir}/{stem}_box.svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.show()


def plot_allbaselines_comparison(results: dict[str, np.ndarray]):
    """2-box poster plot: all baselines pooled vs RL."""
    if "RL (PPO)" not in results:
        print("[--allbaselines] No RL results available — skipping poster plot.")
        return

    os.makedirs("baselinecomparison/final/PNG", exist_ok=True)
    os.makedirs("baselinecomparison/final/SVG", exist_ok=True)
    png_dir = "baselinecomparison/final/PNG"
    svg_dir = "baselinecomparison/final/SVG"
    stem = f"baselinecomparison_{_scenario_tag}{_model_suffix}"

    baseline_pool = np.concatenate([-results[n] for n in results if n != "RL (PPO)"])
    rl_vals = -results["RL (PPO)"]

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
# Multi-scenario helpers (--allbaselinesdifference)
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


def compute_difference_data() -> dict[str, np.ndarray]:
    """For each scenario, compute per-episode cost savings: mean_baseline_cost − RL_cost."""
    diff_data = {}
    scenarios = {t: _SCENARIO_MAP[t] for t in _scenario_versions} if _scenario_versions else _SCENARIO_MAP
    for scenario_tag, scenario_name in scenarios.items():
        ver = _scenario_versions.get(scenario_tag, _version)
        print(f"\n  Scenario: {scenario_name}  (version: {ver})")
        baselines = get_baselines(scenario_tag)
        baseline_rewards = []
        for name, fn, kwargs in baselines:
            r = _evaluate_policy_for(fn, scenario_name, policy_kwargs=kwargs)
            baseline_rewards.append(r)
            print(f"    {name:<22}  mean EUR {np.mean(r):>14,.0f}")
        baseline_rewards = np.array(baseline_rewards)
        mean_baseline_cost = -np.mean(baseline_rewards, axis=0)

        rl_rewards = _evaluate_rl_for(scenario_tag)
        if rl_rewards is None:
            continue
        rl_cost = -rl_rewards
        print(f"    {'RL':<22}  mean EUR {np.mean(rl_rewards):>14,.0f}")

        diff_data[scenario_tag] = mean_baseline_cost - rl_cost  # positive = RL cheaper
    return diff_data


def plot_allbaselines_difference(diff_data: dict[str, np.ndarray]):
    """Box plot: per-episode cost savings (baseline mean − RL) per scenario."""
    out_png = "baselinecomparison/final/allbaselinesdifference/PNG"
    out_svg = "baselinecomparison/final/allbaselinesdifference/SVG"
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_svg, exist_ok=True)

    scenario_tags = list(diff_data.keys())
    labels = [_SHORT_LABELS[t] for t in scenario_tags]
    data   = [diff_data[t] / 1e6 for t in scenario_tags]

    suffix   = "_final" if args.final else ""
    stem     = f"baselinecomparison_allscenarios{suffix}_allbaselinesdifference"
    png_path = f"{out_png}/{stem}.png"
    svg_path = f"{out_svg}/{stem}.svg"

    with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9}):
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
# Main
# ---------------------------------------------------------------------------
BASELINES = get_baselines(_scenario_tag)

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
best_mean = np.mean(results[best_name])
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

if not args.no_plot:
    plot_comparison(results)
if args.allbaselines and not args.no_plot:
    plot_allbaselines_comparison(results)
if args.allbaselinesdifference:
    print("\n--- Computing difference data across all scenarios ---")
    diff_data = compute_difference_data()
    if not args.no_plot:
        plot_allbaselines_difference(diff_data)
