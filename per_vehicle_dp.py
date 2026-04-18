"""
Per-vehicle backward induction DP solver for the fleet replacement MDP.

Decomposes the N-vehicle joint problem into N independent single-vehicle DPs,
with a two-pass charger-coupling strategy to handle the one-time infrastructure
construction cost for the first BET purchase in the fleet.

Charger coupling
----------------
The first vehicle in the fleet to buy a BET pays construction_cost_contrib +
charger_price.  Every subsequent BET slot pays charger_price only.  Once a
vehicle slot has a charger it keeps it, so re-buying BET at the same slot is
free of infra cost.

Two-pass strategy
-----------------
1. Solve dp_infra: fleet charger infra already exists when any BET is bought.
2. Solve dp_first: this vehicle is the designated first buyer; it pays
   construction + charger on its first BET purchase (when has_charger=False).
3. For each candidate first-buyer j, compute
       NPV_j = V_first[j](state_j, t=0) + sum_{i≠j} V_infra[i](state_i, t=0)
   Pick j* = argmin NPV_j.  Use dp_first for vehicle j* and dp_infra for the rest.

State space per vehicle
-----------------------
(tech ∈ {0,1}, age ∈ {1…max_age}, charger_slot ∈ {0,1}, t ∈ {0…T-1})
max_age is scenario-specific; ~1 440 states total — solved in milliseconds.

Usage
-----
    python per_vehicle_dp.py                   # all 5 scenarios, 200 episodes
    python per_vehicle_dp.py SQ                # Status Quo only
    python per_vehicle_dp.py S1 --episodes 500
"""

import argparse

import numpy as np
import gymnasium as gym

import fleetreplacement_env          # registers FleetReplacement-v0
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config
from fleetreplacement_env.envs.costs import compute_step_cost


# ---------------------------------------------------------------------------
# Scenario name map  (tag → CSV "name" column)
# ---------------------------------------------------------------------------

_SCENARIO_MAP: dict[str, str] = {
    "SQ": "Status Quo",
    "S1": "Scenario 1: Tech Stalemate",
    "S2": "Scenario 2: Tech without Mandate",
    "S3": "Scenario 3: Ambition meets Reality",
    "S4": "Scenario 4: Autonomous Green Logistics",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class VehicleDP:
    """
    Single-vehicle backward-induction result.

    Attributes
    ----------
    T                  : planning horizon (steps)
    max_age            : smallest age at which force-replace triggers
    fleet_charger_exists: flag used when solving this DP
    V      : ndarray (T+1, 2, max_age+1, 2)  discounted NPV-to-go
                 axis 0  t ∈ {0 … T}
                 axis 1  tech ∈ {0=DT, 1=BET}
                 axis 2  age  ∈ {0 … max_age}  (index 0 unused)
                 axis 3  charger_slot ∈ {0, 1}
    policy : ndarray (T, 2, max_age+1, 2)    optimal action ∈ {0, 1, 2}
    """
    __slots__ = ("T", "max_age", "fleet_charger_exists", "V", "policy")

    def __init__(
        self,
        T: int,
        max_age: int,
        fleet_charger_exists: bool,
        V: np.ndarray,
        policy: np.ndarray,
    ):
        self.T = T
        self.max_age = max_age
        self.fleet_charger_exists = fleet_charger_exists
        self.V = V
        self.policy = policy


class FleetDPResult:
    """
    Two-pass fleet DP result.

    Attributes
    ----------
    dp_infra   : VehicleDP solved with fleet_charger_exists=True
    dp_first   : VehicleDP solved with fleet_charger_exists=False
    first_buyer: index of vehicle designated as first BET buyer (-1 = no improvement)
    total_npv  : estimated total fleet discounted cost for the chosen assignment
    """
    __slots__ = ("dp_infra", "dp_first", "first_buyer", "total_npv")

    def __init__(
        self,
        dp_infra: VehicleDP,
        dp_first: VehicleDP,
        first_buyer: int,
        total_npv: float,
    ):
        self.dp_infra = dp_infra
        self.dp_first = dp_first
        self.first_buyer = first_buyer
        self.total_npv = total_npv


# ---------------------------------------------------------------------------
# Helper: force-replace age
# ---------------------------------------------------------------------------

def _force_replace_age(cfg: FleetEnvConfig) -> int:
    """
    Smallest age at which the force-replace condition triggers (keep is blocked).

    Mirrors the condition in fleet_replacement.py:
        age + 1 >= max_vehicle_age  OR  (age+1)*akt_base >= max_lifetime_km
    """
    age = 1
    while True:
        if (
            age + 1 >= cfg.cost.max_vehicle_age
            or (age + 1) * cfg.cost.akt_base >= cfg.cost.max_lifetime_km
        ):
            return age
        age += 1


# ---------------------------------------------------------------------------
# Core DP: single-vehicle backward induction
# ---------------------------------------------------------------------------

def run_vehicle_dp(cfg: FleetEnvConfig, fleet_charger_exists: bool) -> VehicleDP:
    """
    Backward induction DP for a single vehicle slot.

    Action masking mirrors fleet_replacement.py action_masks() exactly:
      - keep  (0): blocked when force-replace condition holds
      - DT    (1): blocked for brand-new vehicles (age == 1) or after dt_ban_year
      - BET   (2): blocked for brand-new vehicles (age == 1)

    Infrastructure cost for BET purchase (action 2) with has_charger=False:
      - fleet_charger_exists=True  → pays charger_price only   (n_charger=1 assumed)
      - fleet_charger_exists=False → pays construction+charger  (n_charger=0 assumed)
    When has_charger=True (slot already has a charger), infra cost is 0.

    Parameters
    ----------
    cfg                  : FleetEnvConfig
    fleet_charger_exists : infrastructure assumption (see above)
    """
    T       = cfg.mdp.planning_horizon
    max_age = _force_replace_age(cfg)

    raw_ban     = cfg.cost.dt_ban_year - cfg.mdp.start_year
    dt_ban_step = max(0, min(raw_ban, T))

    # V[t, tech, age, charger]  shape (T+1, 2, max_age+1, 2)
    # age index 0 is intentionally left at inf (never accessed)
    V      = np.full((T + 1, 2, max_age + 1, 2), np.inf)
    policy = np.full((T,     2, max_age + 1, 2), -1, dtype=np.int8)

    # Terminal: zero value after the planning horizon
    V[T] = 0.0

    for t in range(T - 1, -1, -1):
        dt_banned    = (t >= dt_ban_step)
        current_year = cfg.mdp.start_year + t
        disc         = 1.0 / (1.0 + cfg.cost.i_rate) ** t

        for tech in range(2):
            for age in range(1, max_age + 1):
                for charger in range(2):

                    force_replace = (
                        age + 1 >= cfg.cost.max_vehicle_age
                        or (age + 1) * cfg.cost.akt_base >= cfg.cost.max_lifetime_km
                    )
                    brand_new = (age == 1)

                    # n_charger to pass to compute_step_cost for action=2:
                    #   has_charger=True  → infra cost is 0 regardless (keep n_charger=0)
                    #   has_charger=False, infra exists  → n_charger=1 (charger_price)
                    #   has_charger=False, no infra      → n_charger=0 (construction+charger)
                    n_charger_cost = 0 if charger else (1 if fleet_charger_exists else 0)

                    best_val = np.inf
                    best_act = -1

                    for act in range(3):
                        # ---- action masks ----
                        if act == 0 and force_replace:
                            continue
                        if act == 1 and (brand_new or dt_banned):
                            continue
                        if act == 2 and brand_new:
                            continue

                        cost = compute_step_cost(
                            tech=tech,
                            age=float(age),
                            action=act,
                            annual_km=cfg.cost.akt_base,
                            cfg=cfg.cost,
                            current_year=current_year,
                            has_charger=bool(charger),
                            n_charger=n_charger_cost,
                        ).total

                        # ---- state transition ----
                        if act == 0:
                            nt, na, nc = tech, age + 1, charger
                        elif act == 1:
                            nt, na, nc = 0, 1, charger      # charger slot persists
                        else:                               # act == 2
                            nt, na, nc = 1, 1, 1            # slot gets a charger

                        val = disc * cost + V[t + 1, nt, na, nc]
                        if val < best_val:
                            best_val = val
                            best_act = act

                    # Safety fallback: age==1 AND force_replace (theoretically
                    # impossible given max_vehicle_age >> 1, but mirrors env logic)
                    if best_act == -1:
                        cost = compute_step_cost(
                            tech=tech,
                            age=float(age),
                            action=2,
                            annual_km=cfg.cost.akt_base,
                            cfg=cfg.cost,
                            current_year=current_year,
                            has_charger=bool(charger),
                            n_charger=n_charger_cost,
                        ).total
                        best_val = disc * cost + V[t + 1, 1, 1, 1]
                        best_act = 2

                    V[t, tech, age, charger]      = best_val
                    policy[t, tech, age, charger] = best_act

    return VehicleDP(
        T=T,
        max_age=max_age,
        fleet_charger_exists=fleet_charger_exists,
        V=V,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Fleet-level: two-pass charger coupling
# ---------------------------------------------------------------------------

def solve_fleet(
    cfg: FleetEnvConfig,
    fleet_state: np.ndarray,
    charger_slots: np.ndarray,
    dp_infra: VehicleDP | None = None,
    dp_first: VehicleDP | None = None,
) -> FleetDPResult:
    """
    Two-pass charger-coupling assignment.

    Computes (or reuses) two single-vehicle DPs, then selects the vehicle
    whose designation as "first BET buyer" minimises total fleet NPV at t=0.

    Parameters
    ----------
    cfg           : FleetEnvConfig
    fleet_state   : (n_vehicles, 3) array [tech, age, mileage] at episode start
    charger_slots : (n_vehicles,) bool array of charger installations
    dp_infra      : pre-computed VehicleDP(fleet_charger_exists=True)  — optional
    dp_first      : pre-computed VehicleDP(fleet_charger_exists=False) — optional
    """
    if dp_infra is None:
        dp_infra = run_vehicle_dp(cfg, fleet_charger_exists=True)
    if dp_first is None:
        dp_first = run_vehicle_dp(cfg, fleet_charger_exists=False)

    n = cfg.mdp.n_vehicles

    def _v0(dp: VehicleDP, i: int) -> float:
        tech    = int(fleet_state[i, 0])
        age     = int(round(fleet_state[i, 1]))
        charger = int(charger_slots[i])
        age     = max(1, min(age, dp.max_age))  # clamp to valid DP range
        return float(dp.V[0, tech, age, charger])

    total_infra = sum(_v0(dp_infra, i) for i in range(n))

    # Try every vehicle as the designated first BET buyer.
    # Construction cost MUST be paid by whoever triggers it, so we always
    # designate the vehicle whose extra burden (V_first - V_infra) is smallest.
    # That vehicle uses dp_first; all others use dp_infra.
    npv_candidates = [
        (_v0(dp_first, j) + total_infra - _v0(dp_infra, j), j)
        for j in range(n)
    ]
    best_npv, best_j = min(npv_candidates)

    return FleetDPResult(
        dp_infra=dp_infra,
        dp_first=dp_first,
        first_buyer=best_j,
        total_npv=best_npv,
    )


# ---------------------------------------------------------------------------
# Policy lookup: closed-loop action at time t
# ---------------------------------------------------------------------------

def get_dp_action(
    result: FleetDPResult,
    fleet_state: np.ndarray,
    charger_slots: np.ndarray,
    t: int,
) -> np.ndarray:
    """
    Read the optimal action for each vehicle from the pre-computed policy tables.

    Vehicle `result.first_buyer` uses dp_first; all others use dp_infra.

    Parameters
    ----------
    result        : FleetDPResult from solve_fleet()
    fleet_state   : (n_vehicles, 3) current [tech, age, mileage]
    charger_slots : (n_vehicles,) current charger installations
    t             : current time step (0-indexed)

    Returns
    -------
    action : (n_vehicles,) int32 array, values in {0, 1, 2}
    """
    n      = fleet_state.shape[0]
    action = np.empty(n, dtype=np.int32)

    for i in range(n):
        tech    = int(fleet_state[i, 0])
        age     = int(round(fleet_state[i, 1]))
        charger = int(charger_slots[i])

        dp  = result.dp_first if i == result.first_buyer else result.dp_infra
        age = max(1, min(age, dp.max_age))

        action[i] = int(dp.policy[t, tech, age, charger])

    return action


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dp_policy(
    scenario_tag: str,
    n_episodes: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run n_episodes in the environment under the per-vehicle DP policy and
    return summary statistics.

    The two VehicleDPs are solved once per scenario.  For each episode the
    fleet is reset (random composition), solve_fleet() picks the best
    first-buyer assignment using the pre-computed V tables, and actions are
    read from the policy arrays.

    Parameters
    ----------
    scenario_tag : one of SQ / S1 / S2 / S3 / S4
    n_episodes   : number of Monte-Carlo evaluation episodes
    seed         : master RNG seed

    Returns
    -------
    dict with keys scenario, mean_reward, std_reward, min_reward, max_reward
    """
    scenario_name = _SCENARIO_MAP[scenario_tag]
    cfg = FleetEnvConfig(
        mdp  = MDPConfig(),
        cost = load_cost_config(scenario_name=scenario_name),
    )

    # Solve both DPs once
    dp_infra = run_vehicle_dp(cfg, fleet_charger_exists=True)
    dp_first = run_vehicle_dp(cfg, fleet_charger_exists=False)

    env = gym.make("FleetReplacement-v0", config=cfg)
    rng = np.random.default_rng(seed)

    rewards: list[float] = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        env.reset(seed=ep_seed)

        # Pick first-buyer assignment for this episode's initial fleet
        fleet_state   = env.unwrapped.fleet_state.copy()
        charger_slots = env.unwrapped.charger_slots.copy()
        result = solve_fleet(cfg, fleet_state, charger_slots, dp_infra, dp_first)

        total_reward = 0.0
        done         = False
        t            = 0

        while not done:
            fleet_state   = env.unwrapped.fleet_state.copy()
            charger_slots = env.unwrapped.charger_slots.copy()
            action        = get_dp_action(result, fleet_state, charger_slots, t)

            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done  = terminated or truncated
            t    += 1

        rewards.append(total_reward)

    env.close()

    arr = np.array(rewards)
    return {
        "scenario":    scenario_tag,
        "mean_reward": float(arr.mean()),
        "std_reward":  float(arr.std()),
        "min_reward":  float(arr.min()),
        "max_reward":  float(arr.max()),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-vehicle DP solver for the fleet replacement MDP"
    )
    parser.add_argument(
        "scenario",
        choices=list(_SCENARIO_MAP.keys()) + ["ALL"],
        nargs="?",
        default="ALL",
        help="Scenario tag to evaluate (default: ALL)",
    )
    parser.add_argument(
        "--episodes", type=int, default=200, metavar="N",
        help="Episodes per scenario (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42)",
    )
    args = parser.parse_args()

    scenarios = list(_SCENARIO_MAP.keys()) if args.scenario == "ALL" else [args.scenario]

    print(f"\nPer-vehicle DP evaluation  —  {args.episodes} episodes/scenario")
    print("=" * 68)
    print(f"{'Scenario':<10} {'Mean NPV (€)':>16} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 68)

    for tag in scenarios:
        stats = evaluate_dp_policy(tag, n_episodes=args.episodes, seed=args.seed)
        print(
            f"{stats['scenario']:<10}"
            f"{stats['mean_reward']:>16,.0f}"
            f"{stats['std_reward']:>12,.0f}"
            f"{stats['min_reward']:>12,.0f}"
            f"{stats['max_reward']:>12,.0f}"
        )

    print("=" * 68)
