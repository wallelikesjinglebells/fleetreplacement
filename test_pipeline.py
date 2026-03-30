"""
Tests three layers: CSV integrity → config loading → costs → env step.
No training, no SB3.
"""
import sys, traceback
import pandas as pd
import ast
import numpy as np

PASS = "  [OK]"
FAIL = "  [FAIL]"

def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS}  {label}")
    else:
        print(f"{FAIL}  {label}" + (f"  → {detail}" if detail else ""))

errors = []

# ─────────────────────────────────────────────────────────
# LAYER 1: CSV files exist and contain expected rows/columns
# ─────────────────────────────────────────────────────────
section("LAYER 1 — CSV file integrity")

COUNTRIES_COLS = [
    "name", "diesel_price", "energy_price", "toll_ict", "toll_bet",
    "driver_wage", "tax", "maint_km", "tire_km", "insurance_base", "subsidy_data",
]
TRUCKS_COLS = [
    "name", "capex_base", "consumption", "bat_cap", "price_kwh_base",
    "akt_base", "avg_speed", "maint_age_factor",
]
SCENARIOS_COLS = [
    "name", "i_rate", "n_years", "bat_cap_factor", "price_kwh_factor",
    "efficiency_factor_ict", "efficiency_factor_bet", "max_lifetime_km",
    "subsidy_fallback_perc", "subsidy_fallback_max",
    "diesel_price_factor", "energy_price_factor",
    "toll_ict_factor", "toll_bet_multiplier", "toll_bet_share_ict", "tax_factor",
    "maint_manual_factor", "driver_wage_factor", "tire_factor", "insurance_factor",
    "capex_ict_factor", "capex_bet_factor",
    "residual_ict_truck_perc", "residual_bet_truck_perc", "residual_bat_perc",
    "ict_ban_year",
]

try:
    df_c = pd.read_csv("data/countries.csv", sep=";", decimal=",")
    check("countries.csv loads", True)
    for col in COUNTRIES_COLS:
        check(f"  column '{col}' present", col in df_c.columns,
              f"missing from {list(df_c.columns)}")
    de_rows = df_c[df_c["name"] == "Germany"]
    check("Germany row exists", len(de_rows) == 1,
          f"found {len(de_rows)} rows")
    if len(de_rows) == 1:
        sub_raw = de_rows.iloc[0]["subsidy_data"]
        try:
            sub = ast.literal_eval(sub_raw)
            check("subsidy_data parses as dict", isinstance(sub, dict),
                  f"got {type(sub)}")
            for k in ("percentage", "max_amount"):
                check(f"  subsidy_data['{k}'] present", k in sub,
                      f"keys found: {list(sub.keys())}")
        except Exception as e:
            check("subsidy_data ast.literal_eval", False, str(e))
except Exception as e:
    check("countries.csv loads", False, str(e))

try:
    df_t = pd.read_csv("data/trucks.csv", sep=";", decimal=",")
    check("trucks.csv loads", True)
    for col in TRUCKS_COLS:
        check(f"  column '{col}' present", col in df_t.columns,
              f"missing from {list(df_t.columns)}")
    check("01_ICT_Manual_Multi row exists",
          (df_t["name"] == "01_ICT_Manual_Multi").sum() == 1)
    check("05_BET_Manual_Multi row exists",
          (df_t["name"] == "05_BET_Manual_Multi").sum() == 1)
except Exception as e:
    check("trucks.csv loads", False, str(e))

try:
    df_s = pd.read_csv("data/scenarios.csv", sep=";", decimal=",")
    check("scenarios.csv loads", True)
    for col in SCENARIOS_COLS:
        check(f"  column '{col}' present", col in df_s.columns,
              f"missing from {list(df_s.columns)}")
    check("'Status Quo' scenario row exists",
          (df_s["name"] == "Status Quo").sum() == 1)
except Exception as e:
    check("scenarios.csv loads", False, str(e))


# ─────────────────────────────────────────────────────────
# LAYER 2: config.py  →  load_cost_config()
# ─────────────────────────────────────────────────────────
section("LAYER 2 — config.py: load_cost_config()")

try:
    from fleetreplacement_env.envs.config import (
        CostConfig, MDPConfig, FleetEnvConfig, load_cost_config
    )
    check("config.py imports OK", True)

    cfg = load_cost_config()
    check("load_cost_config() returns CostConfig", isinstance(cfg, CostConfig))

    # Spot-check a few fields that should be clearly > 0
    must_be_positive = [
        "diesel_price", "energy_price", "capex_ict", "capex_bet_excl_bat",
        "bat_cap", "price_kwh_base", "consumption_ict", "consumption_bet",
        "akt_base", "avg_speed", "n_years",
    ]
    for field in must_be_positive:
        val = getattr(cfg, field)
        check(f"  cfg.{field} > 0  (got {val:.4g})", val > 0)

    # Factors of 1.0 or nearby are reasonable; just check they are not NaN/zero
    factor_fields = [
        "diesel_price_factor", "energy_price_factor", "capex_ict_factor",
        "capex_bet_factor", "maint_factor", "driver_wage_factor",
        "tire_factor", "insurance_factor",
    ]
    for field in factor_fields:
        val = getattr(cfg, field)
        check(f"  cfg.{field} is finite & > 0  (got {val:.4g})",
              np.isfinite(val) and val > 0)

    mdp = MDPConfig()
    check("MDPConfig() constructs OK", mdp.n_vehicles == 10)
    full = FleetEnvConfig(mdp=mdp, cost=cfg)
    check("FleetEnvConfig() constructs OK", True)

except Exception as e:
    check("config.py / load_cost_config()", False, traceback.format_exc())


# ─────────────────────────────────────────────────────────
# LAYER 3: costs.py  →  compute_step_cost()
# ─────────────────────────────────────────────────────────
section("LAYER 3 — costs.py: compute_step_cost()")

try:
    from fleetreplacement_env.envs.costs import (
        compute_step_cost, compute_opex, compute_replacement_cost, PriceState
    )
    check("costs.py imports OK", True)

    from fleetreplacement_env.envs.config import load_cost_config
    cfg = load_cost_config()
    annual_km = cfg.akt_base

    # Action 0: keep ICT
    sc_keep_ict = compute_step_cost(tech=0, age=5.0, action=0,
                                    annual_km=annual_km, mileage=annual_km*5, cfg=cfg)
    check("keep ICT  → opex_total > 0",  sc_keep_ict.opex_total > 0,
          f"got {sc_keep_ict.opex_total:.0f}")
    check("keep ICT  → capex_gross == 0", sc_keep_ict.capex_gross == 0.0)

    # Action 1: replace with ICT
    sc_repl_ict = compute_step_cost(tech=0, age=5.0, action=1,
                                    annual_km=annual_km, mileage=annual_km*5, cfg=cfg)
    check("replace→ICT → capex_gross > 0", sc_repl_ict.capex_gross > 0,
          f"got {sc_repl_ict.capex_gross:.0f}")
    check("replace→ICT → salvage_revenue > 0 (age=5)", sc_repl_ict.salvage_revenue > 0,
          f"got {sc_repl_ict.salvage_revenue:.0f}")

    # Action 2: replace with BET
    sc_repl_bet = compute_step_cost(tech=0, age=5.0, action=2,
                                    annual_km=annual_km, mileage=annual_km*5, cfg=cfg)
    check("replace→BET → capex_gross > ICT capex",
          sc_repl_bet.capex_gross > sc_repl_ict.capex_gross,
          f"BET={sc_repl_bet.capex_gross:.0f}  ICT={sc_repl_ict.capex_gross:.0f}")

    # Keep BET (age 3)
    sc_keep_bet = compute_step_cost(tech=1, age=3.0, action=0,
                                    annual_km=annual_km, mileage=annual_km*3, cfg=cfg)
    check("keep BET  → opex_total > 0", sc_keep_bet.opex_total > 0,
          f"got {sc_keep_bet.opex_total:.0f}")

    # Breakdowns
    print(f"\n  Cost breakdown — keep ICT (age 5, {annual_km:.0f} km/yr):")
    for k, v in sc_keep_ict.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")
        
    print(f"\n  Cost breakdown — replace→ICT (age 5):")
    for k, v in sc_repl_ict.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

    print(f"\n  Cost breakdown — replace→BET (age 5):")
    for k, v in sc_repl_bet.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

except Exception as e:
    check("costs.py / compute_step_cost()", False, traceback.format_exc())


# ─────────────────────────────────────────────────────────
# LAYER 4: fleet_replacement.py  →  env reset + step
# ─────────────────────────────────────────────────────────
section("LAYER 4 — fleet_replacement.py: env reset + step")

try:
    from fleetreplacement_env.envs.fleet_replacement import FleetReplacementEnv
    check("fleet_replacement.py imports OK", True)

    env = FleetReplacementEnv()
    check("FleetReplacementEnv() constructs (loads CSV internally)", True)

    obs, info = env.reset(seed=42)
    n = env.cfg.mdp.n_vehicles
    expected_obs_shape = (n * 3 + 1,)
    check(f"obs shape == {expected_obs_shape}", obs.shape == expected_obs_shape,
          f"got {obs.shape}")
    check("obs values in [0, 1]",
          float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0,
          f"min={obs.min():.3f}  max={obs.max():.3f}")
    check("info has expected keys",
          {"step","mean_age","mean_mileage","n_bet","n_ict"} <= set(info.keys()))

    # Step: keep all
    action_keep = np.zeros(n, dtype=np.int32)
    obs2, rew, term, trunc, info2 = env.step(action_keep)
    check("step(keep all) → reward is negative float", rew < 0,
          f"reward = {rew:,.0f}")
    check("step(keep all) → not terminated after step 1", not term)
    check("step(keep all) → obs still in [0,1]",
          float(obs2.min()) >= 0.0 and float(obs2.max()) <= 1.0)

    # Step: replace all with BET
    env.reset(seed=42)
    action_bet = np.full(n, 2, dtype=np.int32)
    obs3, rew3, term3, trunc3, info3 = env.step(action_bet)
    check("step(replace all BET) → reward is negative float", rew3 < 0,
          f"reward = {rew3:,.0f}")
    check("replace all BET → n_bet == n_vehicles after step",
          info3["n_bet"] == n, f"n_bet={info3['n_bet']}, n={n}")

    # Full episode: runs to truncation at planning_horizon
    env.reset(seed=0)
    total_rew = 0.0
    done = False
    steps = 0
    while not done:
        a = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(a)
        total_rew += rew
        done = term or trunc
        steps += 1
    check(f"full episode runs {env.cfg.mdp.planning_horizon} steps",
          steps == env.cfg.mdp.planning_horizon, f"ran {steps} steps")
    check("full episode total reward is finite negative",
          np.isfinite(total_rew) and total_rew < 0,
          f"total reward = {total_rew:,.0f}")
    print(f"\n  Full episode reward (random policy): €{total_rew:,.0f}")

    env.close()

except Exception as e:
    check("fleet_replacement.py env", False, traceback.format_exc())

print("\n" + "="*55)
print("  Done. Fix any [FAIL] lines above before training.")
print("="*55 + "\n")
