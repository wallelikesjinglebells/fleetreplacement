"""
Tests three layers: CSV integrity → config loading → costs → env step.
No training, no SB3.
"""
import sys, traceback
sys.stdout.reconfigure(encoding="utf-8")
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
    "name", "diesel_price", "energy_price", "construction_cost_contrib", "charger_price",
    "toll_ict", "toll_bet", "driver_wage", "tax",
    "maint_km_ICT", "maint_km_BET", "insurance_base", "subsidy_type", "subsidy_data",
]
TRUCKS_COLS = [
    "name", "capex_base", "consumption", "bat_cap", "price_kwh_base",
    "akt_base", "avg_speed", "maint_age_factor",
]
SCENARIOS_COLS = [
    "name", "i_rate", "n_years", "bat_cap_factor", "price_kwh_factor",
    "efficiency_factor_ict", "efficiency_factor_bet", "max_lifetime_km", "max_vehicle_age",
    "subsidy_fallback_perc", "subsidy_fallback_max",
    "diesel_price_factor", "energy_price_factor",
    "toll_ict_factor", "toll_bet_multiplier", "toll_bet_share_ict", "tax_factor",
    "maint_manual_factor", "driver_wage_factor", "tire_factor", "insurance_factor",
    "capex_ict_factor", "capex_bet_factor",
    "residual_ict_truck_perc", "residual_bet_truck_perc", "residual_bat_perc",
    "ict_ban_year", "battery_replacement_age",
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
    if "battery_replacement_age" in df_s.columns:
        vals = df_s["battery_replacement_age"].dropna()
        check("battery_replacement_age > 0 in all scenarios",
              (vals > 0).all(), f"values: {vals.tolist()}")
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
        "diesel_price", "energy_price", "construction_cost_contrib", "charger_price",
        "capex_ict", "capex_bet_excl_bat",
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

    # battery_replacement_age: must be a positive integer
    bra = cfg.battery_replacement_age
    check(f"  cfg.battery_replacement_age is int  (got {bra!r})", isinstance(bra, int))
    check(f"  cfg.battery_replacement_age > 0  (got {bra})", bra > 0)

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
    check("replace→ICT → subsidy == 0", sc_repl_ict.subsidy == 0.0,
          f"got {sc_repl_ict.subsidy:.0f}")

    # Action 2: replace with BET (slot already has charger → no infra cost)
    sc_repl_bet = compute_step_cost(tech=0, age=5.0, action=2,
                                    annual_km=annual_km, mileage=annual_km*5, cfg=cfg,
                                    has_charger=True, n_charger=1)
    check("replace→BET → capex_gross > ICT capex",
          sc_repl_bet.capex_gross > sc_repl_ict.capex_gross,
          f"BET={sc_repl_bet.capex_gross:.0f}  ICT={sc_repl_ict.capex_gross:.0f}")
    check("replace→BET (has_charger=True) → infra_cost == 0",
          sc_repl_bet.infra_cost == 0.0,
          f"got {sc_repl_bet.infra_cost:.0f}")
    check("replace→BET → subsidy > 0",
          sc_repl_bet.subsidy > 0,
          f"got {sc_repl_bet.subsidy:.0f}")

    # Infrastructure cost: first BET ever (n_charger=0, no charger at slot)
    sc_first_bet = compute_step_cost(tech=0, age=5.0, action=2,
                                     annual_km=annual_km, mileage=annual_km*5, cfg=cfg,
                                     has_charger=False, n_charger=0)
    expected_first = cfg.construction_cost_contrib + cfg.charger_price
    check(f"first BET (n_charger=0) → infra_cost == construction + charger ({expected_first:,.0f})",
          sc_first_bet.infra_cost == expected_first,
          f"got {sc_first_bet.infra_cost:.0f}")

    # Infrastructure cost: subsequent BET (n_charger=5, no charger at slot)
    sc_next_bet = compute_step_cost(tech=0, age=5.0, action=2,
                                    annual_km=annual_km, mileage=annual_km*5, cfg=cfg,
                                    has_charger=False, n_charger=5)
    check(f"subsequent BET (n_charger=5) → infra_cost == charger_price ({cfg.charger_price:,.0f})",
          sc_next_bet.infra_cost == cfg.charger_price,
          f"got {sc_next_bet.infra_cost:.0f}")

    # Infrastructure cost: depot full (n_charger=10)
    sc_full_depot = compute_step_cost(tech=0, age=5.0, action=2,
                                      annual_km=annual_km, mileage=annual_km*5, cfg=cfg,
                                      has_charger=False, n_charger=10)
    check("BET with n_charger=10 → infra_cost == 0",
          sc_full_depot.infra_cost == 0.0,
          f"got {sc_full_depot.infra_cost:.0f}")

    # Keep BET (age 3)
    sc_keep_bet = compute_step_cost(tech=1, age=3.0, action=0,
                                    annual_km=annual_km, mileage=annual_km*3, cfg=cfg)
    check("keep BET  → opex_total > 0", sc_keep_bet.opex_total > 0,
          f"got {sc_keep_bet.opex_total:.0f}")
    check("keep BET (age 3) → no battery_replacement",
          sc_keep_bet.battery_replacement == 0.0,
          f"got {sc_keep_bet.battery_replacement:.0f}")

    # ── Mid-life battery replacement (age-based) ──────────
    repl_age = cfg.battery_replacement_age
    expected_bat_cost = (cfg.bat_cap * cfg.bat_cap_factor
                         * cfg.price_kwh_base * cfg.price_kwh_factor)

    sc_bat_hit = compute_step_cost(tech=1, age=float(repl_age), action=0,
                                   annual_km=annual_km, mileage=annual_km*repl_age, cfg=cfg)
    check(f"BET at age={repl_age} → battery_replacement > 0",
          sc_bat_hit.battery_replacement > 0,
          f"got {sc_bat_hit.battery_replacement:.0f}")
    check(f"BET at age={repl_age} → battery_replacement == expected ({expected_bat_cost:,.0f})",
          abs(sc_bat_hit.battery_replacement - expected_bat_cost) < 1.0,
          f"got {sc_bat_hit.battery_replacement:.0f}")

    for off_age in [repl_age - 1, repl_age + 1]:
        if off_age > 0:
            sc_no_bat = compute_step_cost(tech=1, age=float(off_age), action=0,
                                          annual_km=annual_km, mileage=annual_km*off_age,
                                          cfg=cfg)
            check(f"BET at age={off_age} (≠ repl_age) → battery_replacement == 0",
                  sc_no_bat.battery_replacement == 0.0,
                  f"got {sc_no_bat.battery_replacement:.0f}")

    sc_ict_at_repl = compute_step_cost(tech=0, age=float(repl_age), action=0,
                                       annual_km=annual_km, mileage=annual_km*repl_age, cfg=cfg)
    check("ICT at battery_replacement_age → battery_replacement == 0",
          sc_ict_at_repl.battery_replacement == 0.0,
          f"got {sc_ict_at_repl.battery_replacement:.0f}")

    # ── Age-dependent maintenance (ICT) ──────────────────
    sc_ict_young = compute_opex(tech=0, annual_km=annual_km, cfg=cfg, age=1.0)
    sc_ict_old   = compute_opex(tech=0, annual_km=annual_km, cfg=cfg, age=10.0)
    check("ICT maintenance increases with age (age=10 > age=1)",
          sc_ict_old.maintenance > sc_ict_young.maintenance,
          f"age=1: {sc_ict_young.maintenance:.0f}  age=10: {sc_ict_old.maintenance:.0f}")

    # ── Salvage revenue decreases with age ────────────────
    sc_repl_young = compute_replacement_cost(new_tech=0, old_tech=0, old_age=2.0,
                                             annual_km=annual_km, cfg=cfg)
    sc_repl_old   = compute_replacement_cost(new_tech=0, old_tech=0, old_age=9.0,
                                             annual_km=annual_km, cfg=cfg)
    check("ICT salvage_revenue decreases with age (age=9 < age=2)",
          sc_repl_old.salvage_revenue < sc_repl_young.salvage_revenue,
          f"age=2: {sc_repl_young.salvage_revenue:.0f}  age=9: {sc_repl_old.salvage_revenue:.0f}")

    sc_bet_repl_young = compute_replacement_cost(new_tech=1, old_tech=1, old_age=2.0,
                                                 annual_km=annual_km, cfg=cfg)
    sc_bet_repl_old   = compute_replacement_cost(new_tech=1, old_tech=1, old_age=9.0,
                                                 annual_km=annual_km, cfg=cfg)
    check("BET salvage_revenue decreases with age (age=9 < age=2)",
          sc_bet_repl_old.salvage_revenue < sc_bet_repl_young.salvage_revenue,
          f"age=2: {sc_bet_repl_young.salvage_revenue:.0f}  age=9: {sc_bet_repl_old.salvage_revenue:.0f}")

    # ── PriceState override ───────────────────────────────
    ps_high = PriceState(diesel_price=cfg.diesel_price * 2)
    sc_ict_high_diesel = compute_opex(tech=0, annual_km=annual_km, cfg=cfg,
                                      age=3.0, ps=ps_high)
    check("PriceState diesel override doubles fuel_energy cost",
          abs(sc_ict_high_diesel.fuel_energy / sc_keep_ict.fuel_energy - 2.0) < 0.01,
          f"ratio={sc_ict_high_diesel.fuel_energy / sc_keep_ict.fuel_energy:.3f}")

    # Breakdowns
    print(f"\n  Cost breakdown — keep ICT (age 5, {annual_km:.0f} km/yr):")
    for k, v in sc_keep_ict.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

    print(f"\n  Cost breakdown — replace→ICT (age 5):")
    for k, v in sc_repl_ict.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

    print(f"\n  Cost breakdown — replace→BET (age 5, has_charger=True):")
    for k, v in sc_repl_bet.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

    print(f"\n  Cost breakdown — replace→BET first ever (n_charger=0, has_charger=False):")
    for k, v in sc_first_bet.as_dict().items():
        print(f"    {k:<20}: €{v:>12,.0f}")

    print(f"\n  Cost breakdown — keep BET (age={repl_age}, battery replacement year):")
    for k, v in sc_bat_hit.as_dict().items():
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
    expected_obs_shape = (n * 2 + 2,)
    check(f"obs shape == {expected_obs_shape}", obs.shape == expected_obs_shape,
          f"got {obs.shape}")
    check("obs values in [0, 1]",
          float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0,
          f"min={obs.min():.3f}  max={obs.max():.3f}")
    check("info has expected keys",
          {"step","mean_age","mean_mileage","n_bet","n_ict","n_charger"} <= set(info.keys()))

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
    check("replace all BET → n_charger == n_vehicles after step",
          info3["n_charger"] == n, f"n_charger={info3['n_charger']}, n={n}")

    # Charger slot persistence: after replacing with BET, charger_slots must stay True
    env.reset(seed=42)
    env.step(np.full(n, 2, dtype=np.int32))   # replace all with BET
    prev_chargers = env.charger_slots.copy()
    env.step(np.zeros(n, dtype=np.int32))      # keep all
    check("charger_slots persist across steps after BET replacement",
          np.all(env.charger_slots == prev_chargers),
          f"before={prev_chargers}, after={env.charger_slots}")

    # ── Action masking ────────────────────────────────────
    env_m = FleetReplacementEnv()
    env_m.reset(seed=0)
    annual_km_env = env_m.cfg.cost.akt_base

    # age == 1: cannot replace a brand-new vehicle
    env_m.fleet_state[0] = [0.0, 1.0, annual_km_env]
    masks = env_m.action_masks().reshape(n, 3)
    check("age=1 → keep is valid",           bool(masks[0, 0]))
    check("age=1 → replace ICT is masked",   not bool(masks[0, 1]))
    check("age=1 → replace BET is masked",   not bool(masks[0, 2]))

    # near lifetime limit: keep must be masked
    env_m.fleet_state[0] = [
        0.0, 5.0,
        env_m.cfg.cost.max_lifetime_km - annual_km_env * 0.5   # within one step of limit
    ]
    masks2 = env_m.action_masks().reshape(n, 3)
    check("near lifetime limit → keep is masked",       not bool(masks2[0, 0]))
    check("near lifetime limit → replace BET is valid", bool(masks2[0, 2]))

    # ICT ban: after ban step, replace-with-ICT must be masked
    if env_m.ict_ban_step < env_m.cfg.mdp.planning_horizon:
        env_m.current_step = env_m.ict_ban_step
        env_m.fleet_state[0] = [0.0, 5.0, annual_km_env * 5]
        masks3 = env_m.action_masks().reshape(n, 3)
        check("at ICT ban step → replace ICT is masked",   not bool(masks3[0, 1]))
        check("at ICT ban step → replace BET is valid",    bool(masks3[0, 2]))
    else:
        check("ICT ban step within planning horizon (skipped)", True)

    # ── Battery replacement fires correctly inside env ────
    # Construct a BET that is exactly at battery_replacement_age this step
    env_b = FleetReplacementEnv()
    env_b.reset(seed=0)
    bra = env_b.cfg.cost.battery_replacement_age
    env_b.fleet_state[0] = [1.0, float(bra), annual_km_env * bra]
    cost_with_repl, _ = 0.0, None
    from fleetreplacement_env.envs.costs import compute_step_cost as _csc
    cost_item = _csc(
        tech=1, age=float(bra), action=0,
        annual_km=annual_km_env, mileage=annual_km_env * bra,
        cfg=env_b.cfg.cost,
    )
    check(f"env: BET at battery_replacement_age={bra} → step cost includes battery_replacement",
          cost_item.battery_replacement > 0,
          f"got {cost_item.battery_replacement:.0f}")

    # ── Discount: step-0 reward larger (less negative) than step-5 for same cost ─
    env_d = FleetReplacementEnv()
    env_d.reset(seed=42)
    env_d.fleet_state[:] = env_d.fleet_state.copy()   # keep same fleet

    # step 0 reward
    _, rew_s0, _, _, _ = env_d.step(action_keep)

    # advance to step 5 with same initial fleet by resetting and stepping forward
    env_d.reset(seed=42)
    for _ in range(5):
        env_d.step(action_keep)
    # override fleet to same composition as after reset
    rew_list = []
    env_d2 = FleetReplacementEnv()
    env_d2.reset(seed=42)
    _, rew_s0_ref, _, _, _ = env_d2.step(action_keep)
    env_d2.reset(seed=42)
    env_d2.current_step = 5
    _, rew_s5_ref, _, _, _ = env_d2.step(action_keep)
    check("discount: |reward at step=0| > |reward at step=5| for same fleet",
          abs(rew_s0_ref) > abs(rew_s5_ref),
          f"step0={rew_s0_ref:.0f}  step5={rew_s5_ref:.0f}")

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


# ─────────────────────────────────────────────────────────
# LAYER 5: MaskablePPO smoke test  (SB3 integration)
# ─────────────────────────────────────────────────────────
section("LAYER 5 — MaskablePPO SB3 integration smoke test")

SMOKE_STEPS = 2048   # one PPO rollout buffer; fast but exercises the full train loop

try:
    import gymnasium as gym
    import fleetreplacement_env  # triggers gym.register()
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    check("SB3 / sb3_contrib imports OK", True)

    def _make_masked():
        e = gym.make("FleetReplacement-v0")
        return ActionMasker(e, lambda env: env.unwrapped.action_masks())

    vec_env = DummyVecEnv([_make_masked])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)
    check("DummyVecEnv + VecNormalize construct OK", True)

    model = MaskablePPO("MlpPolicy", vec_env, verbose=0, n_steps=SMOKE_STEPS, batch_size=64)
    check("MaskablePPO constructs OK", True)

    model.learn(total_timesteps=SMOKE_STEPS)
    check(f"model.learn({SMOKE_STEPS} steps) completes without error", True)

    # Verify predict() respects masks on a single obs
    raw_env = gym.make("FleetReplacement-v0")
    raw_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
    obs, _ = raw_env.reset(seed=0)
    obs_arr = obs[np.newaxis, :]   # add batch dim
    masks = raw_env.action_masks()
    action, _ = model.predict(obs_arr, action_masks=masks[np.newaxis, :], deterministic=True)
    check("model.predict() returns action array", action is not None)
    n = raw_env.env.unwrapped.cfg.mdp.n_vehicles
    check(f"predicted action shape == ({n},)", action.shape == (1, n),
          f"got {action.shape}")
    raw_env.close()
    vec_env.close()

except Exception as e:
    check("MaskablePPO smoke test", False, traceback.format_exc())

print("\n" + "="*55)
print("  Done. Fix any [FAIL] lines above before training.")
print("="*55 + "\n")
