# V1 sanity check — updated parameters from scenarios.csv / costs.py
#
# Key changes vs v0:
#   - n_years = 5 for all scenarios (was 6/5/7/5/7)
#   - subsidy_perc is now scenario-dependent (was uniform 0.6)
#   - S2 res_dt: 0.20 → 0.26
#   - S4 res_dt: 0.20 → 0.26; res_bet/res_bat: 0.34 → 0.35
#   - Battery replacement event removed; replaced by maint_age_factor_bet
#     (age-dependent BET maintenance: 1 + maint_age_bet * age)
#   - maint_age_bet is now scenario-dependent (was 0.0 everywhere in v0)

scenarios = {
    "SQ":  dict(n_years=5,  capex_bet_f=1.0,  price_kwh_f=1.0,   capex_dt_f=1.0,  res_dt=0.27, res_bet=0.14, res_bat=0.14, i=0.025, diesel_f=1.0, energy_f=1.0,  toll_dt_f=1.0, toll_bet_mult=1, toll_bet_share=0.00, maint_f=1.0,   tax_f=1.0, ins_f=1.0, wage_f=1.0, eff_dt=1.0,  eff_bet=1.0,   max_km=1300000, max_age=10, maint_age_bet=0.1554, subsidy_perc=0.40),
    "S1":  dict(n_years=5,  capex_bet_f=1.2,  price_kwh_f=1.25,  capex_dt_f=1.1,  res_dt=0.27, res_bet=0.14, res_bat=0.14, i=0.030, diesel_f=1.0, energy_f=1.2,  toll_dt_f=1.0, toll_bet_mult=0, toll_bet_share=1.00, maint_f=1.15,  tax_f=1.2, ins_f=1.0, wage_f=1.0, eff_dt=1.1,  eff_bet=1.25,  max_km=1300000, max_age=10, maint_age_bet=0.1943, subsidy_perc=0.20),
    "S2":  dict(n_years=5,  capex_bet_f=0.9,  price_kwh_f=0.875, capex_dt_f=0.95, res_dt=0.26, res_bet=0.34, res_bat=0.34, i=0.030, diesel_f=1.0, energy_f=1.0,  toll_dt_f=1.0, toll_bet_mult=0, toll_bet_share=0.75, maint_f=0.925, tax_f=1.0, ins_f=1.0, wage_f=1.0, eff_dt=0.95, eff_bet=0.875, max_km=1560000, max_age=11, maint_age_bet=0.0945, subsidy_perc=0.30),
    "S3":  dict(n_years=5,  capex_bet_f=1.1,  price_kwh_f=1.125, capex_dt_f=1.05, res_dt=0.27, res_bet=0.14, res_bat=0.14, i=0.020, diesel_f=1.1, energy_f=1.0,  toll_dt_f=1.1, toll_bet_mult=0, toll_bet_share=0.50, maint_f=1.075, tax_f=1.0, ins_f=1.0, wage_f=1.0, eff_dt=1.05, eff_bet=1.125, max_km=1300000, max_age=10, maint_age_bet=0.1749, subsidy_perc=0.50),
    "S4":  dict(n_years=5,  capex_bet_f=0.8,  price_kwh_f=0.75,  capex_dt_f=0.9,  res_dt=0.26, res_bet=0.35, res_bat=0.35, i=0.020, diesel_f=1.1, energy_f=0.8,  toll_dt_f=1.1, toll_bet_mult=0, toll_bet_share=0.25, maint_f=0.85,  tax_f=1.0, ins_f=1.0, wage_f=1.0, eff_dt=0.9,  eff_bet=0.75,  max_km=1560000, max_age=11, maint_age_bet=0.0810, subsidy_perc=0.60),
}

diesel_price   = 1.66
energy_price   = 0.399
capex_dt_base  = 150000
capex_bet_excl = 191314.8
bat_cap        = 780
price_kwh_base = 139.34
consumption_dt = 35
consumption_bet= 140
akt_base       = 160000
avg_speed      = 80
toll_dt_base   = 0.348
toll_bet_base  = 0.0
driver_wage    = 27
tax            = 929
ins_base       = 3050
maint_dt       = 0.185
maint_bet      = 0.1324
maint_age_dt   = 0.134
# maint_age_bet is now scenario-dependent (in s["maint_age_bet"])
subsidy_max    = 300000
const_cost     = 37760
charger_price  = 70876


def battery_cost(s):
    return bat_cap * s["price_kwh_f"] * price_kwh_base

def capex_bet(s):
    return capex_bet_excl * s["capex_bet_f"] + battery_cost(s)

def capex_dt(s):
    return capex_dt_base * s["capex_dt_f"]

def subsidy(s):
    prem = max(0, capex_bet(s) - capex_dt(s))
    return min(prem * s["subsidy_perc"], subsidy_max)

def residual_dt(s, age):
    return capex_dt(s) * (s["res_dt"] ** (age / s["n_years"]))

def residual_bet(s, age):
    ct = capex_bet_excl * s["capex_bet_f"]
    cb = battery_cost(s)
    return ct * (s["res_bet"] ** (age / s["n_years"])) + cb * (s["res_bat"] ** (age / s["n_years"]))

def opex_dt(s, age):
    fuel  = akt_base * (consumption_dt / 100) * s["eff_dt"] * diesel_price * s["diesel_f"]
    toll  = akt_base * toll_dt_base * s["toll_dt_f"]
    maint = akt_base * maint_dt * s["maint_f"] * (1 + maint_age_dt * age)
    hours = akt_base / avg_speed
    drv   = hours * driver_wage * s["wage_f"]
    ins   = ins_base * s["ins_f"]
    t     = tax * s["tax_f"]
    return dict(fuel=fuel, toll=toll, maint=maint, driver=drv, ins=ins, tax=t,
                total=fuel + toll + maint + drv + ins + t)

def opex_bet(s, age):
    fuel  = akt_base * (consumption_bet / 100) * s["eff_bet"] * energy_price * s["energy_f"]
    dt_toll_adj   = toll_dt_base * s["toll_dt_f"]
    base_bet_toll = toll_bet_base * s["toll_bet_mult"]
    floor_bet     = dt_toll_adj * s["toll_bet_share"]
    toll  = akt_base * max(base_bet_toll, floor_bet)
    # Age-dependent maintenance (new in v1: maint_age_bet is scenario-specific)
    maint = akt_base * maint_bet * s["maint_f"] * (1 + s["maint_age_bet"] * age)
    hours = akt_base / avg_speed
    drv   = hours * driver_wage * s["wage_f"]
    ins   = ins_base * s["ins_f"]
    t     = tax * s["tax_f"]
    return dict(fuel=fuel, toll=toll, maint=maint, driver=drv, ins=ins, tax=t,
                total=fuel + toll + maint + drv + ins + t)


# CHECK 1: Annual OPEX comparison DT vs BET at age 1
print("=" * 80)
print("CHECK 1: Annual OPEX — DT vs BET at age 1")
print("=" * 80)
for name, s in scenarios.items():
    dt  = opex_dt(s, 1)
    bet = opex_bet(s, 1)
    diff = bet["total"] - dt["total"]
    print(f"\n  {name}")
    print(f"    {'Component':<16} {'DT':>10} {'BET':>10} {'Diff':>10}")
    print(f"    {'-'*48}")
    for k in ["fuel", "toll", "maint", "driver", "ins", "tax"]:
        d = dt.get(k, 0); b = bet.get(k, 0)
        print(f"    {k:<16} {d:>10,.0f} {b:>10,.0f} {b-d:>10,.0f}")
    print(f"    {'TOTAL':<16} {dt['total']:>10,.0f} {bet['total']:>10,.0f} {diff:>10,.0f}  ({bet['total']/dt['total']*100:.1f}% of DT)")


# CHECK 2: CAPEX breakdown
print("\n" + "=" * 80)
print("CHECK 2: CAPEX, subsidy, and first-install infrastructure")
print("=" * 80)
print(f"  {'':20} {'DT':>10} {'BET truck':>10} {'BET bat':>10} {'BET total':>10} {'Subsidy':>10} {'1st charger':>12} {'Net BET 1st':>12}")
print(f"  {'-'*90}")
for name, s in scenarios.items():
    dt_c  = capex_dt(s)
    ct    = capex_bet_excl * s["capex_bet_f"]
    cb    = battery_cost(s)
    bet_c = ct + cb
    sub   = subsidy(s)
    ch    = const_cost + charger_price
    net   = bet_c - sub + ch
    print(f"  {name:<20} {dt_c:>10,.0f} {ct:>10,.0f} {cb:>10,.0f} {bet_c:>10,.0f} {sub:>10,.0f} {ch:>12,.0f} {net:>12,.0f}")


# CHECK 3: Residual value as % of CAPEX by age
print("\n" + "=" * 80)
print("CHECK 3: Residual value as % of CAPEX (age 0–8)")
print("=" * 80)
ages = list(range(0, 9))
print(f"  {'':20} " + "  ".join(f"{'a='+str(a):>6}" for a in ages))
print(f"  {'-'*76}")
for name, s in scenarios.items():
    dt_c  = capex_dt(s)
    bet_c = capex_bet(s)
    dt_r  = [f"{residual_dt(s, a)/dt_c*100:6.1f}" for a in ages]
    bet_r = [f"{residual_bet(s, a)/bet_c*100:6.1f}" for a in ages]
    print(f"  {name+' DT':<20} " + "  ".join(dt_r))
    print(f"  {name+' BET':<20} " + "  ".join(bet_r))
    print()


# CHECK 4: Break-even DT->BET (replacing aged DT with new BET)
print("=" * 80)
print("CHECK 4: DT->BET break-even — replacing a DT at various ages")
print("  (OPEX savings = DT OPEX at that age minus BET OPEX at age 1)")
print("=" * 80)
print(f"  {'':20} {'DT age':>8} {'Net CAPEX':>12} {'OPEX save/yr':>14} {'Break-even':>12}")
print(f"  {'-'*70}")
for name, s in scenarios.items():
    for dt_age in [2, 5, 7]:
        salv    = residual_dt(s, dt_age)
        net_cap = capex_bet(s) - subsidy(s) - salv + (const_cost + charger_price)
        saving  = opex_dt(s, dt_age)["total"] - opex_bet(s, 1)["total"]
        be      = net_cap / saving if saving > 0 else float("inf")
        label   = name if dt_age == 2 else ""
        print(f"  {label:<20} {dt_age:>8} {net_cap:>12,.0f} {saving:>14,.0f} {be:>11.1f}y")
    print()


# CHECK 5: BET maintenance age-dependence (replaces battery replacement event from v0)
print("=" * 80)
print("CHECK 5: BET maintenance cost by age (battery degradation distributed annually)")
print("  (v1 removes one-time battery replacement; maint_age_bet scales BET maint with age)")
print("=" * 80)
print(f"  {'':20} {'maint_age_bet':>14} " + "  ".join(f"{'a='+str(a):>8}" for a in [1, 2, 3, 5, 7, 8]))
print(f"  {'-'*84}")
for name, s in scenarios.items():
    cols = []
    for age in [1, 2, 3, 5, 7, 8]:
        m = akt_base * maint_bet * s["maint_f"] * (1 + s["maint_age_bet"] * age)
        cols.append(f"{m:>8,.0f}")
    print(f"  {name:<20} {s['maint_age_bet']:>14.4f} " + "  ".join(cols))


# CHECK 6: Total undiscounted lifetime cost — hold until mileage limit
print("\n" + "=" * 80)
print("CHECK 6: Total undiscounted cost — hold new vehicle until mileage limit")
print("  (BET includes first charger install; salvage deducted at end)")
print("  (No battery replacement event in v1; age-dependent BET maintenance instead)")
print("=" * 80)
print(f"  {'':20} {'Max life':>10} {'DT total':>12} {'BET total':>12} {'BET-DT':>12} {'BET/DT%':>8}")
print(f"  {'-'*72}")
for name, s in scenarios.items():
    max_life = int(s["max_km"] / akt_base)

    dt_total = capex_dt(s)
    for age in range(1, max_life + 1):
        dt_total += opex_dt(s, age)["total"]
    dt_total -= residual_dt(s, max_life)

    bet_total = capex_bet(s) - subsidy(s) + const_cost + charger_price
    for age in range(1, max_life + 1):
        bet_total += opex_bet(s, age)["total"]
    bet_total -= residual_bet(s, max_life)

    diff = bet_total - dt_total
    print(f"  {name:<20} {max_life:>10} {dt_total:>12,.0f} {bet_total:>12,.0f} {diff:>12,.0f} {bet_total/dt_total*100:>7.1f}%")


# CHECK 7: BET->BET net replacement cost vs holding (age 2-8)
print("\n" + "=" * 80)
print("CHECK 7: BET->BET net replacement cost vs keeping (by retirement age)")
print("  Negative = profitable to cycle; Positive = better to hold")
print("=" * 80)
print(f"  {'':20} " + "  ".join(f"{'a='+str(a):>9}" for a in range(2, 9)))
print(f"  {'-'*84}")
for name, s in scenarios.items():
    bet_c = capex_bet(s)
    sub   = subsidy(s)
    cols  = []
    for age in range(2, 9):
        salv = residual_bet(s, age)
        net  = bet_c - sub - salv
        cols.append(f"{net:>9,.0f}")
    print(f"  {name:<20} " + "  ".join(cols))
