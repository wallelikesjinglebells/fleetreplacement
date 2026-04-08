"""
Salvage value curves: current CSV values vs paper-calibrated values.
One plot per scenario, showing DT and BET curves for both parameterisations.
Output: salvage_comparison/PNGs/ and salvage_comparison/SVGs/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tum_colors import (
    TUM_BLUE, TUM_ORANGE, TUM_DARK_BLUE_1, TUM_DARK_BLUE_2,
    GRAY_50, GRAY_80, BLACK
)

# ── scenario definitions ───────────────────────────────────────────────────────
# n_years, current residual params, max operating age (km-limit / akt_base)
scenarios = {
    "Status Quo": dict(
        n_years=6,
        cur_dt=0.27,  cur_bet_truck=0.14, cur_bet_bat=0.14,
        paper_bet_pct5=0.14, paper_dt_pct5=0.27,
        max_age=8.1,
    ),
    "S1: Tech Stalemate": dict(
        n_years=5,
        cur_dt=0.27,  cur_bet_truck=0.14, cur_bet_bat=0.14,
        paper_bet_pct5=0.34, paper_dt_pct5=0.26,
        max_age=8.1,
    ),
    "S2: Tech without Mandate": dict(
        n_years=7,
        cur_dt=0.20,  cur_bet_truck=0.34, cur_bet_bat=0.34,
        paper_bet_pct5=0.34, paper_dt_pct5=0.26,
        max_age=9.8,
    ),
    "S3: Ambition meets Reality": dict(
        n_years=5,
        cur_dt=0.27,  cur_bet_truck=0.14, cur_bet_bat=0.14,
        paper_bet_pct5=0.34, paper_dt_pct5=0.26,
        max_age=8.1,
    ),
    "S4: Autonomous Green Logistics": dict(
        n_years=7,
        cur_dt=0.20,  cur_bet_truck=0.34, cur_bet_bat=0.34,
        paper_bet_pct5=0.35, paper_dt_pct5=0.26,
        max_age=9.8,
    ),
}

# Base CAPEX values (scenario-scaled versions of these are used for absolute €,
# but here we express everything as % of own CAPEX so scaling cancels out)
capex_bet_excl = 191314.8
bat_cap        = 780
price_kwh_base = 139.34
capex_dt_base  = 150000

CAPEX_BET_FACTORS = {
    "Status Quo":                   (1.0,  1.0),
    "S1: Tech Stalemate":           (1.2,  1.25),
    "S2: Tech without Mandate":     (0.9,  0.875),
    "S3: Ambition meets Reality":   (1.1,  1.125),
    "S4: Autonomous Green Logistics":(0.8, 0.75),
}
CAPEX_DT_FACTORS = {
    "Status Quo":                   1.0,
    "S1: Tech Stalemate":           1.1,
    "S2: Tech without Mandate":     0.95,
    "S3: Ambition meets Reality":   1.05,
    "S4: Autonomous Green Logistics":0.9,
}


def residual_from_pct5(pct5: float, n_years: int) -> float:
    """Back-calculate residual_perc so that pct5 is hit at age 5."""
    return pct5 ** (n_years / 5)


def bet_residual_pct(age: float, res_truck: float, res_bat: float,
                     n_years: int, capex_truck: float, capex_bat: float) -> float:
    """Combined BET residual as fraction of total CAPEX."""
    val = capex_truck * res_truck ** (age / n_years) + capex_bat * res_bat ** (age / n_years)
    return val / (capex_truck + capex_bat)


def dt_residual_pct(age: float, res_dt: float, n_years: int) -> float:
    return res_dt ** (age / n_years)


# ── plot ───────────────────────────────────────────────────────────────────────
fig_w, fig_h = 7, 4.5

for scen_name, s in scenarios.items():
    n      = s["n_years"]
    max_a  = s["max_age"]
    ages   = np.linspace(0, max_a, 300)

    # CAPEX components (only ratio matters for % curves, but keep split for BET)
    bet_f, kwh_f = CAPEX_BET_FACTORS[scen_name]
    capex_truck  = capex_bet_excl * bet_f
    capex_bat    = bat_cap * price_kwh_base * kwh_f

    # Current params
    cur_bet_r = s["cur_bet_truck"]   # same for truck and bat in current CSV
    cur_dt_r  = s["cur_dt"]

    # Paper-calibrated params
    new_bet_r = residual_from_pct5(s["paper_bet_pct5"], n)
    new_dt_r  = residual_from_pct5(s["paper_dt_pct5"], n)

    # Curves (as %)
    cur_bet_curve = [bet_residual_pct(a, cur_bet_r, cur_bet_r, n, capex_truck, capex_bat) * 100
                     for a in ages]
    new_bet_curve = [bet_residual_pct(a, new_bet_r, new_bet_r, n, capex_truck, capex_bat) * 100
                     for a in ages]
    cur_dt_curve  = [dt_residual_pct(a, cur_dt_r, n) * 100 for a in ages]
    new_dt_curve  = [dt_residual_pct(a, new_dt_r, n) * 100 for a in ages]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Current curves — solid
    ax.plot(ages, cur_dt_curve,  color=TUM_ORANGE, lw=2.0, ls="-",  label="DT — current")
    ax.plot(ages, cur_bet_curve, color=TUM_BLUE,   lw=2.0, ls="-",  label="BET — current")

    # Paper-calibrated curves — dashed
    ax.plot(ages, new_dt_curve,  color=TUM_ORANGE, lw=2.0, ls="--", label="DT — paper")
    ax.plot(ages, new_bet_curve, color=TUM_BLUE,   lw=2.0, ls="--", label="BET — paper")

    # Annotation: age-5 target markers
    ax.axvline(5, color=GRAY_50, lw=0.8, ls=":")
    ax.annotate("age 5\n(calibration\npoint)",
                xy=(5, 2), xytext=(5.2, 4), fontsize=7, color=GRAY_80)

    # Mark max operating age
    ax.axvline(max_a, color=GRAY_50, lw=0.8, ls=":")
    ax.annotate(f"max age\n({max_a:.1f} yr)",
                xy=(max_a, 2), xytext=(max_a - 1.9, 55), fontsize=7, color=GRAY_80)

    ax.set_xlim(0, max_a + 0.3)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Vehicle age (years)", fontsize=10)
    ax.set_ylabel("Residual value (% of CAPEX)", fontsize=10)
    ax.set_title(f"Salvage value curves — {scen_name}\n"
                 f"(n_years={n}, BET paper target: {s['paper_bet_pct5']*100:.0f}% "
                 f"at age 5, DT: {s['paper_dt_pct5']*100:.0f}% at age 5)",
                 fontsize=9)

    # Legend: two dimensions (vehicle type × parameterisation)
    leg_dt  = mlines.Line2D([], [], color=TUM_ORANGE, lw=2, label="DT")
    leg_bet = mlines.Line2D([], [], color=TUM_BLUE,   lw=2, label="BET")
    leg_cur = mlines.Line2D([], [], color=GRAY_80, lw=2, ls="-",  label="Current CSV")
    leg_new = mlines.Line2D([], [], color=GRAY_80, lw=2, ls="--", label="Paper-calibrated")
    ax.legend(handles=[leg_dt, leg_bet, leg_cur, leg_new],
              fontsize=8, loc="upper right", framealpha=0.9)

    ax.grid(axis="y", color=GRAY_50, alpha=0.3, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    slug = scen_name.replace(":", "").replace(" ", "_").replace("/", "")
    fig.savefig(f"salvage_comparison/PNGs/salvage_{slug}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"salvage_comparison/SVGs/salvage_{slug}.svg",          bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {slug}")

print("Done.")
