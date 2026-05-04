"""
Configuration dataclasses and CSV loaders for the fleet replacement environment.

Defines MDPConfig (MDP hyperparameters), CostConfig (scenario cost tables), and FleetEnvConfig (combined config), plus helpers to load cost data from CSV.
"""

import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# Based on Clara's TCO model (https://gitlab.lrz.de/global-drive-liu-tum/tco-modeling)

# MDP parameters for fleet_replacement.py
@dataclass
class MDPConfig:
    # MDP parameters with default values
    n_vehicles: int = 10            # fleet size
    # max_possible_vehicle_age: int = 0       # max max_vehicle_age across all scenarios, used for obs normalization; set via load_max_vehicle_age()
    # max_possible_lifetime_km: float = 0.0   # max max_lifetime_km across all scenarios, used for obs normalization; set via load_max_lifetime_km()
    planning_horizon: int = 30      # planning time horizon in years (when is one training episode over?)
    start_year: int = 2026          # current year (needed for calculating DT purchase ban step)

# Cost parameters for costs.py
@dataclass
class CostConfig:
    # Germany TCO parameters (loaded from CSVs)

    # --- countries.csv ---

    diesel_price: float
    energy_price: float
    construction_cost_contrib: float
    charger_price: float
    toll_dt: float
    toll_bet: float
    driver_wage: float
    tax: float
    maint_km_dt: float
    maint_km_bet: float
    insurance_base: float
    subsidy_max: float

    # --- trucks.csv ---

    # row 1, 01_DT_Manual_Multi
    capex_dt: float            # corresponds to capex_base
    consumption_dt: float      # corresponds to consumption, L/100km
    maint_age_factor_dt: float # adapted from Emiliano et al. (2020), maintenance cost +13.4% per year of age for diesel bus

    # row 5, 05_BET_Manual_Multi
    capex_bet_excl_bat: float   # corresnponds to capex_base (truck price excluding battery)
    bat_cap: float              # battery capacity
    price_kwh_base: float       # battery price
    consumption_bet: float      # consumption kWh/100km
    maint_age_factor_bet: float # is 0 for now

    # shared params
    akt_base: float             # Clara: annual km for long-haul, no longer in MDPConfig (SSOT here)
    avg_speed: float
    # wage_factor: float        # is 1, omitted

    # --- scenarios.csv ---

    # Finance
    i_rate: float
    n_years: float

    # Battery params
    bat_cap_factor: float   
    price_kwh_factor: float

    # Efficiency scaling
    efficiency_factor_dt: float
    efficiency_factor_bet: float

    # Lifetime cap (from scenario; can align with MDPConfig.max_mileage)
    max_lifetime_km: float

    # Subsidy
    subsidy_perc: float
    subsidy_expiry_year: int
    subsidy_fallback_perc: float
    subsidy_fallback_max: float
    diesel_price_factor: float
    energy_price_factor: float 
    toll_dt_factor: float
    toll_bet_multiplier: float   
    toll_bet_share_dt: float
    tax_factor: float

    # OpEx scaling
    maint_factor: float  # maint_manual_factor
    driver_wage_factor: float
    insurance_factor: float

    # CapEx scaling
    capex_dt_factor: float
    capex_bet_factor: float

    # Residual value percentages
    residual_dt_perc: float
    residual_bet_truck_perc: float
    residual_bat_perc: float

    # DT ban year
    dt_ban_year: int

    # Max vehicle age before forced retirement (years)
    max_vehicle_age: int

    # Omitted
    # ee_energy_factor: float     # only activates for automated truck with EE in name
    # ro_energy_factor: float     # only activates for automated truck with RO in name            
    # ro_price_kwh_factor: float  # only activates for automated truck with RO in name

# Composition
@dataclass
class FleetEnvConfig:
    mdp: MDPConfig
    cost: CostConfig

# Load parameters
def load_cost_config(
    countries_path: str | Path = "data/countries.csv",
    trucks_path: str | Path = "data/trucks.csv",
    scenarios_path: str | Path = "data/scenarios.csv",
    scenario_name:  str        = "Status Quo",              # default
) -> CostConfig:
    
    # Load and filter COUNTRIES
    df_c = pd.read_csv(countries_path, sep=";", decimal=",")
    de   = df_c[df_c["name"] == "Germany"].iloc[0]

    # Load and filter TRUCKS (only looking at manual)
    df_t   = pd.read_csv(trucks_path, sep=";", decimal=",")
    dt     = df_t[df_t["name"] == "01_DT_Manual_Multi"].iloc[0]
    bet    = df_t[df_t["name"] == "05_BET_Manual_Multi"].iloc[0]

    # Load and filter SCENARIOS
    df_s  = pd.read_csv(scenarios_path, sep=";", decimal=",")
    scen  = df_s[df_s["name"] == scenario_name].iloc[0]

    # Function for getting float value from CSV
    def get_float(row, key: str, default: float = 0.0) -> float:
        val = row.get(key)
        return float(default) if (pd.isna(val) or val == "") else float(val)

    return CostConfig(

        # --- countries.csv ---
        diesel_price = get_float(de, "diesel_price"),
        energy_price = get_float(de, "energy_price"),
        construction_cost_contrib = get_float(de, "construction_cost_contrib"),
        charger_price = get_float(de, "charger_price"),
        toll_dt = get_float(de, "toll_dt"),
        toll_bet = get_float(de, "toll_bet"),
        driver_wage = get_float(de, "driver_wage"),
        tax = get_float(de, "tax"),
        maint_km_dt = get_float(de, "maint_km_DT"),
        maint_km_bet = get_float(de, "maint_km_BET"),
        insurance_base = get_float(de, "insurance_base"),
        subsidy_max = get_float(de, "subsidy_max"),

        # --- trucks.csv (DT) ---
        capex_dt = get_float(dt, "capex_base"),
        consumption_dt = get_float(dt, "consumption"),
        maint_age_factor_dt = get_float(dt, "maint_age_factor"),

        # --- trucks.csv (BET) ---
        capex_bet_excl_bat = get_float(bet, "capex_base"),
        bat_cap = get_float(bet, "bat_cap"),
        price_kwh_base = get_float(bet, "price_kwh_base"),
        consumption_bet = get_float(bet, "consumption"),
        maint_age_factor_bet = get_float(scen, "maint_age_factor_bet"),

        # --- trucks.csv (shared) ---
        akt_base = get_float(dt, "akt_base"),
        avg_speed = get_float(dt, "avg_speed"),

        # --- scenarios.csv ---
        i_rate = get_float(scen, "i_rate"),
        n_years = get_float(scen, "n_years"),
        bat_cap_factor = get_float(scen, "bat_cap_factor"),
        price_kwh_factor = get_float(scen, "price_kwh_factor"),
        efficiency_factor_dt = get_float(scen, "efficiency_factor_dt"),
        efficiency_factor_bet = get_float(scen, "efficiency_factor_bet"),
        max_lifetime_km = get_float(scen, "max_lifetime_km"),
        max_vehicle_age = int(get_float(scen, "max_vehicle_age")),
        subsidy_perc = get_float(scen, "subsidy_perc"),
        subsidy_expiry_year = int(get_float(scen, "subsidy_expiry_year")),
        subsidy_fallback_perc = get_float(scen, "subsidy_fallback_perc"),
        subsidy_fallback_max = get_float(scen, "subsidy_fallback_max"),
        diesel_price_factor = get_float(scen, "diesel_price_factor"),
        energy_price_factor = get_float(scen, "energy_price_factor"),
        toll_dt_factor = get_float(scen, "toll_dt_factor"),
        toll_bet_multiplier = get_float(scen, "toll_bet_multiplier"),
        toll_bet_share_dt = get_float(scen, "toll_bet_share_dt"),
        tax_factor = get_float(scen, "tax_factor"),
        maint_factor = get_float(scen, "maint_manual_factor"),
        driver_wage_factor = get_float(scen, "driver_wage_factor"),
        insurance_factor = get_float(scen, "insurance_factor"),
        capex_dt_factor = get_float(scen, "capex_dt_factor"),
        capex_bet_factor = get_float(scen, "capex_bet_factor"),
        residual_dt_perc = get_float(scen, "residual_dt_truck_perc"),
        residual_bet_truck_perc = get_float(scen, "residual_bet_truck_perc"),
        residual_bat_perc = get_float(scen, "residual_bat_perc"),
        dt_ban_year = int(get_float(scen, "dt_ban_year")),

    )

# --- NOT NEEDED ANYMORE, BUT KEPT IN CASE FOR FUTURE USE ---
def load_max_lifetime_km(
    scenarios_path: str | Path = "data/scenarios.csv",
) -> float:
    """
    Helper function to read all rows of scenarios.csv 
    Returns the maximum max_lifetime_km across all scenarios
    Ror use as a fixed normalization denominator in fleet_replacement.py
    """
    df = pd.read_csv(scenarios_path, sep=";", decimal=",")
    return float(df["max_lifetime_km"].max())

def load_max_vehicle_age(
    scenarios_path: str | Path = "data/scenarios.csv",
) -> int:
    """
    Helper function to read all rows of scenarios.csv
    Returns the maximum max_vehicle_age across all scenarios
    For use as a fixed normalization denominator in fleet_replacement.py
    """
    df = pd.read_csv(scenarios_path, sep=";", decimal=",")
    return int(df["max_vehicle_age"].max())
