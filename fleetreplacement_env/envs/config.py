import pandas as pd
from dataclasses import dataclass

# MDP parameters for fleet_replacement.py
@dataclass
class MDPConfig:
    # MDP parameters with default values
    n_vehicles: int = 10            # fleet size
    max_vehicle_age: int = 20       # max truck age in years before forced retirement
    max_mileage: int = 500_000      # max truck mileage in km before forced retirement
    planning_horizon: int = 10      # planning time horizon in years (when is one training episode over?)

# Cost parameters for costs.py
@dataclass
class CostConfig:
    # Germany TCO parameters (loaded from CSVs)

    # --- countries.csv ---

    diesel_price: float 
    energy_price: float
    toll_ict: float
    toll_bet: float
    driver_wage: float
    tax: float
    maint_km: float
    tire_km: float
    insurance_base: float
    subsidy_perc: float
    subsidy_max: float

    # --- trucks.csv ---

    # row 1, 01_ICT_Manual_Multi
    capex_ict: float            # corresponds to capex_base
    consumption_ict: float      # corresponds to consumption, L/100km

    # row 5, 05_BET_Manual_Multi
    capex_bet_excl_bat: float   # corresnponds to capex_base (truck price excluding battery)
    bat_cap: float              # battery capacity
    price_kwh_base: float       # battery price
    consumption_bet: float      # consumption kWh/100km

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
    efficiency_factor_ict: float
    efficiency_factor_bet: float

    # Lifetime cap (from scenario; can align with MDPConfig.max_mileage)
    max_lifetime_km: float

    # Price scaling
    subsidy_fallback_perc: float
    subsidy_fallback_max: float
    diesel_price_factor: float
    energy_price_factor: float 
    toll_ict_factor: float
    toll_bet_multiplier: float   
    toll_bet_share_ict: float
    tax_factor: float

    # OpEx scaling
    maint_factor: float  # maint_manual_factor
    driver_wage_factor: float
    tire_factor: float
    insurance_factor: float

    # CapEx scaling
    capex_ict_factor: float
    capex_bet_factor: float

    # Residual value percentages
    residual_ict_perc: float
    residual_bet_truck_perc: float
    residual_bat_perc: float

# Composition
@dataclass
class FleetEnvConfig:
    mdp: MDPConfig
    cost: CostConfig

def load_config(
    countries_path="data/countries.csv",
    trucks_path="data/trucks.csv",
    **mdp_kwargs
) -> FleetEnvConfig:
    de = pd.read_csv(countries_path, sep=";", decimal=",")
    de = de[de["name"] == "Germany"].iloc[0]

    trucks = pd.read_csv(trucks_path, sep=";", decimal=",")
    dt_row  = trucks[trucks["type"] == "ICT"].iloc[0]
    bet_row = trucks[trucks["type"] == "BET"].iloc[0]

    return FleetEnvConfig(
        diesel_price=float(de["diesel_price"]),
        energy_price=float(de["energy_price"]),
        # ... rest of fields
        purchase_cost_dt=float(dt_row["capex_base"]),
        purchase_cost_bet=float(bet_row["capex_base"]),
        **mdp_kwargs   # n_vehicles, planning_horizon, etc.
    )
