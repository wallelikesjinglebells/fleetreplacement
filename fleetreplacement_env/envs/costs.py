from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import CostConfig

@dataclass
class PriceState:
    """
    STOCHASTICITY like Winkelmann et al. eq. (16), currently PLACEHOLDER, later: sample value from stochastic process
    Field left to None defaults to deterministic value from CostConfig
    """
    diesel_price: Optional[float] = None        # €/L
    energy_price: Optional[float] = None        # €/kWh
    capex_ict: Optional[float] = None           # € (gross truck price before factor)
    capex_bet_excl_bat: Optional[float] = None  # € (truck excl. battery, before factor)
    price_kwh: Optional[float] = None           # €/kWh (battery cell price, before factor)


@dataclass
class StepCost:
    """
    COST BREAKDOWN
    Adapted from Clara: class Truck → cost_breakdown
    """
    # CAPEX (replacement year only)
    capex_gross: float = 0.0        # purchase price of new vehicle (incl. battery for BET)
    subsidy: float = 0.0            # subsidy received on new BET (positive → reduces cost)
    salvage_revenue: float = 0.0    # current market value of retired vehicle (positive → reduces cost)

    # OPEX (keep year only)
    fuel_energy: float = 0.0        # diesel or electricity cost
    toll: float = 0.0
    maintenance: float = 0.0
    tires: float = 0.0
    driver: float = 0.0             # hours-based, using avg_speed
    insurance: float = 0.0
    tax: float = 0.0

    @property
    def capex_net(self) -> float:
        """Net capital outlay: gross CAPEX minus subsidy and salvage of old vehicle"""
        return self.capex_gross - self.subsidy - self.salvage_revenue

    @property
    def opex_total(self) -> float:
        """Total OPEX"""
        return (
            self.fuel_energy + self.toll + self.maintenance
            + self.tires + self.driver + self.insurance + self.tax
        )

    @property
    def total(self) -> float:
        """Scalar cost for this vehicle this year, negated in fleet_replacement.py for RL reward"""
        return self.capex_net + self.opex_total

    def as_dict(self) -> dict[str, float]:
        """Utility method for conversion into dictionary for debugging"""
        return {
            "capex_gross": self.capex_gross,
            "subsidy": -self.subsidy,           # displayed as negative (saving)
            "salvage_revenue": -self.salvage_revenue,  # displayed as negative (saving)
            "fuel_energy": self.fuel_energy,
            "toll": self.toll,
            "maintenance": self.maintenance,
            "tires": self.tires,
            "driver": self.driver,
            "insurance": self.insurance,
            "tax": self.tax,
            "total": self.total,
        }


def _diesel_price(cfg: CostConfig, ps: Optional[PriceState]) -> float:
    """
    Diesel price
    Adapted from Clara: class ICT → self.country.diesel_price * f_diesel
    """
    base = (ps.diesel_price if ps and ps.diesel_price is not None
            else cfg.diesel_price)
    return base * cfg.diesel_price_factor


def _energy_price(cfg: CostConfig, ps: Optional[PriceState]) -> float:
    """
    Energy price
    Adapted from Clara: class BET → self.country.energy_price * f_energy
    """
    base = (ps.energy_price if ps and ps.energy_price is not None
            else cfg.energy_price)
    return base * cfg.energy_price_factor


def _capex_ict_gross(cfg: CostConfig, ps: Optional[PriceState]) -> float:
    """
    CAPEX for ICT
    Adapted from Clara: class ICT → self.capex_truck = (retail_price * f_capex_powertrain) + (capex_auto_base * f_capex_auto) (no auto in my case)
    """
    base = (ps.capex_ict if ps and ps.capex_ict is not None
            else cfg.capex_ict)                                 # cfg.capex_ict = retail_price
    return base * cfg.capex_ict_factor                          # cfg.capex_ict_factor = f_capex_powertrain


def _capex_bet_truck_gross(cfg: CostConfig, ps: Optional[PriceState]) -> float:
    """
    CAPEX for BET truck body (without battery)
    Adapted from Clara: class BET → self.capex_truck = (price_excl_bat * f_capex_powertrain) + (capex_auto_base * f_capex_auto) (no auto in my case)
    """
    base = (ps.capex_bet_excl_bat if ps and ps.capex_bet_excl_bat is not None
            else cfg.capex_bet_excl_bat)                        # cfg.capex_bet_excl_bat = price_excl_bat
    return base * cfg.capex_bet_factor                          # capex_bet_factor = f_capex_powertrain


def _battery_cost(cfg: CostConfig, ps: Optional[PriceState]) -> float:
    """
    Effective battery pack cost applying both capacity and price scaling factors
    Adapted from Clara: class BET → self.battery_price = (bat_cap * f_bat_cap) * (price_kwh_base * f_price_kwh)
    """
    kwh_price = (ps.price_kwh if ps and ps.price_kwh is not None
                 else cfg.price_kwh_base)
    return cfg.bat_cap * cfg.bat_cap_factor * kwh_price * cfg.price_kwh_factor


def _market_value(
    tech: int,
    age: float,
    cfg: CostConfig,
    ps: Optional[PriceState] = None,
) -> float:
    """
    Current market value of a vehicle of given technology and age
    Geometric-degressive depreciation (Winkelmann)
    """
    n = max(cfg.n_years, 1.0)
    age_eff = age + 1     # new vehicle immediately has value of 1-year-old vehicle

    # ICT
    if tech == 0:
        capex = _capex_ict_gross(cfg, ps)
        val = capex * (cfg.residual_ict_perc ** (age_eff / n))
        return max(0.0, val)                                    # prevent negative values if a vehicle is held past its assumed lifetime

    else:  # BET: separate depreciation for truck body and battery
        capex_truck = _capex_bet_truck_gross(cfg, ps)
        capex_bat = _battery_cost(cfg, ps)
        val_truck = capex_truck * (cfg.residual_bet_truck_perc ** (age_eff / n))
        val_bat = capex_bat * (cfg.residual_bat_perc ** (age_eff / n))
        return max(0.0, val_truck + val_bat)
    

def _bet_subsidy(
    cfg: CostConfig,
    capex_bet_total: float,
    ps: Optional[PriceState] = None,
) -> float:
    """
    German BET purchase subsidy
    Adapted from Clara: class BET → calculate_subsidy
    """
    capex_ict_ref = _capex_ict_gross(cfg, ps)
    premium = max(0.0, capex_bet_total - capex_ict_ref)             # premium for BET (Clara: "diff"), prevent negative value if BET price is lower than ICT

    if cfg.subsidy_perc > 0 or cfg.subsidy_max > 0:
        return min(premium * cfg.subsidy_perc, cfg.subsidy_max)

    # Fallback (scenario with expired subsidy programme)
    return min(capex_bet_total * cfg.subsidy_fallback_perc, cfg.subsidy_fallback_max)


def compute_replacement_cost(
    new_tech: int,
    old_tech: int,
    old_age: float,
    annual_km: float,
    cfg: CostConfig,
    ps: Optional[PriceState] = None,
) -> StepCost:
    """
    Lump-sum cost for replacing a vehicle in the current timestep

    Parameters
    new_tech  : 0 = ICT, 1 = BET
    old_tech  : technology of the vehicle being retired
    old_age   : age of the vehicle being retired, in years
    cfg       : CostConfig (Germany, manual drivetrain)
    ps        : optional stochastic price overrides

    Returns
    StepCost 
    """
    cost = StepCost()

    if new_tech == 0:  # new ICT
        cost.capex_gross = _capex_ict_gross(cfg, ps)
        cost.subsidy = 0.0  # no purchase subsidy for diesel in Germany

    else:  # new BET
        capex_truck = _capex_bet_truck_gross(cfg, ps)
        capex_bat = _battery_cost(cfg, ps)
        cost.capex_gross = capex_truck + capex_bat
        cost.subsidy = _bet_subsidy(cfg, cost.capex_gross, ps)

    cost.salvage_revenue = _market_value(old_tech, old_age, cfg, ps)    # Winkelmann: eq. (15), age-dependent sales price is accounted for in state cost calculation

    # OPEX !=0 for new vehicle in replacement year
    opex = compute_opex(tech=new_tech, annual_km=annual_km, cfg=cfg, age = 0.0, ps=ps)
    cost.fuel_energy = opex.fuel_energy
    cost.toll = opex.toll
    cost.maintenance = opex.maintenance
    cost.tires = opex.tires
    cost.driver = opex.driver
    cost.insurance = opex.insurance
    cost.tax = opex.tax

    return cost


def compute_opex(
    tech: int,
    annual_km: float,
    cfg: CostConfig,
    age: float,
    ps: Optional[PriceState] = None,
) -> StepCost:
    """
    Annual operating cost for a vehicle that is kept (not replaced) this year.

    Parameters
    tech       : 0 = ICT, 1 = BET
    annual_km  : km driven this year, pass cfg.akt_base as default from the env
    cfg        : CostConfig
    ps         : optional stochastic price overrides

    Cost components adapted from Winkelmann eq. (15) and Clara:
    fuel/energy, toll, maintenance, tires, driver (hours-based), insurance, tax.

    FUTURE EXPANSIONS:
        - CO2 carbon cost (Winkelmann: monetarized with stochastic carbon price)
        - Age-dependent maintenance scaling (maint_km is flat per km)
        - Battery degradation/mid-life battery replacement for BET
    """
    cost = StepCost()
    hours = annual_km / cfg.avg_speed  # driving hours per year

    if tech == 0:  # ICT
        cost.fuel_energy = (                   # adapted from Clara: class ICT → self.cost_breakdown["Diesel"]
            annual_km
            * (cfg.consumption_ict / 100.0)    # L/km
            * cfg.efficiency_factor_ict        # scenario efficiency scaling
            * _diesel_price(cfg, ps)
        )
        cost.toll = annual_km * cfg.toll_ict * cfg.toll_ict_factor          # adapted from Clara: class ICT → self.cost_breakdown["Toll"]
        age_scale = 1.0 + cfg.maint_age_factor_ict * age                    # age-dependent maintenance cost
        cost.maintenance = annual_km * cfg.maint_km * cfg.maint_factor * age_scale      # adapted from Clara: class ICT → self.cost_breakdown["Maintenance"]
        cost.tires = annual_km * cfg.tire_km * cfg.tire_factor              # adapted from Clara: class ICT → self.cost_breakdown["Tires"]
        cost.driver = hours * cfg.driver_wage * cfg.driver_wage_factor      # adapted from Clara: class ICT → self.cost_breakdown["Driver"]
        cost.insurance = cfg.insurance_base * cfg.insurance_factor          # adapted from Clara: class ICT → self.cost_breakdown["Insurance"], base_ins = self.country.insurance_base * f_ins 
        cost.tax = cfg.tax * cfg.tax_factor                                 # adapted from Clara: class ICT → self.cost_breakdown["Tax"]

    else:  # BET
        cost.fuel_energy = (                  # Adapted from Clara: class BET → 
            annual_km
            * (cfg.consumption_bet / 100.0)   # kWh/km
            * cfg.efficiency_factor_bet
            * _energy_price(cfg, ps)          # adapted from Clara: class BET → electricity_cost = self.AKT * ((consumption_base / 100) * final_efficiency) * (self.country.energy_price * f_energy)
        )
        # BET toll: adapted from Clara 
        ict_toll_adj = cfg.toll_ict * cfg.toll_ict_factor               # adapted from Clara: class BET → future_ict_toll = self.country.toll_ict * f_toll_ict
        base_bet_toll = cfg.toll_bet * cfg.toll_bet_multiplier          # adapted from Clara: class BET → base_bet_toll = self.country.toll_bet * f_toll_mult
        floor_bet_toll = ict_toll_adj * cfg.toll_bet_share_ict          # adapted from Clara: class BET → min_bet_toll (= floor_bet_toll)
        cost.toll = annual_km * max(base_bet_toll, floor_bet_toll)      # adapted from Clara: class BET → self.cost_breakdown["Toll"], actual_toll_bet = max(base_bet_toll, min_bet_toll)
        age_scale = 1.0 + cfg.maint_age_factor_bet * age
        cost.maintenance = annual_km * cfg.maint_km * cfg.maint_factor * age_scale  # adapted from Clara: class BET → self.cost_breakdown["Maintenance"]
        cost.tires = annual_km * cfg.tire_km * cfg.tire_factor          # adapted from Clara: class BET → self.cost_breakdown["Tires"]
        cost.driver = hours * cfg.driver_wage * cfg.driver_wage_factor  # adapted from Clara: class BET → self.cost_breakdown["Driver"]
        cost.insurance = cfg.insurance_base * cfg.insurance_factor      # adapted from Clara: class BET → self.cost_breakdown["Insurance"]
        cost.tax = cfg.tax * cfg.tax_factor                             # adapted from Clara: class BET → self.cost_breakdown["Tax"]

    return cost


def compute_step_cost(
    tech: int,
    age: float,
    action: int,
    annual_km: float,
    cfg: CostConfig,
    ps: Optional[PriceState] = None,
) -> StepCost:
    """
    Unified entry point for fleet_replacement.py step()

    Parameters
    tech       : current vehicle technology (0=ICT, 1=BET)
    age        : current vehicle age in years
    action     : 0=keep, 1=replace with ICT, 2=replace with BET
    annual_km  : km driven this step, uses cfg.akt_base as default
    cfg        : CostConfig
    ps         : optional stochastic price overrides

    Returns
    StepCost (.total gives the scalar cost in fleet_replacement.py)
    """
    if action == 0:
        return compute_opex(tech=tech, annual_km=annual_km, cfg=cfg, age=age, ps=ps)
    elif action == 1:
        return compute_replacement_cost(
            new_tech=0, old_tech=int(tech), old_age=age, annual_km=annual_km, cfg=cfg, ps=ps
        )
    else:  # action == 2
        return compute_replacement_cost(
            new_tech=1, old_tech=int(tech), old_age=age, annual_km=annual_km, cfg=cfg, ps=ps
        )
