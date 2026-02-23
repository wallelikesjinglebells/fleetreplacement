from enum import Enum
import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np
from dataclasses import dataclass

# Dataclass for flexible parameter adjustment
@dataclass
class FleetConfig:
    n_vehicles: int = 10            # fleet size
    max_vehicle_age: int = 20       # max truck age in years before forced retirement
    max_mileage: int = 500_000      # max truck mileage in km before forced retirement
    planning_horizon: int = 10      # planning time horizon in years (when is one training episode over?)
    annual_mileage: int = 50_000    # one vehicle's annual mileage in km

    # Simple cost parameters
    purchase_cost: float = 150_000          # purchase cost for BET in €
    base_maintenance_cost: float = 5_000    # maintenance cost per year in €
    maintenance_age_factor: float = 500     # extra cost per year of age in €
    fuel_cost_per_km: float = 0.35          # cost for diesel per km in €
    electricity_cost_per_km: float = 0.08   # cost for electricity per km in €
    salvage_value_base: float = 50_000      # new truck salvage value in €
    salvage_depreciation: float = 0.85      # factor per year of truck age

class FleetReplacementEnv(gym.Env):
    
    def __init__(self, config: FleetConfig | None = None):
        super().__init__()
        self.config = config or FleetConfig()
        n_vehicles = self.n_vehicles
        max_vehicle_age = self.max_vehicle_age
        max_mileage = self.max_mileage
        
        # State space as box, matrix of shape (n_vehicles, 3 (technology, age, mileage))
        self.observation_space = spaces.Box(
            low = np.zeros((n_vehicles, 3), dtype = np.float32),    # lower bound
            high = np.array([[1, max_vehicle_age, max_mileage]] * n_vehicles, dtype = np.float32),
            dtype = np.float32
        )

        # Action space
        self.action_space = spaces.MultiBinary(n_vehicles)      # simple for now, 0=keep, 1=replace with BET
        # self.action_space = spaces.MultiDiscrete([3] * n_vehicles)    # future alternative? 0=keep, 1=replace with DT, 2=replace with BET