# from enum import Enum
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}  # for rendering

    def __init__(self, config: FleetConfig | None = None, render_mode: str | None = None):
        super().__init__()

        self.config = config or FleetConfig()
        self.current_step = 0
        self.fleet_state: np.ndarray | None = None  # represent unitialized state, for guard in step()
        self.render_mode = render_mode              # for rendering

        # Unpack config calues
        n_vehicles = self.config.n_vehicles
        max_vehicle_age = self.config.max_vehicle_age
        max_mileage = self.config.max_mileage
        # annual_mileage = self.config.annual_mileage
        # salvage_value_base = self.config.salvage_value_base
        # salvage_depreciation = self.config.salvage_depreciation
        # purchase_cost = self.config.purchase_cost
        # fuel_cost_per_km = self.config.fuel_cost_per_km
        # base_maintenance_cost = self.config.base_maintenance_cost
        # maintenance_age_factor = self.config.maintenance_age_factor
        planning_horizon = self.config.planning_horizon
        
        # State space as box, matrix of shape (n_vehicles, 3 (technology, age, mileage))
        # After including current_step in state space: vector with n_vehicles*3+1 elements
        self.observation_space = spaces.Box(
            low  = np.zeros(n_vehicles * 3 + 1, dtype=np.float32),                                                                                
            high = np.ones( n_vehicles * 3 + 1, dtype=np.float32),
            dtype = np.float32
        )

        # Action space
        self.action_space = spaces.MultiBinary(n_vehicles)              # simple for now, 0=keep, 1=replace with BET
        # self.action_space = spaces.MultiDiscrete([3] * n_vehicles)    # future alternative? 0=keep, 1=replace with DT, 2=replace with BET

    # Construting observations for NN
    def _get_obs(self):
        tech = self.fleet_state[:, 0]
        age = self.fleet_state[:, 1] / self.config.max_vehicle_age                                          # normalize
        mileage = self.fleet_state[:, 2] / self.config.max_mileage                                          # normalize
        flat_fleet = np.stack([tech, age, mileage], axis=1).flatten().astype(np.float32)        
        step_feature = np.array([self.current_step / self.config.planning_horizon], dtype=np.float32)       # normalize
        return np.concatenate([flat_fleet, step_feature])
    def _get_info(self):
        return {
            "step": self.current_step,
            "mean_age": float(self.fleet_state[:, 1].mean()),
            "mean_mileage": float(self.fleet_state[:, 2].mean()),
        }
    
    # Reset function, starts new episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0

        ages = self.np_random.integers(0, 10, size=self.config.n_vehicles).astype(np.float32)      # generate random vehicle age, convert to float (as defined in obs space)
        mileages = ages * self.config.annual_mileage                                               # starting mileage, derived from age                                            
        technologies = np.zeros(self.config.n_vehicles, dtype=np.float32)                          # 0 = diesel, all DT

        self.fleet_state = np.stack([technologies, ages, mileages], axis=1)            # combine above arrays to matrix of shape (n_vehicles, 3) to make columns parameters, rows vehicles
        return self._get_obs(), self._get_info()
    
    # Step function
    def step(self, action: np.ndarray):     # takes binary array from agent (1 = replace, 0 = keep, with n_vehicles length)
        assert self.fleet_state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=np.int32)
        total_cost = 0.0    # initialize cost

        for i in range(self.config.n_vehicles):
            tech, age, mileage = self.fleet_state[i]    # unpack row i (vehicle i) into three variables
            replace = bool(action[i])                   # convert binary action into true/false

            # Forced replacement if limits exceeded (regardless of action)
            # Returns true or false if one is true
            force_replace = (
                age + 1 >= self.config.max_vehicle_age
                or mileage + self.config.annual_mileage >= self.config.max_mileage
            )

            # TO DO: ADD BRANCH FOR BET COST CALCULATION
            if replace or force_replace:
                salvage = self.config.salvage_value_base * (self.config.salvage_depreciation ** age)        # value of DT when sold
                total_cost += self.config.purchase_cost - salvage                                           # cost of new truck
                self.fleet_state[i] = [1.0, 0.0, 0.0]                                                       # brand-new BET
            else:
                fuel_cost = self.config.annual_mileage * self.config.fuel_cost_per_km
                maintenance = self.config.base_maintenance_cost + self.config.maintenance_age_factor * age
                total_cost += fuel_cost + maintenance
                self.fleet_state[i] = [tech, age + 1, mileage + self.config.annual_mileage]

        self.current_step += 1
        reward = -total_cost    # agent's reward is negative cost
        truncated = self.current_step >= self.config.planning_horizon
        terminated = False  

        # Rendering with declared mode
        if self.render_mode == "human":
            self._render_frame(action, total_cost)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    # Rendering method
    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self, action=None, cost=None):
        print(f"\n── Step {self.current_step} ──────────────────────────────")
        print(f"{'#':<5} {'Tech':<8} {'Age':>5} {'Mileage':>10}  Action")
        for i, (tech, age, km) in enumerate(self.fleet_state):
            act = "REPLACE" if (action is not None and action[i]) else "keep"
            print(f"{i:<5} {'DT' if tech == 0 else 'BET':<8} {int(age):>5} {int(km):>10}  {act}")
        if cost is not None:
            print(f"\nTotal cost: €{cost:>12,.0f}   Reward: €{-cost:>12,.0f}")

    
    def close(self):
        pass