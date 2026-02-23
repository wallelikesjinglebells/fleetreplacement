from enum import Enum
import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np
from dataclasses import dataclass

# Dataclass for flexible parameter adjustment
@dataclass
class FleetConfig:
    n_vehicles: int = 10              # fleet size
    max_vehicle_age: int = 20       # max truck age before forced retirement
    max_mileage: int = 500000       # max truck mileage before forced retirement
    planning_horizon: int = 10      # planning time horizon (when is one training episode over?)
    

class FleetReplacementEnv(gym.Env):
    
    def __init__(self, n_vehicles, config=None):
        super().__init__()
        self.config = config or FleetConfig()
        self.n_vehicles = n_vehicles

        # State space as box, matrix of shape (n_vehicles, 3 (technology, age, mileage))
        self.observation_space = spaces.Box(
            low = np.zeros((n_vehicles, 3), dtype = np.float32),    # lower bound
            high = np.ones((n_vehicles, 3), dtype = np.float32),    # upper bound (normalized → 1)
            dtype = np.float32
        )

        # Action space
        self.action_space = spaces.MultiBinary(n_vehicles)      # simple for now, 0=keep, 1=replace with BET
        # self.action_space = spaces.MultiDiscrete([3] * n_vehicles)    # future alternative? 0=keep, 1=replace with DT, 2=replace with BET

