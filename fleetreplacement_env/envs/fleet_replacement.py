# from enum import Enum
import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np
# from dataclasses import dataclass
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config

class FleetReplacementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}  # for rendering

    def __init__(self, config: FleetEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()

        # If no config is pased, build a default config by loading from CSV files
        if config is None:
            config = FleetEnvConfig(
                mdp = MDPConfig(),
                cost = load_cost_config()     # call loading config
            )
        self.cfg = config                     # holds self.cfg.mdp and self.cfg.cost

        self.current_step = 0
        self.fleet_state: np.ndarray | None = None  # represent unitialized state, for guard in step()
        self.render_mode = render_mode              # for rendering

        # Unpack config calues
        n_vehicles = self.cfg.mdp.n_vehicles
        max_vehicle_age = self.cfg.mdp.max_vehicle_age
        max_mileage = self.cfg.mdp.max_mileage
        
        # State space as box, matrix of shape (n_vehicles, 3 (technology, age, mileage))
        # After including current_step in state space: vector with n_vehicles*3+1 elements
        self.observation_space = spaces.Box(
            low  = np.zeros(n_vehicles * 3 + 1, dtype=np.float32),                                                                                
            high = np.ones( n_vehicles * 3 + 1, dtype=np.float32),
            dtype = np.float32
        )

        # Action space
        # self.action_space = spaces.MultiBinary(n_vehicles)              # simple for now, 0=keep, 1=replace with BET
        self.action_space = spaces.MultiDiscrete([3] * n_vehicles)        # 0=keep, 1=replace with DT, 2=replace with BET

    # Constructing observations for NN
    def _get_obs(self):
        tech = self.fleet_state[:, 0]
        age = self.fleet_state[:, 1] / self.cfg.mdp.max_vehicle_age                                          # normalize
        mileage = self.fleet_state[:, 2] / self.cfg.mdp.max_mileage                                          # normalize
        flat_fleet = np.stack([tech, age, mileage], axis=1).flatten().astype(np.float32)        
        step_feature = np.array([self.current_step / self.cfg.mdp.planning_horizon], dtype=np.float32)       # normalize
        return np.concatenate([flat_fleet, step_feature])
    
    def _get_info(self):
        return {
            "step": self.current_step,
            "mean_age": float(self.fleet_state[:, 1].mean()),
            "mean_mileage": float(self.fleet_state[:, 2].mean()),
            "n_bet": int((self.fleet_state[:, 0] == 1).sum()),
            "n_dt":  int((self.fleet_state[:, 0] == 0).sum()),
        }
    
    # Reset function, starts new episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0

        ages = self.np_random.integers(0, 10, size=self.cfg.mdp.n_vehicles).astype(np.float32)      # generate random vehicle age, convert to float (as defined in obs space)
        mileages = ages * self.cfg.cost.akt_base                                         # starting mileage, derived from age                                            
        technologies = np.zeros(self.cfg.mdp.n_vehicles, dtype=np.float32)                          # 0 = diesel, all DT

        self.fleet_state = np.stack([technologies, ages, mileages], axis=1)            # combine above arrays to matrix of shape (n_vehicles, 3) to make columns parameters, rows vehicles

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()
    
    # Step function
    def step(self, action: np.ndarray):     # takes binary array from agent (1 = replace, 0 = keep, with n_vehicles length)
        assert self.fleet_state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=np.int32)
        total_cost = 0.0    # initialize cost

        resolved_action = np.empty(self.cfg.mdp.n_vehicles, dtype=np.int32)      # initializes array to store actual action (override or agent)

        for i in range(self.cfg.mdp.n_vehicles):
            tech, age, mileage = self.fleet_state[i]    # unpack row i (vehicle i) into three variables
            # replace = bool(action[i])                   # convert binary action into true/false
            act = int(action[i])                        # assign value of replacement to act (0 = keep, 1 = replace with DT, 2 = replace with BET)

            # Forced replacement if limits exceeded (regardless of action)
            # Returns true or false if one is true
            force_replace = (
                age + 1 >= self.cfg.mdp.max_vehicle_age
                or mileage + self.cfg.cost.akt_base >= self.cfg.mdp.max_mileage
            )

            # If agent says no replacement, but force_replace is true, default to replace with DT
            if force_replace and act == 0:
                act = 1

            resolved_action[i] = act

            # Replace with DT
            if act == 1:
                salvage = self.cfg.mdp.salvage_value_base * (self.cfg.mdp.salvage_depreciation ** age)
                total_cost += self.cfg.mdp.purchase_cost_dt - salvage
                self.fleet_state[i] = [0.0, 0.0, 0.0]

            # Replace with BET
            elif act == 2:
                salvage = self.cfg.mdp.salvage_value_base * (self.cfg.mdp.salvage_depreciation ** age)
                total_cost += self.cfg.mdp.purchase_cost_bet - salvage
                self.fleet_state[i] = [1.0, 0.0, 0.0]

            # Keep
            else:
                maintenance = self.cfg.mdp.base_maintenance_cost + self.cfg.mdp.maintenance_age_factor * age
                if tech == 0.0:     # keep DT
                    running_cost = self.cfg.cost.akt_base * self.cfg.mdp.fuel_cost_per_km
                else:               # keep BET
                    running_cost = self.cfg.cost.akt_base * self.cfg.mdp.electricity_cost_per_km
                total_cost += running_cost + maintenance
                self.fleet_state[i] = [tech, age + 1, mileage + self.cfg.cost.akt_base]

        self.current_step += 1
        reward = -total_cost    # agent's reward is negative cost
        truncated = self.current_step >= self.cfg.mdp.planning_horizon
        terminated = False  

        # Rendering with declared mode
        if self.render_mode == "human":
            self._render_frame(resolved_action, total_cost)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    # Rendering method
    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self, action=None, cost=None):
        act_labels = {0: "kept", 1: "→ DT", 2: "→ BET"}
        print(f"\n── Step {self.current_step} ──────────────────────────────")
        print(f"{'#':<5} {'Tech':<8} {'Age':>5} {'Mileage':>10}  Action")
        for i, (tech, age, km) in enumerate(self.fleet_state):
            act = act_labels.get(int(action[i]), "-") if action is not None else "-"
            print(f"{i:<5} {'DT' if tech == 0 else 'BET':<8} {int(age):>5} {int(km):>10}  {act}")
        if cost is not None:
            print(f"\nTotal cost: €{cost:>12,.0f}   Reward: €{-cost:>12,.0f}")

    
    def close(self):
        pass