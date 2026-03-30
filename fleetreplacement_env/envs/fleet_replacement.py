import gymnasium as gym
from gymnasium import spaces
import numpy as np
from fleetreplacement_env.envs.config import FleetEnvConfig, MDPConfig, load_cost_config, load_max_lifetime_km
from fleetreplacement_env.envs.costs import compute_step_cost

class FleetReplacementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}  # for rendering

    def __init__(self, config: FleetEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()

        # If no config is pased, build a default config by loading from CSV files
        if config is None:
            config = FleetEnvConfig(
                mdp = MDPConfig(max_possible_lifetime_km=load_max_lifetime_km()),
                cost = load_cost_config()     # call loading config
            )
        self.cfg = config                     # holds self.cfg.mdp and self.cfg.cost
        # Assertion to catch case where MDPConfig() is constructed manually and max_possible_lifetime_km is not set
        assert self.cfg.mdp.max_possible_lifetime_km > 0, \
            "MDPConfig.max_possible_lifetime_km must be set — use load_max_lifetime_km()"

        self.current_step = 0
        self.fleet_state: np.ndarray | None = None  # represent unitialized state, for guard in step()
        self.render_mode = render_mode              # for rendering

        # Unpack config calues
        n_vehicles = self.cfg.mdp.n_vehicles
        
        # State space as box, matrix of shape (n_vehicles, 3 (technology, age, mileage))
        # After including current_step in state space: vector with n_vehicles*3+1 elements
        self.observation_space = spaces.Box(
            low  = np.zeros(n_vehicles * 3 + 1, dtype=np.float32),                                                                                
            high = np.ones( n_vehicles * 3 + 1, dtype=np.float32),
            dtype = np.float32
        )

        # Action space
        self.action_space = spaces.MultiDiscrete([3] * n_vehicles)        # 0=keep, 1=replace with ICT, 2=replace with BET

        # Derive ICT purchase ban step from calendar year, limit to [0, planning_horizon]
        raw_ban_step = self.cfg.cost.ict_ban_year - self.cfg.mdp.start_year 
        self.ict_ban_step = max(0, min(raw_ban_step, self.cfg.mdp.planning_horizon))

    # Constructing observations for NN
    def _get_obs(self):
        tech = self.fleet_state[:, 0]
        age = self.fleet_state[:, 1] / self.cfg.mdp.max_vehicle_age                                          # normalize
        # mileage = self.fleet_state[:, 2] / self.cfg.mdp.max_mileage 
        mileage = self.fleet_state[:, 2] / self.cfg.mdp.max_possible_lifetime_km                          # normalize using max_possible_lifetime_km
        flat_fleet = np.stack([tech, age, mileage], axis=1).flatten().astype(np.float32)        
        step_feature = np.array([self.current_step / self.cfg.mdp.planning_horizon], dtype=np.float32)       # normalize
        return np.concatenate([flat_fleet, step_feature])
    
    def _get_info(self):
        return {
            "step": self.current_step,
            "mean_age": float(self.fleet_state[:, 1].mean()),
            "mean_mileage": float(self.fleet_state[:, 2].mean()),
            "n_bet": int((self.fleet_state[:, 0] == 1).sum()),
            "n_ict":  int((self.fleet_state[:, 0] == 0).sum()),
        }
    
    # Reset function, starts new episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize age
        max_init_age = int(self.cfg.cost.max_lifetime_km / self.cfg.cost.akt_base) - 1                            # computes the highest safe age to initialize; -1 ensures vehicle still has one full year of operation left before hitting force-replace threshold
        ages = self.np_random.integers(1, max_init_age + 1, size=self.cfg.mdp.n_vehicles).astype(np.float32)      # generate random vehicle age; +1 allows max_init_age to be included in range (np is exclusive on upper bound); convert to float (as defined in obs space)
        # Initialize mileage
        mileages = ages * self.cfg.cost.akt_base                                                                  # starting mileage, derived from age
        technologies = np.zeros(self.cfg.mdp.n_vehicles, dtype=np.float32)                                        # 0 = diesel, all ICT

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
            act = int(action[i])                        # assign value of replacement to act (0 = keep, 1 = replace with ICT, 2 = replace with BET)

            # # Forced replacement if limits exceeded (regardless of action)
            # # Returns true or false if one is true
            # force_replace = (
            #     age + 1 >= self.cfg.mdp.max_vehicle_age
            #     # or mileage + self.cfg.cost.akt_base >= self.cfg.mdp.max_mileage
            #     or mileage + self.cfg.cost.akt_base >= self.cfg.cost.max_lifetime_km
            # )

            # # If agent says no replacement, but force_replace is true, default to replace with ICT
            # if force_replace and act == 0:
            #     act = 1

            resolved_action[i] = act

            # Calculate cost by calling compute_step_cost in costs.py
            cost_item = compute_step_cost(
                tech=int(tech),
                age=age,
                action=act,
                annual_km=self.cfg.cost.akt_base,
                mileage=mileage,
                cfg=self.cfg.cost,
            )
            total_cost += cost_item.total

            # Update fleet state after cost is computed
            if act == 0:  # keep
                self.fleet_state[i] = [tech, age + 1, mileage + self.cfg.cost.akt_base]
            else:         # replace with ICT (act=1) or BET (act=2)
                new_tech = 0.0 if act == 1 else 1.0
                self.fleet_state[i] = [new_tech, 1.0, self.cfg.cost.akt_base]      # assupmtion: new vehicle does operate in the year of purchase in this model

        self.current_step += 1
        reward = -total_cost    # agent's reward is negative cost
        truncated = self.current_step >= self.cfg.mdp.planning_horizon
        terminated = False  

        # Rendering with declared mode
        if self.render_mode == "human":
            self._render_frame(resolved_action, total_cost)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    # Action masking
    def action_masks(self) -> np.ndarray:
        """
        Returns a flat bool array of shape (n_vehicles * 3,) for MaskablePPO
        [keep, replace with ICT, replace with BET]
        True = action is valid, False = action is masked out
        """
        assert self.fleet_state is not None, "Call reset() before action_masks()."
        n = self.cfg.mdp.n_vehicles
        masks = np.ones((n, 3), dtype=bool)     # n=vehicles, 3 possible actions

        ict_banned = self.current_step >= self.ict_ban_step

        for i in range(n):
            tech, age, mileage = self.fleet_state[i]

            # Rule 1: Block replacing a brand-new vehicle (age == 1)
            if age <= 1.0:
                masks[i, 1] = False  # block replace with ICT
                masks[i, 2] = False  # block replace with BET

            # Rule 2: Force replace if at lifetime limit
            must_replace = (
                age + 1 >= self.cfg.mdp.max_vehicle_age
                or mileage + self.cfg.cost.akt_base >= self.cfg.cost.max_lifetime_km    # primary limit, see config.py
            )
            if must_replace:
                masks[i, 0] = False  # cannot keep, but agent can decide if replace with BET or ICT

            # Rule 3: Block ICT purchase after ban year
            if ict_banned:
                masks[i, 1] = False  # block replace with ICT

            # Safety: ensure at least one action is always valid
            # Edge case: age==1 AND must_replace, theoretically impossible given max_vehicle_age >> 1
            if not masks[i].any():
                masks[i, 2] = True  # BET replacement is the last-resort fallback

        return masks.flatten()
    
    # Rendering method
    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self, action=None, cost=None):
        act_labels = {0: "kept", 1: "→ ICT", 2: "→ BET"}
        print(f"\n── Step {self.current_step} ──────────────────────────────")
        print(f"{'#':<5} {'Tech':<8} {'Age':>5} {'Mileage':>10}  Action")
        for i, (tech, age, km) in enumerate(self.fleet_state):
            act = act_labels.get(int(action[i]), "-") if action is not None else "-"
            print(f"{i:<5} {'ICT' if tech == 0 else 'BET':<8} {int(age):>5} {int(km):>10}  {act}")
        if cost is not None:
            print(f"\nTotal cost: €{cost:>12,.0f}   Reward: €{-cost:>12,.0f}")

    
    def close(self):
        pass