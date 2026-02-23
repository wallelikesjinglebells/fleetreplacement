from gymnasium.envs.registration import register

register(
    id="FleetReplacement-v0",
    entry_point="fleetreplacement_env.envs.fleet_replacement:FleetReplacementEnv",
)
