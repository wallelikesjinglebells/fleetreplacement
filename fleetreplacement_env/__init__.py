from gymnasium.envs.registration import register

register(
    id="fleetreplacement_env/GridWorld-v0",
    entry_point="fleetreplacement_env.envs:GridWorldEnv",
)
