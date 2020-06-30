from gym.envs.registration import register

register(
    id='trussbot-v0',
    entry_point='gym_trussbot.envs:TrussbotEnv',
)
