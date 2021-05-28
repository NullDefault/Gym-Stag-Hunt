from gym.envs.registration import register

register(
    id='StagHunt-v0',
    entry_point='gym_stag_hunt.envs:StagHuntEnv'
)
