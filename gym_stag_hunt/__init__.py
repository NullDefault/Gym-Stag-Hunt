from gym.envs.registration import register

register(
    id='StagHunt-Hunt-v0',
    entry_point='gym_stag_hunt.envs:HuntEnv'
)

register(
    id='StagHunt-Simple-v0',
    entry_point='gym_stag_hunt.envs:SimpleEnv'
)

register(
    id='StagHunt-Harvest-v0',
    entry_point='gym_stag_hunt.envs:HarvestEnv'
)

register(
    id='StagHunt-Escalation-v0',
    entry_point='gym_stag_hunt.envs:EscalationEnv'
)
