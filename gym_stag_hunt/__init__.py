from gym.envs.registration import register

register(
    id='StagHunt-Markov-v0',
    entry_point='gym_stag_hunt.envs:MarkovStagHunt'
)

register(
    id='StagHunt-Classic-v0',
    entry_point='gym_stag_hunt.envs:ClassicStagHunt'
)

register(
    id='StagHunt-Harvest-v0',
    entry_point='gym_stag_hunt.envs:HarvestStagHunt'
)
