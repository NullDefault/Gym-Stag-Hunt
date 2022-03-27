from gym.envs.registration import register

register(id="StagHunt-Hunt-v0", entry_point="gym_stag_hunt.envs:HuntEnv")

register(id="StagHunt-Simple-v0", entry_point="gym_stag_hunt.envs:SimpleEnv")

register(id="StagHunt-Harvest-v0", entry_point="gym_stag_hunt.envs:HarvestEnv")

register(id="StagHunt-Escalation-v0", entry_point="gym_stag_hunt.envs:EscalationEnv")

register(id="StagHunt-Hunt-PZ-v0", entry_point="gym_stag_hunt.envs:HuntPZEnv")

register(id="StagHunt-Harvest-PZ-v0", entry_point="gym_stag_hunt.envs:HarvestPZEnv")

register(
    id="StagHunt-Escalation-PZ-v0", entry_point="gym_stag_hunt.envs:EscalationPZEnv"
)
