from itertools import product
from pettingzoo.test import parallel_api_test

from gym_stag_hunt.envs import (
    ZooHuntEnvironment,
    ZooHarvestEnvironment,
    ZooEscalationEnvironment,
)

str_help = ["Hunt", "Harvest", "Escalation"]
envs = [ZooHuntEnvironment, ZooHarvestEnvironment, ZooEscalationEnvironment]
obs_types = ["coords", "image"]

if __name__ == "__main__":
    envs_to_test = product(envs, obs_types)
    for idx, environment in enumerate(envs):
        for obs_type in obs_types:
            print("Environment: " + str_help[idx])
            print("Observation Type: " + str(obs_type))
            env = environment(obs_type=obs_type, enable_multiagent=True)

            parallel_api_test(env, num_cycles=100)
