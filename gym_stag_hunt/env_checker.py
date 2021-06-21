from stable_baselines3.common import env_checker
from itertools import product

from gym_stag_hunt.envs import HuntEnv, HarvestEnv, EscalationEnv

envs = [HuntEnv, HarvestEnv, EscalationEnv]
obs_types = ["image", "coords"]

if __name__ == "__main__":
    envs_to_test = product(envs, obs_types)
    for test_item in envs_to_test:
        environment, observation_type = test_item[0], test_item[1]
        env = environment(obs_type=observation_type)
        env_checker.check_env(env, warn=True)
        env.close()
