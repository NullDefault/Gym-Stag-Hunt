from time import sleep

from gym_stag_hunt.demos import print_ep
from gym_stag_hunt.envs import (
    ZooHuntEnvironment,
    ZooHarvestEnvironment,
    ZooEscalationEnvironment,
)

ENVS = {
    "HUNT": ZooHuntEnvironment,
    "HARVEST": ZooHarvestEnvironment,
    "ESCALATION": ZooEscalationEnvironment,
}

ENV = "HUNT"

if __name__ == "__main__":
    env = ENVS[ENV](obs_type="image", enable_multiagent=True)
    obs = env.reset()
    for i in range(100):
        actions = {agent: env._action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, done, info = env.step(actions)
        env.render()
        sleep(0.4)
    env.close()
    quit()
