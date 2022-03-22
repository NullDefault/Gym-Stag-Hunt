from time import sleep

from gym_stag_hunt.demos import print_ep
from gym_stag_hunt.envs.pettingzoo import hunt, harvest, escalation

ENVS = {
    'HUNT': hunt.parallel_env,
    'HARVEST': harvest.parallel_env,
    'ESCALATION': escalation.parallel_env
}

ENV = 'HUNT'

if __name__ == "__main__":
    env = ENVS[ENV](obs_type='image', enable_multiagent=True)
    obs = env.reset()
    for i in range(100):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, done, info = env.step(actions)
        print_ep(obs, rewards, done, info)
        sleep(.4)
        env.render(mode='human')
    env.close()
    quit()
