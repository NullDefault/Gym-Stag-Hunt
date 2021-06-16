from time import sleep

from gym_stag_hunt.envs.simple import SimpleEnv
from gym_stag_hunt.envs.hunt import HuntEnv
from gym_stag_hunt.envs.escalation import EscalationEnv
from gym_stag_hunt.envs.harvest import HarvestEnv

ENVS = {
    'CLASSIC': SimpleEnv,
    'HUNT': HuntEnv,
    'HARVEST': HarvestEnv,
    'ESCALATION': EscalationEnv
}
ENV = 'HUNT'

if __name__ == "__main__":
    env = ENVS[ENV](obs_type='coords', load_renderer=True)
    env.reset()
    for i in range(10000):
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
        print(obs)
        print(env.observation_space.sample())

        sleep(.2)
        if ENV == 'CLASSIC':
            env.render(rewards=rewards)
        else:
            env.render()
    env.close()
    quit()
