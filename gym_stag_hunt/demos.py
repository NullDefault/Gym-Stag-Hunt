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
ENV = 'ESCALATION'

if __name__ == "__main__":
    env = ENVS[ENV](obs_type='image', load_renderer=True)
    env.reset()
    for i in range(10000):
        actions = [env.action_space.sample(), env.action_space.sample()]

        if ENV != 'CLASSIC':
            a, b = env.game.AGENTS
            while (env.game._move_entity(a, actions[0]) == a).all():
                actions[0] = env.action_space.sample()
            while (env.game._move_entity(b, actions[1]) == b).all():
                actions[1] = env.action_space.sample()
        obs, rewards, done, info = env.step(actions=actions)

        sleep(.6)
        if ENV == 'CLASSIC':
            env.render(rewards=rewards)
        else:
            env.render()
    env.close()
    quit()
