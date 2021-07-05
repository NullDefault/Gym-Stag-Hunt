from time import sleep

from gym_stag_hunt.envs.simple import SimpleEnv
from gym_stag_hunt.envs.hunt import HuntEnv
from gym_stag_hunt.envs.escalation import EscalationEnv
from gym_stag_hunt.envs.harvest import HarvestEnv
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT

ENVS = {
    'CLASSIC': SimpleEnv,
    'HUNT': HuntEnv,
    'HARVEST': HarvestEnv,
    'ESCALATION': EscalationEnv
}
ENV = 'ESCALATION'


def print_ep(obs, reward, done, info):
    print({
        'observation': obs,
        'reward': reward,
        'simulation over': done,
        'info': info
    })


def dir_parse(key):
    d = {
        LEFT: "LEFT",
        UP: "UP",
        DOWN: "DOWN",
        RIGHT: "RIGHT"
    }
    return d[key]


def manual_input():
    i = input()
    if i in ['w', 'W']:
        i = UP
    elif i in ['a', 'A']:
        i = LEFT
    elif i in ['s', 'S']:
        i = DOWN
    elif i in ['d', 'D']:
        i = RIGHT

    return i


if __name__ == "__main__":
    env = ENVS[ENV](obs_type='image', opponent_policy='pursuit')
    obs = env.reset()
    for i in range(10000):
        actions = env.game._seek_entity(env.game.A_AGENT, env.game.MARK)

        obs, rewards, done, info = env.step(actions=actions)
        # print_ep(obs, rewards, done, info)
        sleep(.6)
        if ENV == 'CLASSIC':
            env.render(rewards=rewards)
        else:
            env.render(mode='human')
    env.close()
    quit()
