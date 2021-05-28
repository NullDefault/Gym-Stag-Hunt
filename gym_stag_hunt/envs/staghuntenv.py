from random import choice

import gym
import numpy as np
from gym.spaces import Discrete, Box

from gym_stag_hunt.engine.game import Game

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


class StagHuntEnv(gym.Env):
    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self, grid_size=(5, 5), stag_reward=5, screen_size=(600, 600), forage_reward=1, enable_render=True):
        super(StagHuntEnv, self).__init__()

        self.enable_render = enable_render

        self.game = Game(game_name="OpenAI Gym - Stag Hunt (%d x %d)" % grid_size,
                         grid_size=grid_size,
                         screen_size=screen_size,
                         enable_render=enable_render)

        self.stag_reward = stag_reward  # if the stag is caught, both agents receive this much
        self.forage_reward = forage_reward  # an agent receives this much for foraging

        # Up, Down, Left, Right
        self.action_space = Discrete(4)
        # Observation is the rgb pixel array
        self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=np.int64)

        # initial state
        self.done = False

        # simulation related variables
        self.seed()

    def step(self, actions):
        obs, reward, done = self.game.update(actions)
        # Generate Info (If Appropriate)
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.game.reset_entities()
        self.done = False
        return self.game.RENDERER.update()

    def render(self, mode="human", close=False):
        if close:
            self.game.RENDERER.quit()

        return self.game.RENDERER.update(mode)

    def close(self):
        self.game.RENDERER.quit()


if __name__ == "__main__":
    env = StagHuntEnv()
    env.reset()
    for i in range(1000):
        obs, rewards, done, info = env.step([choice(ACTIONS), choice(ACTIONS)])
    env.close()
    quit()
