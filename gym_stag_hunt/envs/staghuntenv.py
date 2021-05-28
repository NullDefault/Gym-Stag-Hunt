import gym
import numpy as np
from gym.spaces import Discrete, Box

from gym_stag_hunt.engine.game import Game


class StagHuntEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"]
    }

    def __init__(self,
                 grid_size=(5, 5),
                 screen_size=(600, 600),
                 obs_type='image',
                 episodes_per_game=1000,
                 stag_reward=5,
                 forage_quantity=2,
                 forage_reward=1,
                 mauling_punishment=-5
                 ):
        super(StagHuntEnv, self).__init__()

        self.obs_type = obs_type

        self.game = Game(game_name="OpenAI Gym - Stag Hunt (%d x %d)" % grid_size,
                         grid_size=grid_size,
                         screen_size=screen_size,
                         obs_type=obs_type,
                         episodes_per_game=episodes_per_game,
                         stag_reward=stag_reward,
                         forage_quantity=forage_quantity,
                         forage_reward=forage_reward,
                         mauling_punishment=mauling_punishment
                         )

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
        return self.game.get_observation()

    def render(self, mode="human", close=False):
        if close:
            self.game.RENDERER.quit()
        if mode == "human":
            if self.obs_type == 'image':
                self.game.RENDERER.render()
            else:
                print(np.array(self.game.get_observation()))

    def close(self):
        if self.game.RENDERER:
            self.game.RENDERER.quit()


if __name__ == "__main__":
    env = StagHuntEnv(obs_type='image')
    env.reset()
    for i in range(1000):
        env.render()
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
    env.close()
    quit()
