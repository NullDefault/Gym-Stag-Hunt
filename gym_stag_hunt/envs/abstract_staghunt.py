from abc import ABC

from gym import Env


class AbstractStagHunt(Env, ABC):
    def __init__(self,
                 stag_reward=5,
                 forage_reward=1,
                 mauling_punishment=-5):
        """
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        if not (stag_reward > forage_reward >= 0 > mauling_punishment):
            raise AttributeError('The game does not qualify as a Stag Hunt, please change parameters so that '
                                 'stag_reward > forage_reward >= 0 > mauling_punishment')

        super(AbstractStagHunt, self).__init__()

        self.stag_reward = stag_reward
        self.forage_reward = forage_reward
        self.mauling_punishment = mauling_punishment
        self.reward_range = (mauling_punishment, stag_reward)

        self.done = False
        self.seed()
