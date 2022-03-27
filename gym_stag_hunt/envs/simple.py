from sys import stdout

from gym import Env
from gym.spaces import Discrete, Box
from numpy.random import randint

COOPERATE = 0
DEFECT = 1


class SimpleEnv(Env):
    def __init__(
        self,
        cooperation_reward=5,
        defect_alone_reward=1,
        defect_together_reward=1,
        failed_cooperation_punishment=-5,
        eps_per_game=1,
    ):
        """
        :param cooperation_reward: How much reinforcement the agents get for catching the stag
        :param defect_alone_reward: How much reinforcement an agent gets for defecting if the other one doesn't
        :param defect_together_reward: How much reinforcement an agent gets for defecting if the other one does also
        :param failed_cooperation_punishment: How much reinforcement the agents get for trying to catch a stag alone
        :param eps_per_game: How many games happen before the internal done flag is set to True. Only included for
                             the sake of convenience.
        """

        if not (
            cooperation_reward
            > defect_alone_reward
            >= defect_together_reward
            > failed_cooperation_punishment
        ):
            raise AttributeError(
                "The game does not qualify as a Stag Hunt, please change parameters so that "
                "stag_reward > forage_reward_single >= forage_reward_both > mauling_punishment"
            )

        super(SimpleEnv, self).__init__()

        # Reinforcement Variables
        self.cooperation_reward = cooperation_reward
        self.defect_alone_reward = defect_alone_reward
        self.defect_together_reward = defect_together_reward
        self.failed_cooperation_punishment = failed_cooperation_punishment

        # State Variables
        self.done = False
        self.ep = 0
        self.final_ep = eps_per_game
        self.seed()

        # Environment Config
        self.action_space = Discrete(2)  # cooperate or defect
        self.observation_space = Box(
            low=0, high=1, shape=(2,), dtype=int
        )  # last agent actions
        self.reward_range = (failed_cooperation_punishment, cooperation_reward)

    def step(self, actions):
        """
        Play one stag hunt game.
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agent takes the specified action.
        :return: observation, rewards, is the game done, additional info
        """
        self.ep = self.ep + 1
        if self.ep >= self.final_ep:
            done = True
            self.ep = 0
        else:
            done = False

        if isinstance(actions, list):
            a_action = actions[0]
            if len(actions) > 1:
                b_action = actions[1]
            else:
                b_action = randint(0, 1)
        else:
            a_action = actions
            b_action = randint(0, 1)

        b_cooperated = b_action == COOPERATE

        if a_action == COOPERATE:
            reward = (
                (self.cooperation_reward, self.cooperation_reward)
                if b_cooperated
                else (self.failed_cooperation_punishment, self.defect_alone_reward)
            )
        else:
            reward = (
                (self.defect_alone_reward, self.failed_cooperation_punishment)
                if b_cooperated
                else (self.defect_together_reward, self.defect_together_reward)
            )

        obs = (a_action, b_action)

        return obs, reward, done, {}

    def reset(self):
        """
        Reset the game state
        """
        self.done = False
        self.ep = 0

    def render(self, mode="human", rewards=None):
        """
        :return:
        """
        if rewards is None:
            print("Please supply rewards to render.")
            pass
        else:
            top_right = "  "
            top_left = "  "
            bot_left = "  "
            bot_right = "  "

            if rewards == (self.cooperation_reward, self.cooperation_reward):
                top_left = "AB"
            elif rewards == (
                self.defect_alone_reward,
                self.failed_cooperation_punishment,
            ):
                bot_left = "A "
                top_right = " B"
            elif rewards == (
                self.failed_cooperation_punishment,
                self.defect_alone_reward,
            ):
                top_left = "A "
                top_right = " B"
            elif rewards == (self.defect_together_reward, self.defect_together_reward):
                bot_right = "AB"

            stdout.write("\n\n\n")
            stdout.write("      B   \n")
            stdout.write("    C   D \n")
            stdout.write("   ╔══╦══╗\n")
            stdout.write(" C ║" + top_left + "║" + top_right + "║\n")
            stdout.write("   ║  ║  ║\n")
            stdout.write("A  ╠══╬══╣\n")
            stdout.write("   ║  ║  ║\n")
            stdout.write(" D ║" + bot_left + "║" + bot_right + "║\n")
            stdout.write("   ╚══╩══╝\n\r")
            stdout.flush()

    def close(self):
        quit()
