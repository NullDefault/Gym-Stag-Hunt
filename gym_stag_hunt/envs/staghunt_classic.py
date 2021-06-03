import time
from random import randint
from sys import stdout

from gym.spaces import Discrete

from gym_stag_hunt.envs.abstract_stag_hunt import AbstractStagHunt

COOPERATE = 0
DEFECT = 1


class ClassicStagHunt(AbstractStagHunt):
    def __init__(self):
        super(ClassicStagHunt, self).__init__()
        self.action_space = Discrete(2)             # cooperate or defect
        self.observation_space = Discrete(2)        # last agent actions

    def step(self, actions):
        """
        Play one stag hunt game.
        :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
                        random move, or two, in which case each agents takes the specified action.
        :return: observation, rewards, is the game done, additional info
                 Note: this environment formally doesn't have any observations, so the passed agent actions are returned
                       instead.
        """
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
            reward = (self.stag_reward, self.stag_reward) if b_cooperated \
                else (self.mauling_punishment, self.forage_reward)
        else:
            reward = (self.forage_reward, self.mauling_punishment) if b_cooperated \
                else (self.forage_reward, self.forage_reward)

        obs = (a_action, b_action)

        return obs, reward, done, {}

    def reset(self):
        """
        Reset the game state
        """
        self.done = False

    def render(self, mode="human", rewards=None):
        """
        :return:
        """
        if rewards is None:
            print("Please supply rewards to render.")
            pass
        else:
            top_right = '  '
            top_left  = '  '
            bot_left  = '  '
            bot_right = '  '

            if rewards == (self.stag_reward, self.stag_reward):
                top_left = 'AB'
            elif rewards == (self.forage_reward, self.mauling_punishment):
                bot_left = 'A '
                top_right = ' B'
            elif rewards == (self.mauling_punishment, self.forage_reward):
                top_left = 'A '
                top_right = ' B'
            elif rewards == (self.forage_reward, self.forage_reward):
                bot_right = 'AB'

            stdout.write('\n\n\n')
            stdout.write('      B   \n')
            stdout.write('    C   D \n')
            stdout.write('   ╔══╦══╗\n')
            stdout.write(' C ║' + top_left + '║' + top_right + '║\n')
            stdout.write('   ║  ║  ║\n')
            stdout.write('A  ╠══╬══╣\n')
            stdout.write('   ║  ║  ║\n')
            stdout.write(' D ║' + bot_left + '║' + bot_right + '║\n')
            stdout.write('   ╚══╩══╝\n\r')
            stdout.flush()

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        quit()


if __name__ == "__main__":
    env = ClassicStagHunt()
    env.reset()
    for i in range(1000):
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
        time.sleep(.4)
        env.render(rewards=rewards)
    env.close()
    quit()
