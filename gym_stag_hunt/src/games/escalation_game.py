from random import randint

from numpy import zeros, uint8

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame
from gym_stag_hunt.src.utils import overlaps_entity

"""
Entity Keys
"""
A_AGENT = 0
B_AGENT = 1
MARK = 2


class Escalation(AbstractGridGame):
    def __init__(self,
                 streak_break_punishment_factor,
                 # Super Class Params
                 window_title, grid_size, screen_size, obs_type, load_renderer):
        """
        :param streak_break_punishment_factor: Negative reinforcement for breaking the streak
        """

        super(Escalation, self).__init__(grid_size=grid_size, screen_size=screen_size, obs_type=obs_type)

        self._streak_break_punishment_factor = streak_break_punishment_factor

        self._mark = zeros(2, dtype=uint8)
        self._streak_active = False
        self._streak = 0
        self.reset_entities()

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == 'image' or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.escalation_renderer import EscalationRenderer
            self._renderer = EscalationRenderer(game=self, window_title=window_title, screen_size=screen_size)

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        a_on_mark = overlaps_entity(self.A_AGENT, self.MARK)
        b_on_mark = overlaps_entity(self.B_AGENT, self.MARK)

        punishment = 0 - (self._streak_break_punishment_factor*self._streak)

        if a_on_mark and b_on_mark:
            rewards = 1, 1
        elif a_on_mark:
            rewards = punishment, 0
        elif b_on_mark:
            rewards = 0, punishment
        else:
            rewards = 0, 0

        if 1 in rewards:
            if not self._streak_active:
                self._streak_active = True
            self._streak = self._streak + 1
            self._mark = self._random_move(self.MARK)
        else:
            self._streak = 0
            self._streak_active = False

        return rewards

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: List of actions for the two agents. If nothing is passed for the second agent, it does a
                            a random action.
        :return: observation, rewards, is the game done
        """
        self._move_agents(agent_moves=agent_moves)
        iteration_rewards = self._calc_reward()
        obs = self.get_observation()
        info = {}

        return obs, iteration_rewards, False, info

    def _coord_observation(self):
        """
        :return: list of all the entity coordinates
        """
        return [self.A_AGENT, self.B_AGENT, self.MARK]

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self.MARK = (randint(0, self.GRID_W - 1), randint(0, self.GRID_H - 1))

    """
    Properties
    """

    @property
    def MARK(self):
        return self._mark

    @MARK.setter
    def MARK(self, new_pos):
        self._mark[0], self._mark[1] = new_pos[0], new_pos[1]

    @property
    def STREAK_ACTIVE(self):
        return self._streak_active

    @property
    def STREAK(self):
        return self._streak

    @property
    def ENTITY_POSITIONS(self):
        return {
            'a_agent': self.A_AGENT,
            'b_agent': self.B_AGENT,
            'mark': self.MARK,
            'streak_active': self.STREAK_ACTIVE
        }
