from random import randint

import numpy as np

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame, overlaps_entity

"""
Entity Keys
"""
A_AGENT = 0
B_AGENT = 1
MARK = 3


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

        self._mark = (0, 0)
        self._streak_active = False
        self._streak = 0
        self.reset_entities()

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == 'image' or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.rendering.escalation_renderer import EscalationRenderer
            self._renderer = EscalationRenderer(game=self, window_title=window_title, screen_size=screen_size)

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        a_on_mark = overlaps_entity(self.A_AGENT, self.MARK)
        b_on_mark = overlaps_entity(self.B_AGENT, self.MARK)

        rewards = 0, 0
        punishment = 0 - (self._streak_break_punishment_factor*self._streak)

        if self._streak_active:
            if a_on_mark:
                if b_on_mark:
                    rewards = 1, 1
                else:
                    rewards = punishment, 0
            else:
                if b_on_mark:
                    rewards = 0, punishment
                else:
                    rewards = 0, 0

        if 1 in rewards:
            if not self._streak_active:
                self._streak_active = True
            self._streak = self._streak + 1
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
        if isinstance(agent_moves, list):
            self.A_AGENT = self._move_entity(self.A_AGENT, agent_moves[0])
            if len(agent_moves) > 1:
                self.B_AGENT = self._move_entity(self.B_AGENT, agent_moves[1])
            else:
                self.B_AGENT = self._random_move(self.B_AGENT)
        else:
            self.A_AGENT = self._move_entity(self.A_AGENT, agent_moves)
            self.B_AGENT = self._random_move(self.B_AGENT)

        if self._streak_active:
            self._mark = self._random_move(self.MARK)

        # Get Rewards
        iteration_rewards = self._calc_reward()

        obs = self.get_observation()

        return obs, iteration_rewards, False

    def _coord_observation(self):
        """
        :return: 3d array observation of the grid
        :format: NxN matrix where each index has 4 entries - 0 or 1 depending on if the given entity is in that cell
                 So an individual row (here of length 5) looks like:
                 [..[[A, B,    [A, B,    [A, B,    [A, B,    [A, B,
                     S, P],   S, P],     S, P],    S, P],    S, P]],..]
                 Where A is A Agent, S is stag e.t.c
                 An actually printed matrix row will look like this:
                 [[0 0 0 0] [0 0 1 0] [0 0 0 0] [0 0 0 0] [0 0 0 0]]
                 Which translates to there being a stag in the second column of this row
        """
        matrix = np.full((self._grid_size[0], self._grid_size[1], 4), False, dtype=bool)
        a, b, mark = self.A_AGENT, self.B_AGENT, self.MARK

        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True
        matrix[mark[0]][mark[1]][MARK]        = True

        return np.flipud(np.rot90(matrix))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self._mark = (randint(0, self.GRID_W), randint(0, self.GRID_H))

    """
    Properties
    """

    @property
    def MARK(self):
        return self._mark

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
