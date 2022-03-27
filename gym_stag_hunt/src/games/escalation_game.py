from numpy import zeros, uint8, array
from numpy.random import randint

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame
from gym_stag_hunt.src.utils import overlaps_entity

"""
Entity Keys
"""
A_AGENT = 0
B_AGENT = 1
MARK = 2


class Escalation(AbstractGridGame):
    def __init__(
        self,
        streak_break_punishment_factor,
        opponent_policy,
        # Super Class Params
        window_title,
        grid_size,
        screen_size,
        obs_type,
        load_renderer,
        enable_multiagent,
    ):
        """
        :param streak_break_punishment_factor: Negative reinforcement for breaking the streak
        """

        super(Escalation, self).__init__(
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
        )

        self._streak_break_punishment_factor = streak_break_punishment_factor
        self._opponent_policy = opponent_policy
        self._mark = zeros(2, dtype=uint8)
        self._streak_active = False
        self._streak = 0
        self.reset_entities()

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == "image" or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.escalation_renderer import (
                EscalationRenderer,
            )

            self._renderer = EscalationRenderer(
                game=self, window_title=window_title, screen_size=screen_size
            )

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        a_on_mark = overlaps_entity(self.A_AGENT, self.MARK)
        b_on_mark = overlaps_entity(self.B_AGENT, self.MARK)

        punishment = 0 - (self._streak_break_punishment_factor * self._streak)
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
            self.MARK = self._move_entity(self.MARK, self._random_move(self.MARK))
        else:
            self._streak = 0
            self._streak_active = False

        return float(rewards[0]), float(rewards[1])

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: List of actions for the two agents. If nothing is passed for the second agent, it does a
                            a random action.
        :return: observation, rewards, is the game done
        """
        if self._enable_multiagent:
            self._move_agents(agent_moves=agent_moves)
        else:
            if self._opponent_policy == "random":
                self._move_agents(
                    agent_moves=[agent_moves, self._random_move(self.B_AGENT)]
                )
            elif self._opponent_policy == "pursuit":
                self._move_agents(
                    agent_moves=[
                        agent_moves,
                        self._seek_entity(self.B_AGENT, self.MARK),
                    ]
                )

        iteration_rewards = self._calc_reward()
        obs = self.get_observation()
        info = {}
        done = False

        if self._enable_multiagent:
            if self._obs_type == "coords":
                return (
                    (obs, self._flip_coord_observation_perspective(obs)),
                    iteration_rewards,
                    done,
                    info,
                )
            else:
                return (obs, obs), iteration_rewards, done, info
        else:
            return obs, iteration_rewards[0], done, info

    def _coord_observation(self):
        """
        :return: list of all the entity coordinates
        """
        return array([self.A_AGENT, self.B_AGENT, self.MARK]).flatten()

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self.MARK = [randint(0, self.GRID_W - 1), randint(0, self.GRID_H - 1)]

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
            "a_agent": self.A_AGENT,
            "b_agent": self.B_AGENT,
            "mark": self.MARK,
            "streak_active": self.STREAK_ACTIVE,
        }
