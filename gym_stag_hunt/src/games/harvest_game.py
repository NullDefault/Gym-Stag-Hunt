from numpy import array
from numpy.random import uniform

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame
from gym_stag_hunt.src.utils import overlaps_entity, spawn_plants, respawn_plants

# Entity Keys
A_AGENT = 0
B_AGENT = 1
Y_PLANT = 2
M_PLANT = 3


class Harvest(AbstractGridGame):
    def __init__(
        self,
        max_plants,
        chance_to_mature,
        chance_to_die,
        young_reward,
        mature_reward,
        # Super Class Params
        window_title,
        grid_size,
        screen_size,
        obs_type,
        load_renderer,
        enable_multiagent,
    ):
        """
        :param max_plants: What is the maximum number of plants that can be on the board.
        :param chance_to_mature: What chance does a young plant have to mature each time step.
        :param chance_to_die: What chance does a mature plant have to die each time step.
        :param young_reward: Reward for harvesting a young plant (awarded to the harvester)
        :param mature_reward: Reward for harvesting a mature plant (awarded to both agents)
        """

        super(Harvest, self).__init__(
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
        )

        # Game Config
        self._max_plants = max_plants
        self._chance_to_mature = chance_to_mature
        self._chance_to_die = chance_to_die
        self._tagged_plants = []  # harvested plants that need to be re-spawned

        # Reinforcement variables
        self._young_reward = young_reward
        self._mature_reward = mature_reward

        # Entity Positions
        self._plants = []
        self._maturity_flags = [False] * max_plants
        self.reset_entities()  # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == "image" or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.harvest_renderer import HarvestRenderer

            self._renderer = HarvestRenderer(
                game=self, window_title=window_title, screen_size=screen_size
            )

    """
    Collision Logic
    """

    def _overlaps_plants(self, a, plants):
        """
        :param a: (X, Y) tuple for entity 1
        :param plants: Array of (X, Y) tuples corresponding to plant positions
        :return: True if a overlaps any of the plants, False otherwise
        """
        for x in range(0, len(plants)):
            pos = plants[x]
            if overlaps_entity(a, pos):
                is_mature = self._maturity_flags[x]
                if x not in self._tagged_plants:
                    self._tagged_plants.append(x)
                return True, is_mature
        return False, False

    """
    State Updating Methods
    """

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        a_collision, a_with_mature = self._overlaps_plants(self.A_AGENT, self.PLANTS)
        b_collision, b_with_mature = self._overlaps_plants(self.B_AGENT, self.PLANTS)

        a_reward, b_reward = 0, 0

        if a_collision:
            if a_with_mature:
                a_reward += self._mature_reward
                b_reward += self._mature_reward
            else:
                a_reward += self._young_reward

        if b_collision:
            if b_with_mature:
                a_reward += self._mature_reward
                b_reward += self._mature_reward
            else:
                b_reward += self._young_reward

        return float(a_reward), float(b_reward)

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
            self._move_agents(
                agent_moves=[agent_moves, self._random_move(self.B_AGENT)]
            )

        for idx, plant in enumerate(self._plants):
            is_mature = self._maturity_flags[idx]
            if is_mature:
                if uniform(0, 1) <= self._chance_to_die:
                    self._maturity_flags[idx] = False
                    self._tagged_plants.append(idx)
            else:
                if uniform(0, 1) <= self._chance_to_mature:
                    self._maturity_flags[idx] = True

        # Get Rewards
        iteration_rewards = self._calc_reward()

        if len(self._tagged_plants) > 0:
            self._plants = respawn_plants(
                plants=self.PLANTS,
                tagged_plants=self._tagged_plants,
                grid_dims=self.GRID_DIMENSIONS,
                used_coordinates=self.AGENTS,
            )
            self._tagged_plants = []

        obs = self.get_observation()
        done = False
        info = {}

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
        :return: tuple of all the entity coordinates
        """
        a, b = self.AGENTS
        shipback = [a[0], a[1], b[0], b[1]]
        maturity_flags = self.MATURITY_FLAGS
        for idx, element in enumerate(self.PLANTS):
            new_entry = [0, 0, 0]
            new_entry[0], new_entry[1], new_entry[2] = (
                element[0],
                element[1],
                int(maturity_flags[idx]),
            )
            shipback = shipback + new_entry

        return array(shipback).flatten()

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self._plants = spawn_plants(
            grid_dims=self.GRID_DIMENSIONS,
            how_many=self._max_plants,
            used_coordinates=self.AGENTS,
        )
        self._maturity_flags = [False] * self._max_plants

    """
    Properties
    """

    @property
    def PLANTS(self):
        return self._plants

    @property
    def MATURITY_FLAGS(self):
        return self._maturity_flags

    @property
    def ENTITY_POSITIONS(self):
        return {
            "a_agent": self.A_AGENT,
            "b_agent": self.B_AGENT,
            "plant_coords": self.PLANTS,
            "maturity_flags": self.MATURITY_FLAGS,
        }
