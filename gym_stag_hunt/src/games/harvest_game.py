from itertools import product
from random import choice, uniform

import numpy as np

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame

"""
Entity Keys
"""
A_AGENT = 0
B_AGENT = 1
Y_PLANT = 2
M_PLANT = 3


class Harvest(AbstractGridGame):
    def __init__(self,
                 max_plants,
                 chance_to_mature,
                 chance_to_die,
                 young_reward,
                 mature_reward,
                 # Super Class Params
                 window_title, grid_size, screen_size, obs_type, load_renderer):
        """
        :param max_plants: What is the maximum number of plants that can be on the board.
        :param chance_to_mature: What chance does a young plant have to mature each time step.
        :param chance_to_die: What chance does a mature plant have to die each time step.
        :param young_reward: Reward for harvesting a young plant (awarded to the harvester)
        :param mature_reward: Reward for harvesting a mature plant (awarded to both agents)
        """

        super(Harvest, self).__init__(grid_size=grid_size, screen_size=screen_size, obs_type=obs_type)

        # Game Config
        self._max_plants = max_plants
        self._chance_to_mature = chance_to_mature
        self._chance_to_die = chance_to_die
        self._tagged_plants = []                            # harvested plants that need to be re-spawned

        # Reinforcement variables
        self._young_reward = young_reward
        self._mature_reward = mature_reward

        # Entity Positions
        # plants = ???
        self.reset_entities()                               # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == 'image' or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.rendering.harvest_renderer import HarvestRenderer
            self._renderer = HarvestRenderer(game=self, window_title=window_title, screen_size=screen_size)

    """
    Plant Spawning Methods
    """

    def _place_entity_in_unoccupied_cell(self, existing_plants=None):
        """
        Places an individual plant on a position on the grid currently unoccupied by anything.
        :param existing_plants: The positions of the other plants
        :return: a tuple corresponding to the chosen position
        """
        if existing_plants is None:
            existing_plants = self._young_plants_pos + self._mature_plants_pos

        # get and parse all the entity positions
        a = (self.A_AGENT[0], self.A_AGENT[1])
        b = (self.B_AGENT[0], self.B_AGENT[1])

        # First we make a list of all the possible x,y coordinates in our grid
        coords = list(product(range(0, self.GRID_W), range(0, self.GRID_H)))

        # Then we remove used coordinates from consideration
        if a in coords:             # the if check is here to prevent issues when removing already removed coordinates,
            coords.remove(a)        # which is what happens if two entities are on an overlapping grid cell
        if b in coords:
            coords.remove(b)
        for plant in existing_plants:
            plant = (plant[0], plant[1])
            if plant in coords:
                coords.remove(plant)

        chosen_coords = choice(coords)
        new_pos = np.zeros(2, dtype=int)
        new_pos[0], new_pos[1] = chosen_coords[0], chosen_coords[1]

        return new_pos

    def _respawn_plants(self):
        """
        Checks which plants are due for re-spawning and respawns them.
        :return:
        """
        plants = self.PLANTS
        for eaten_plant in self._tagged_plants:
            plants[eaten_plant] = self._place_entity_in_unoccupied_cell(existing_plants=plants)
        self._tagged_plants = []
        self.PLANTS = plants

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
            if a[0] == pos[0] and a[1] == pos[1]:
                self._tagged_plants.append(x)
                return True
        return False

    """
    State Updating Methods
    """

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        return 0, 0

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
        a, b, young_plants, mature_plants = self.A_AGENT, self.B_AGENT, self.YOUNG_PLANTS, self.MATURE_PLANTS

        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True
        for plant in young_plants:
            matrix[plant[0]][plant[1]][Y_PLANT] = True
        for plant in mature_plants:
            matrix[plant[0]][plant[1]][M_PLANT] = True

        return np.flipud(np.rot90(matrix))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self._plants = self._spawn_plants()

    """
    Properties
    """

    @property
    def ENTITY_POSITIONS(self):
        return {
            'a_agent': self.A_AGENT,
            'b_agent': self.B_AGENT,
        }

