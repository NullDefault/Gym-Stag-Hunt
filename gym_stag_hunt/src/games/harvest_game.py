from random import uniform

from numpy import flipud, rot90, full

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame
from gym_stag_hunt.src.utils import overlaps_entity, place_entity_in_unoccupied_cell

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
        self._plants = []
        self._maturity_flags = [False] * max_plants
        self.reset_entities()                               # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == 'image' or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.harvest_renderer import HarvestRenderer
            self._renderer = HarvestRenderer(game=self, window_title=window_title, screen_size=screen_size)

    """
    Plant Spawning Methods
    """
    def _spawn_plants(self):
        """
        Generate new coordinates for all the plants.
        :return:
        """
        new_plants = []
        for x in range(self._max_plants):
            new_plants.append(place_entity_in_unoccupied_cell(grid_dims=self.GRID_DIMENSIONS,
                                                              used_coordinates=new_plants+self.AGENTS))
        return new_plants

    def _respawn_plants(self):
        """
        Checks which plants are due for re-spawning and respawns them.
        :return:
        """
        plants = self.PLANTS
        for eaten_plant in self._tagged_plants:
            plants[eaten_plant] = place_entity_in_unoccupied_cell(grid_dims=self.GRID_DIMENSIONS,
                                                                  used_coordinates=plants+self.AGENTS)
        self._tagged_plants = []
        self._plants = plants

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

        return a_reward, b_reward

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
            self._respawn_plants()

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
        matrix = full((self._grid_size[0], self._grid_size[1], 4), False, dtype=bool)
        a, b, plants = self.A_AGENT, self.B_AGENT, self.PLANTS

        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True

        maturity_flags = self.MATURITY_FLAGS
        for idx, plant in enumerate(plants):
            plant_age = M_PLANT if maturity_flags[idx] is True else Y_PLANT
            matrix[plant[0]][plant[1]][plant_age] = True

        return flipud(rot90(matrix))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self._plants = self._spawn_plants()
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
            'a_agent': self.A_AGENT,
            'b_agent': self.B_AGENT,
            'plant_coords': self.PLANTS,
            'maturity_flags': self.MATURITY_FLAGS
        }
