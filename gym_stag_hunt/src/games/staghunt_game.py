from itertools import product
from math import hypot
from random import choice
from numpy import zeros, flipud, full, rot90

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame, overlaps_entity

from gym_stag_hunt.src.games.abstract_grid_game import UP, DOWN, LEFT, RIGHT

"""
Entity Keys
"""
A_AGENT = 0
B_AGENT = 1
STAG    = 2
PLANT   = 3


class StagHunt(AbstractGridGame):
    def __init__(self,
                 episodes_per_game,
                 stag_reward,
                 stag_follows,
                 run_away_after_maul,
                 forage_quantity,
                 forage_reward,
                 mauling_punishment,
                 # Super Class Params
                 window_title, grid_size, screen_size, obs_type, load_renderer):
        """
        :param episodes_per_game: How many timesteps take place before we reset the entity positions.
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        super(StagHunt, self).__init__(grid_size=grid_size, screen_size=screen_size, obs_type=obs_type)

        # Config
        self._stag_follows        = stag_follows
        self._run_away_after_maul = run_away_after_maul

        # Reinforcement Variables
        self._stag_reward        = stag_reward              # record RL values as attributes
        self._forage_quantity    = forage_quantity
        self._forage_reward      = forage_reward
        self._mauling_punishment = mauling_punishment

        # State Variables
        self._tagged_plants = []                            # harvested plants that need to be re-spawned
        self._eps_to_go     = episodes_per_game             # state variable to keep track of how many eps till reset
        self._eps_per_game  = episodes_per_game             # record episodes per game as attribute

        # Entity Positions
        self._stag_pos   = zeros(2, dtype=int)
        self._plants_pos = []
        self.reset_entities()                               # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == 'image' or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.markov_renderer import MarkovRenderer
            self._renderer = MarkovRenderer(game=self, window_title=window_title, screen_size=screen_size)

    """
    Plant Spawning Methods
    """

    def _spawn_plants(self):
        """
        Generate new coordinates for all the plants.
        :return:
        """
        new_plants = []
        for x in range(self._forage_quantity):
            new_plants.append(self._place_entity_in_unoccupied_cell())
        return new_plants

    def _place_entity_in_unoccupied_cell(self, existing_plants=None):
        """
        Places an individual plant on a position on the grid currently unoccupied by anything.
        :param existing_plants: The positions of the other plants
        :return: a tuple corresponding to the chosen position
        """
        if existing_plants is None:
            existing_plants = self._plants_pos

        # get and parse all the entity positions
        a = (self.A_AGENT[0], self.A_AGENT[1])
        b = (self.B_AGENT[0], self.B_AGENT[1])
        stag = (self.STAG[0], self.STAG[1])
        # First we make a list of all the possible x,y coordinates in our grid
        coords = list(product(range(0, self.GRID_W), range(0, self.GRID_H)))

        # Then we remove used coordinates from consideration
        if a in coords:             # the if check is here to prevent issues when removing already removed coordinates,
            coords.remove(a)        # which is what happens if two entities are on an overlapping grid cell
        if b in coords:
            coords.remove(b)
        if stag in coords:
            coords.remove(stag)
        for plant in existing_plants:
            plant = plant[0], plant[1]
            coords.remove(plant)

        chosen_coords = choice(coords)
        new_pos = zeros(2, dtype=int)
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

        if overlaps_entity(self.A_AGENT, self.STAG):
            if overlaps_entity(self.B_AGENT, self.STAG):
                return self._stag_reward, self._stag_reward                 # Successful stag hunt
            else:
                if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                    return self._mauling_punishment, self._forage_reward    # A is mauled, B foraged
                else:
                    return self._mauling_punishment, 0                      # A is mauled, B did not forage

        elif overlaps_entity(self.B_AGENT, self.STAG):
            """
            we already covered the case where a and b are both on the stag,
            so we can skip that check here
            """
            if self._overlaps_plants(self.A_AGENT, self.PLANTS):
                return self._forage_reward, self._mauling_punishment        # A foraged, B is mauled
            else:
                return 0, self._mauling_punishment                          # A did not forage, B is mauled

        elif self._overlaps_plants(self.A_AGENT, self.PLANTS):
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                return self._forage_reward, self._forage_reward             # Both agents foraged
            else:
                return self._forage_reward, 0                               # Only A foraged

        else:
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                return 0, self._forage_reward                               # Only B foraged
            else:
                return 0, 0                                                 # No one got anything

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: List of actions for the two agents. If nothing is passed for the second agent, it does a
                            a random action.
        :return: observation, rewards, is the game done
        """
        self._eps_to_go = self._eps_to_go - 1   # decrement reset counter

        # Move Entities
        self._move_stag()

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

        # Reset prey if it was caught
        if iteration_rewards == (self._stag_reward, self._stag_reward):
            self.STAG = self._place_entity_in_unoccupied_cell()
        elif self._run_away_after_maul and self._mauling_punishment in iteration_rewards:
            self.STAG = self._place_entity_in_unoccupied_cell()
        elif self._forage_reward in iteration_rewards:
            self._respawn_plants()

        game_done = self._eps_to_go == 0

        if game_done:
            self._eps_to_go = self._eps_per_game
            self.reset_entities()

        obs = self.get_observation()

        return obs, iteration_rewards, game_done

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
        a, b, stag, plants = self.A_AGENT, self.B_AGENT, self.STAG, self.PLANTS

        matrix[a[0]][a[1]][A_AGENT]           = True
        matrix[b[0]][b[1]][B_AGENT]           = True
        matrix[stag[0]][stag[1]][STAG]        = True
        for plant in plants:
            matrix[plant[0]][plant[1]][PLANT] = True

        return flipud(rot90(matrix))

    """
    Movement Methods
    """

    def _seek_agent(self, agent_to_seek):
        """
        Moves the stag towards the specified agent
        :param agent_to_seek: agent to pursue
        :return: new position tuple for the stag
        """
        agent = self.A_AGENT
        if agent_to_seek == 'b':
            agent = self.B_AGENT

        stag_x, stag_y = self.STAG[0], self.STAG[1]

        left = stag_x > agent[0]
        right = stag_x < agent[0]
        up = stag_y > agent[1]
        down = stag_y < agent[1]

        options = []
        if left:
            options.append(LEFT)
        if down:
            options.append(DOWN)
        if right:
            options.append(RIGHT)
        if up:
            options.append(UP)

        if not options:
            options = [LEFT, DOWN, RIGHT, UP]

        return self._move_entity(self.STAG, choice(options))

    def _move_stag(self):
        """
        Moves the stag towards the nearest agent.
        :return:
        """
        if self._stag_follows:
            stag_x, stag_y = self.STAG[0], self.STAG[1]
            a_dist = hypot(stag_x - self.A_AGENT[0], stag_y - self.A_AGENT[1])
            b_dist = hypot(stag_x - self.B_AGENT[0], stag_y - self.B_AGENT[1])

            if a_dist < b_dist:
                agent_to_seek = 'a'
            else:
                agent_to_seek = 'b'

            self.STAG = self._seek_agent(agent_to_seek)
        else:
            self.STAG = self._random_move(self.STAG)

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self.STAG = [self.GRID_W // 2, self.GRID_H // 2]
        self.PLANTS = self._spawn_plants()

    """
    Properties
    """

    @property
    def STAG(self):
        return self._stag_pos

    @STAG.setter
    def STAG(self, new_pos):
        self._stag_pos[0], self._stag_pos[1] = new_pos[0], new_pos[1]

    @property
    def PLANTS(self):
        return self._plants_pos

    @PLANTS.setter
    def PLANTS(self, new_pos):
        if len(new_pos) == self._forage_quantity:
            self._plants_pos = new_pos
        else:
            print("Something's fucked with the plants")

    @property
    def ENTITY_POSITIONS(self):
        return {
            'a_agent': self.A_AGENT,
            'b_agent': self.B_AGENT,
            'stag': self.STAG,
            'plants': self.PLANTS
        }

