from itertools import product
from math import hypot
from random import randint, choice

import numpy as np

from gym_stag_hunt.engine.renderer import Renderer


class Game:
    def __init__(self,
                 game_name="OpenAI - Stag Hunt",
                 grid_size=(5, 5),
                 enable_render=True,
                 screen_size=(600, 600),
                 episodes_per_game=1000,
                 stag_reward=5,
                 forage_quantity=2,
                 forage_reward=1,
                 mauling_punishment=-5,
                 ):
        # Rendering
        self._renderer = None
        self._enable_render = enable_render

        # Reinforcement Variables
        self._stag_reward = stag_reward
        self._forage_quantity = forage_quantity
        self._forage_reward = forage_reward
        self._mauling_punishment = mauling_punishment

        # State Variables
        self._tagged_plants = []
        self._grid_size = grid_size
        self._eps_to_go = episodes_per_game
        self._eps_per_game = episodes_per_game

        # Entity Positions
        self._a_pos = np.zeros(2, dtype=int)
        self._b_pos = np.zeros(2, dtype=int)
        self._stag_pos = np.zeros(2, dtype=int)
        self._plants_pos = []
        self.reset_entities()

        # If rendering is enabled, we will instantiate the rendering pipeline
        if self._enable_render is True:
            self._renderer = Renderer(game_state=self,
                                      game_name=game_name,
                                      screen_size=screen_size)

    """
    Plant Spawning Methods
    """

    def _spawn_plants(self):
        new_plants = []
        for x in range(self._forage_quantity):
            new_plants.append(self._place_plant())
        return new_plants

    def _place_plant(self, existing_plants=None):
        if existing_plants is None:
            existing_plants = self._plants_pos
        a = (self.A_AGENT[0], self.A_AGENT[1])
        b = (self.B_AGENT[0], self.B_AGENT[1])
        stag = (self.STAG[0], self.STAG[1])
        coords = list(product(range(0, self.GRID_W), range(0, self.GRID_H)))
        if a in coords:
            coords.remove(a)
        if b in coords:
            coords.remove(b)
        if stag in coords:
            coords.remove(stag)
        for plant in existing_plants:
            plant = (plant[0], plant[1])
            if plant in coords:
                coords.remove(plant)
        chosen_coords = choice(coords)
        new_pos = np.zeros(2, dtype=int)
        new_pos[0], new_pos[1] = chosen_coords[0], chosen_coords[1]

        return new_pos

    def _respawn_plants(self):
        plants = self.PLANTS
        for eaten_plant in self._tagged_plants:
            plants[eaten_plant] = self._place_plant(existing_plants=plants)
        self._tagged_plants = []
        self.PLANTS = plants

    """
    Collision Logic
    """

    def _overlaps_stag(self, a, b):
        if a[0] == b[0] and a[1] == b[1]:
            return True
        else:
            return False

    def _overlaps_plant(self, a, b):
        for x in range(0, len(b)):
            pos = b[x]
            if a[0] == pos[0] and a[1] == pos[1]:
                self._tagged_plants.append(x)
                return True
        return False

    """
    State Updating Methods
    """

    def _calc_reward(self):
        if self._overlaps_stag(self.A_AGENT, self.STAG):
            if self._overlaps_stag(self.B_AGENT, self.STAG):
                return self._stag_reward, self._stag_reward  # Successful stag hunt
            else:
                if self._overlaps_plant(self.B_AGENT, self.PLANTS):
                    return self._mauling_punishment, self._forage_reward  # A is mauled, B foraged
                else:
                    return self._mauling_punishment, 0  # A is mauled, B did not forage

        elif self._overlaps_stag(self.B_AGENT, self.STAG):
            """
            we already covered the case where a and b are both on the stag,
            so we can skip that check here
            """
            if self._overlaps_plant(self.A_AGENT, self.PLANTS):
                return self._forage_reward, self._mauling_punishment  # A foraged, B is mauled
            else:
                return 0, self._mauling_punishment  # A did not forage, B is mauled

        elif self._overlaps_plant(self.A_AGENT, self.PLANTS):
            if self._overlaps_plant(self.B_AGENT, self.PLANTS):
                return self._forage_reward, self._forage_reward  # Both agents foraged
            else:
                return self._forage_reward, 0  # Only A foraged

        else:
            if self._overlaps_plant(self.B_AGENT, self.PLANTS):
                return 0, self._forage_reward  # Only B foraged
            else:
                return 0, 0  # No one got anything

    def update(self, agent_moves):
        self._eps_to_go = self._eps_to_go - 1

        # Move Entities
        self._move_stag()
        self.A_AGENT = self._move_entity(self.A_AGENT, agent_moves[0])
        self.B_AGENT = self._move_entity(self.B_AGENT, agent_moves[1])

        # Get Rewards
        iteration_rewards = self._calc_reward()

        # Reset prey if it was caught
        if iteration_rewards == (5, 5):
            self.STAG = self.GRID_W - 2, self.GRID_H - 1
        elif 1 in iteration_rewards:
            self._respawn_plants()

        game_done = self._eps_to_go == 0

        if game_done:
            self._eps_to_go = self._eps_per_game
            self.reset_entities()

        if self._enable_render:
            obs = self.RENDERER.update()
        else:
            obs = self._coord_observation()

        return obs, iteration_rewards, game_done

    def _coord_observation(self):
        plants = self.PLANTS
        a_obs = self.A_AGENT, self.B_AGENT, self.STAG, *plants
        b_obs = self.B_AGENT, self.A_AGENT, self.STAG, *plants

        return a_obs, b_obs

    """
    Movement Methods
    """

    def _seek_agent(self, agent_to_seek):
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
            options.append("LEFT")
        if right:
            options.append("RIGHT")
        if up:
            options.append("UP")
        if down:
            options.append("DOWN")

        if not options:
            options = ["UP", "DOWN", "LEFT", "RIGHT"]

        return self._move_entity(self.STAG, choice(options))

    def _move_stag(self):
        stag_x, stag_y = self.STAG[0], self.STAG[1]
        a_dist = hypot(stag_x - self.A_AGENT[0], stag_y - self.A_AGENT[1])
        b_dist = hypot(stag_x - self.B_AGENT[0], stag_y - self.B_AGENT[1])

        if a_dist < b_dist:
            agent_to_seek = 'a'
        else:
            agent_to_seek = 'b'

        self.STAG = self._seek_agent(agent_to_seek)

    def _move_entity(self, entity_pos, action):
        if action == 'UP':
            return self._move_up(entity_pos)
        elif action == 'DOWN':
            return self._move_down(entity_pos)
        elif action == 'LEFT':
            return self._move_left(entity_pos)
        elif action == 'RIGHT':
            return self._move_right(entity_pos)

    def _reset_agents(self):
        self.A_AGENT = [0, 0]
        self.B_AGENT = [self.GRID_W - 1, 0]

    def reset_entities(self):
        self._reset_agents()
        self.STAG = [self.GRID_W // 2, self.GRID_H // 2]
        self.PLANTS = self._spawn_plants()

    def _random_move(self, pos):
        if randint(0, 1) == 0:
            if randint(0, 1) == 0:
                return self._move_left(pos)
            else:
                return self._move_right(pos)
        else:
            if randint(0, 1) == 0:
                return self._move_up(pos)
            else:
                return self._move_down(pos)

    def _move_left(self, pos):
        new_x = pos[0] - 1
        if new_x == -1:
            new_x = 0
        return new_x, pos[1]

    def _move_right(self, pos):
        new_x = pos[0] + 1
        if new_x == self.GRID_W:
            new_x = self.GRID_W - 1
        return new_x, pos[1]

    def _move_up(self, pos):
        new_y = pos[1] - 1
        if new_y == -1:
            new_y = 0
        return pos[0], new_y

    def _move_down(self, pos):
        new_y = pos[1] + 1
        if new_y == self.GRID_H:
            new_y = self.GRID_H - 1
        return pos[0], new_y

    def make_random_moves(self):
        self.A_AGENT = self._random_move(self.A_AGENT)
        self.B_AGENT = self._random_move(self.B_AGENT)
        self.STAG = self._random_move(self.STAG)

    """
    Properties
    """

    @property
    def GRID_DIMENSIONS(self):
        return self.GRID_W, self.GRID_H

    @property
    def GRID_W(self):
        return int(self._grid_size[0])

    @property
    def GRID_H(self):
        return int(self._grid_size[1])

    @property
    def AGENTS(self):
        return self._a_pos, self._b_pos

    @property
    def A_AGENT(self):
        return self._a_pos

    @A_AGENT.setter
    def A_AGENT(self, new_pos):
        self._a_pos[0], self._a_pos[1] = new_pos[0], new_pos[1]

    @property
    def B_AGENT(self):
        return self._b_pos

    @B_AGENT.setter
    def B_AGENT(self, new_pos):
        self._b_pos[0], self._b_pos[1] = new_pos[0], new_pos[1]

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

    @property
    def RENDERER(self):
        return self._renderer
