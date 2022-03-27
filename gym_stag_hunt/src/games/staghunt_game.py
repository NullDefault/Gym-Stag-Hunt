from numpy import zeros, uint8, array, hypot

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame

from gym_stag_hunt.src.utils import (
    overlaps_entity,
    place_entity_in_unoccupied_cell,
    spawn_plants,
    respawn_plants,
)

# Entity Keys
A_AGENT = 0
B_AGENT = 1
STAG = 2
PLANT = 3


class StagHunt(AbstractGridGame):
    def __init__(
        self,
        stag_reward,
        stag_follows,
        run_away_after_maul,
        opponent_policy,
        forage_quantity,
        forage_reward,
        mauling_punishment,
        # Super Class Params
        window_title,
        grid_size,
        screen_size,
        obs_type,
        load_renderer,
        enable_multiagent,
    ):
        """
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        super(StagHunt, self).__init__(
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
        )

        # Config
        self._stag_follows = stag_follows
        self._run_away_after_maul = run_away_after_maul
        self._opponent_policy = opponent_policy

        # Reinforcement Variables
        self._stag_reward = stag_reward  # record RL values as attributes
        self._forage_quantity = forage_quantity
        self._forage_reward = forage_reward
        self._mauling_punishment = mauling_punishment

        # State Variables
        self._tagged_plants = []  # harvested plants that need to be re-spawned

        # Entity Positions
        self._stag_pos = zeros(2, dtype=uint8)
        self._plants_pos = []
        self.reset_entities()  # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == "image" or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.hunt_renderer import HuntRenderer

            self._renderer = HuntRenderer(
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
                rewards = self._stag_reward, self._stag_reward  # Successful stag hunt
            else:
                if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                    rewards = (
                        self._mauling_punishment,
                        self._forage_reward,
                    )  # A is mauled, B foraged
                else:
                    rewards = (
                        self._mauling_punishment,
                        0,
                    )  # A is mauled, B did not forage

        elif overlaps_entity(self.B_AGENT, self.STAG):
            """
            we already covered the case where a and b are both on the stag,
            so we can skip that check here
            """
            if self._overlaps_plants(self.A_AGENT, self.PLANTS):
                rewards = (
                    self._forage_reward,
                    self._mauling_punishment,
                )  # A foraged, B is mauled
            else:
                rewards = 0, self._mauling_punishment  # A did not forage, B is mauled

        elif self._overlaps_plants(self.A_AGENT, self.PLANTS):
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                rewards = (
                    self._forage_reward,
                    self._forage_reward,
                )  # Both agents foraged
            else:
                rewards = self._forage_reward, 0  # Only A foraged

        else:
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                rewards = 0, self._forage_reward  # Only B foraged
            else:
                rewards = 0, 0  # No one got anything

        return float(rewards[0]), float(rewards[1])

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: If multi-agent, a tuple of actions. Otherwise a single action and the opponent takes an
                            action according to its established policy.
        :return: observation, rewards, is the game done
        """
        # Move Entities
        self._move_stag()
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
                        self._seek_entity(self.B_AGENT, self.STAG),
                    ]
                )

        # Get Rewards
        iteration_rewards = self._calc_reward()

        # Reset prey if it was caught
        if iteration_rewards == (self._stag_reward, self._stag_reward):
            self.STAG = place_entity_in_unoccupied_cell(
                grid_dims=self.GRID_DIMENSIONS,
                used_coordinates=self.PLANTS + self.AGENTS + [self.STAG],
            )
        elif (
            self._run_away_after_maul and self._mauling_punishment in iteration_rewards
        ):
            self.STAG = place_entity_in_unoccupied_cell(
                grid_dims=self.GRID_DIMENSIONS,
                used_coordinates=self.PLANTS + self.AGENTS + [self.STAG],
            )
        elif self._forage_reward in iteration_rewards:
            new_plants = respawn_plants(
                plants=self.PLANTS,
                tagged_plants=self._tagged_plants,
                grid_dims=self.GRID_DIMENSIONS,
                used_coordinates=self.AGENTS + [self.STAG],
            )
            self._tagged_plants = []
            self.PLANTS = new_plants

        obs = self.get_observation()
        info = {}

        if self._enable_multiagent:
            if self._obs_type == "coords":
                return (
                    (obs, self._flip_coord_observation_perspective(obs)),
                    iteration_rewards,
                    False,
                    info,
                )
            else:
                return (obs, obs), iteration_rewards, False, info
        else:
            return obs, iteration_rewards[0], False, info

    def _coord_observation(self):
        """
        :return: list of all the entity coordinates
        """
        shipback = [self.A_AGENT, self.B_AGENT, self.STAG]
        shipback = shipback + self.PLANTS
        return array(shipback).flatten()

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
        if agent_to_seek == "b":
            agent = self.B_AGENT

        move = self._seek_entity(self.STAG, agent)

        return self._move_entity(self.STAG, move)

    def _move_stag(self):
        """
        Moves the stag towards the nearest agent.
        :return:
        """
        if self._stag_follows:
            stag, agents = self.STAG, self.AGENTS
            a_dist = hypot(
                int(agents[0][0]) - int(stag[0]), int(agents[0][1]) - int(stag[1])
            )
            b_dist = hypot(
                int(agents[1][0]) - int(stag[0]), int(agents[1][1]) - int(stag[1])
            )

            if a_dist < b_dist:
                agent_to_seek = "a"
            else:
                agent_to_seek = "b"

            self.STAG = self._seek_agent(agent_to_seek)
        else:
            self.STAG = self._move_entity(self.STAG, self._random_move(self.STAG))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._reset_agents()
        self.STAG = [self.GRID_W // 2, self.GRID_H // 2]
        self.PLANTS = spawn_plants(
            grid_dims=self.GRID_DIMENSIONS,
            how_many=self._forage_quantity,
            used_coordinates=self.AGENTS + [self.STAG],
        )

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
        self._plants_pos = new_pos

    @property
    def ENTITY_POSITIONS(self):
        return {
            "a_agent": self.A_AGENT,
            "b_agent": self.B_AGENT,
            "stag": self.STAG,
            "plants": self.PLANTS,
        }
