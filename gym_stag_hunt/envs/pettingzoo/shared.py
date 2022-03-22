from pettingzoo.utils import wrappers
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from gym.spaces import Box
import cv2
import numpy as np


def default_wrappers(env_init):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


class PettingZooEnv(AECEnv):
    def __init__(self, og_env, obs_shape=(42, 42)):
        super().__init__()
        self.env = og_env
        self.possible_agents = ["player_" + str(n) for n in range(2)]
        self.agents = self.possible_agents[:]

        self.shape = obs_shape
        observation_space = Box(low=0, high=255, shape=self.shape + self.env.observation_space.shape[2:],
                                dtype=np.uint8)
        self.observation_spaces = {agent: observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.env.action_space for agent in self.possible_agents}
        self.has_reset = True

        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_observation = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        self.t = 0
        self.last_rewards = [0, 0]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self):
        obs = self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.current_observation = {agent: obs for agent in self.agents}
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.t = 0

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        obs, rewards, done, info = self.env.step(actions)
        self.last_rewards = rewards

        info = {"t": self.t}

        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.current_observation[agent] = obs[idx]
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        returned_observation = self.current_observation[agent]
        returned_observation = cv2.resize(returned_observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return returned_observation

    def render(self, mode='human'):
        self.env.render(mode)

    def state(self):
        pass

    def close(self):
        self.env.close()
