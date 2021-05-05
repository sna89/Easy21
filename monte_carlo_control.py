from constants import Constant, Action
from gym import spaces
import numpy as np
from typing import Tuple


class MonteCarloControl:
    def __init__(self, env, observation_space: spaces, action_space: spaces):
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space

        self.policy = None
        self._init_policy()

        self.Q = None
        self._init_Q()

    def _init_policy(self):
        self.policy = {
            player_sum:
                {
                    dealer_card: [(0.5, Action.STICK), (0.5, Action.HIT)]
                    for dealer_card in range(Constant.CARD_LOWEST_VALUE,
                                             Constant.CARD_HIGHEST_VALUE + 1)
                }
            for player_sum in range(1, Constant.MAX_SUM + 1)
        }

    def _init_Q(self):
        self.Q = dict()
        for player_sum in range(1, self.observation_space[0].n):
            for dealer_card in range(1, self.observation_space[1].n + 1):
                for action in range(self.action_space.n):
                    self.Q[((player_sum, dealer_card), action)] = 0

    def generate_episode(self):
        episode = list()
        done = False

        next_state = self.env.reset()
        # action, reward, next_state, done = self._generate_single_iteration(next_state)
        #
        # sar_tuple = (next_state, action, reward)
        # episode.append(sar_tuple)
        while not done:
            current_state = next_state
            action, reward, next_state, done = self._generate_single_iteration(current_state)
            sar_tuple = (current_state, action, reward)
            episode.append(sar_tuple)
        return episode

    def _generate_single_iteration(self, state):
        action = self._choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        return action, reward, next_state, done

    def _choose_action(self, state: Tuple):
        action_prob = [prob for prob, action in self.policy[state[0]][state[1]]]
        actions = [action for prob, action in self.policy[state[0]][state[1]]]
        action = np.random.choice(actions, p=action_prob)
        return action
