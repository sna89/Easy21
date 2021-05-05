from constants import Constant, Action
from gym import spaces
import numpy as np
from typing import Tuple
import math
from itertools import product
from tqdm import tqdm


class MonteCarloControl:
    def __init__(self, env, observation_space: spaces, action_space: spaces):
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space

        self.policy = None
        self._init_policy()

        self.Q = self._init_state_action_mapping()
        self.N = self._init_state_action_mapping()

    def _state_iterator(self):
        return iter([(player_sum, dealer_card) for (player_sum, dealer_card) in
                     product(range(1, self.observation_space[0].n), range(1, self.observation_space[1].n + 1))])

    def _action_iterator(self):
        return iter([action for action in range(self.action_space.n)])

    def _init_policy(self):
        self.policy = {
            state: [(0.5, action) for action in self._action_iterator()]
            for state in self._state_iterator()
        }

    def _init_state_action_mapping(self):
        state_action_mapping = dict()
        for state in self._state_iterator():
            for action in self._action_iterator():
                state_action_mapping[(state, action)] = 0
        return state_action_mapping

    def run(self, gamma=1, n_epsilon=100, iterations=1000000):
        for _ in tqdm(range(iterations)):
            episode = self._generate_episode()
            rewards = [reward for (state, action, reward) in episode]
            visited_states = []
            for step, (state, action, reward) in enumerate(episode):
                self._update_N(state, action)
                self._update_Q(rewards, step, gamma, state, action)
                visited_states.append(state)
            self._update_eps_greedy_policy(n_epsilon, visited_states)

        return self.policy

    def _generate_episode(self):
        episode = list()
        done = False

        next_state = self.env.reset()
        while not done:
            current_state = next_state
            action, reward, next_state, done = self._generate_single_step(current_state)
            episode.append((current_state, action, reward))
        return episode

    def _generate_single_step(self, state):
        action = self._choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        return action, reward, next_state, done

    def _choose_action(self, state: Tuple):
        action_prob = [prob for prob, action in self.policy[state]]
        actions = [action for prob, action in self.policy[state]]
        action = np.random.choice(actions, p=action_prob)
        return action

    @staticmethod
    def _calc_returns(rewards, step=0, gamma=1):
        returns = 0
        for i, reward in enumerate(rewards[step:]):
            returns += math.pow(gamma, i) * reward
        return returns

    def _update_N(self, state, action):
        self.N[(state, action)] += 1

    def _update_Q(self, rewards, step, gamma, state, action):
        returns = self._calc_returns(rewards, step, gamma)
        alpha = 1 / float(self.N[(state, action)])
        self.Q[(state, action)] += alpha * (returns - self.Q[(state, action)])

    def _update_eps_greedy_policy(self, n_epsilon, visited_states):
        for state in visited_states:
            eps = self._calc_state_eps(state, n_epsilon)
            action_prob = self._calc_action_prob_for_state(state, eps)
            self.policy[state] = action_prob

    def _choose_optimal_action(self, state):
        optimal_action = None
        max_q_val = np.NINF

        for action in self._action_iterator():
            q_val = self.Q[(state, action)]

            if q_val > max_q_val:
                optimal_action = action
                max_q_val = q_val

        return optimal_action

    def _calc_state_eps(self, state, n_epsilon):
        visits = 0
        for action in self._action_iterator():
            visits += self.N[(state, action)]
        eps = n_epsilon / (n_epsilon + float(visits))
        return eps

    def _calc_action_prob_for_state(self, state, eps):
        action_prob = []
        num_actions = self.action_space.n
        optimal_action = self._choose_optimal_action(state)
        for action in self._action_iterator():
            prob = eps / num_actions
            if action == optimal_action:
                prob += 1 - eps
            action_prob.append((prob, action))
        return action_prob

    def get_value_function(self):
        value_function = dict()
        for state in self._state_iterator():
            optimal_action = self._choose_optimal_action(state)
            value_function[state] = self.Q[(state, optimal_action)]
        return value_function
