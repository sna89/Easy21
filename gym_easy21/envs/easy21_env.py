import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from enum import Enum
import copy


class Color(Enum):
    RED = 0
    BLACK = 1


class Easy21Env(gym.Env):
    metadata = {'render.modes': ['human']}

    CARD_HIGHEST_VALUE = 10
    CARD_LOWEST_VALUE = 1
    COLORS = [Color.RED, Color.BLACK]
    COLORS_PROB = [1 / float(3), 2 / float(3)]

    def __init__(self):
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(22), spaces.Discrete(10)]
        )
        self.action_space = spaces.Discrete(2)

        self.last_action = None

        self.player_sum = None
        self.dealer_card = None
        self.dealer_sum = None

        self.reset()

    def step(self, action):
        assert action in self.action_space
        reward = 0
        done = False
        self.last_action = action

        if action:  # hit
            self._draw_card_and_add_to_sum(is_player=True)

            if self._is_bust(is_player=True):
                reward = -1
                done = True
                self._switch_to_bust_state()

            return self._get_obs(), reward, done, {}

        else:  # stick
            while self.dealer_sum < 17:
                self._draw_card_and_add_to_sum(is_player=False)

            done = True
            if self._is_bust(is_player=False):
                reward = 1
            else:
                reward = self._compare_sums()
            return self._get_obs(), reward, done, {}

    def reset(self):
        self.player_sum, _ = self._draw_card()
        self.dealer_card, _ = self._draw_card()
        self.dealer_sum = copy.deepcopy(self.dealer_card)
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_obs(self):
        return self.player_sum, self.dealer_card

    def _draw_card(self):
        value = np.random.randint(self.CARD_LOWEST_VALUE, self.CARD_HIGHEST_VALUE + 1, size=1)
        color = np.random.choice(self.COLORS, p=self.COLORS_PROB)
        return value, color

    def _is_bust(self, is_player=True):
        if is_player:
            return self.player_sum > 21 or self.player_sum < 1
        else:
            return self.dealer_sum > 21 or self.dealer_sum < 1

    def _switch_to_bust_state(self):
        self.player_sum = -1

    def _draw_card_and_add_to_sum(self, is_player):
        card_value, card_color = self._draw_card()
        sum_ = self.player_sum if is_player else self.dealer_sum
        if card_color == Color.BLACK:
            sum_ += card_value
        elif card_color == Color.RED:
            sum_ -= card_value

        if is_player:
            self.player_sum = sum_
        else:
            self.dealer_sum = sum_

    def _compare_sums(self):
        if self.player_sum > self.dealer_sum:
            return 1
        elif self.player_sum == self.dealer_sum:
            return 0
        else:
            return -1