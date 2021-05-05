from enum import Enum


class Color(Enum):
    RED = 0
    BLACK = 1


class Action(Enum):
    STICK = 0
    HIT = 1


class Constant:
    CARD_HIGHEST_VALUE = 10
    CARD_LOWEST_VALUE = 1
    MAX_SUM = 21
