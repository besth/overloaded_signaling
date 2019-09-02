from enum import Enum

GOAL_REWARD = 100
ACTION_COST = 1
PICKUP_COST = 10

# goal inference
SIGNAL_DICT_INV = {
    0: "help",
    1: "help-A",
    2: "help-B",
    3: "help-Any",
    4: "get-A",
    5: "get-B"
}
SIGNAL_REW_DICT = {0: -0.01, 1: -2, 2: -2, 3: -2, 4: 0, 5: 0}
WORLD_DICT_INV = {0: "hands-free", 1: "hands-tied"}

# TODO: combine goal space and dictionary
GOAL_SPACE = [{0}, {1}, set()]
GOAL_DICT_INV = {0: 'A', 1: 'B', 2: "Any"}
# GOAL_DICT_INV = {0: 'A', 1: 'B'}

SIGNAL_DICT = {v: k for k, v in SIGNAL_DICT_INV.items()}
WORLD_DICT = {v: k for k, v in WORLD_DICT_INV.items()}
GOAL_DICT = {v: k for k, v in GOAL_DICT_INV.items()}


# state table encodings
class StateEncoding(Enum):
    TERMINAL = 0
    BAD = -1
    GOOD = 1


class Color(Enum):
    WHITE = (255, 255, 255)
    L_GREY = (230, 230, 230)
    BLACK = (0, 0, 0)
