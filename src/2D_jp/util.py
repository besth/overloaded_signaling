from enum import Enum

GOAL_SPACE = [{0}, {1}, set()]
GOAL_REWARD = 100
ACTION_COST = 1
PICKUP_COST = 10


# state table encodings
class StateEncoding(Enum):
    TERMINAL = 0
    BAD = -1
    GOOD = 1
