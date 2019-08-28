# goal inference
SIGNAL_DICT_INV = {
    0: "help",
    1: "help-A",
    2: "help-B",
    3: "help-Any",
    4: "get-A",
    5: "get-B"
}
SIGNAL_REW_DICT = {
    0: -2,
    1: -10,
    2: -10,
    3: -5,
    4: {
        0: -100,  # world index: 0 --> "hands-free", 1 --> "hands-tied".
        1: -100
    },
    5: {
        0: -1,  # world index: 0 --> "hands-free", 1 --> "hands-tied".
        1: -100
    }
}
WORLD_DICT_INV = {0: "hands-free", 1: "hands-tied"}
# GOAL_DICT_INV = {0: 'A', 1: 'B'}
GOAL_DICT_INV = {0: 'A', 1: 'B'}

SIGNAL_DICT = {v: k for k, v in SIGNAL_DICT_INV.items()}
WORLD_DICT = {v: k for k, v in WORLD_DICT_INV.items()}
GOAL_DICT = {v: k for k, v in GOAL_DICT_INV.items()}

# env
ENV_TYPE = [0, 2]
GOAL_REWARD = 100
ACTION_COST = 1
