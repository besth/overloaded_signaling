import numpy as np
import threading

from env import Env2D
from algo import ValueIteration
from util import GOAL_SPACE, WORLD_DICT


def create_envs(env_type):
    env_size = (2, 8)
    obj_poss = [(0, 0), (0, 6)]

    terminal_pos = (0, env_size[1] - 1)
    envs = [
        Env2D(env_size=env_size,
              env_type=env_type,
              goal=goal,
              obj_poss=obj_poss,
              terminal_pos=terminal_pos) for goal in GOAL_SPACE
    ]

    return envs


def compute_v_parallel(env_type, gamma, epsilon, tau):
    # create envs based on goal
    envs = create_envs(env_type)
    print("Envs created")
    VIs = [ValueIteration(gamma, epsilon, tau, env) for env in envs]
    print("VI created")

    threads = [threading.Thread(target=vi) for vi in VIs]

    print("threads created")

    for thread in threads:
        thread.start()
        print("One thread started")

    for thread in threads:
        thread.join()
        print("One thread finished")


if __name__ == "__main__":
    env_type = 0
    gamma = 0.99
    epsilon = 0.01
    tau = 0.2

    compute_v_parallel(env_type, gamma, epsilon, tau)
    print("Done")