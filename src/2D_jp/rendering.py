import os
import copy
import time
import numpy as np

from env import Env2D
from util import GOAL_SPACE


class Render2DGrid:
    def __init__(self, env):
        self.env = env
        self.goal_poss = copy.deepcopy(self.env.obj_poss)
        self.reset()

    def reset(self):
        self.grid = np.asarray([['.' for _ in range(self.env.env_size[1])]
                                for _ in range(self.env.env_size[0])])

        for gp in self.goal_poss:
            self.grid[gp[0], gp[1]] = 'G'

    def __call__(self, state):
        self.reset()
        for i, pos in enumerate(state[:2]):
            x = pos[0]
            y = pos[1]
            names = ['A', 'B']
            if self.grid[x, y] == '.':
                self.grid[x, y] = names[i]
            elif pos in self.goal_poss:
                self.grid[x, y] = names[i]
                self.goal_poss.remove(pos)
            else:
                self.grid[x, y] = 'X'

        print("\n".join(map(lambda x: ' '.join(x), self.grid)))
        time.sleep(0.3)
        # for more continuous rendering
        os.system('cls' if os.name == 'nt' else 'clear')
