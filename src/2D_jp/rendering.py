import os
import copy
import time
import numpy as np
import pygame
from pygame.locals import *

from env import Env2D
from util import GOAL_SPACE, Color


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
        # os.system('cls' if os.name == 'nt' else 'clear')


class Render:
    def __init__(self,
                 fps=1000,
                 screen_size=(480, 112),
                 env_type=1,
                 obj_init_pos=[(0, 0), (0, 6)],
                 ag_init_pos=[(1, 3), (0, 7)]):
        self.fps = fps
        self.screen_size = screen_size
        self.env_type = env_type
        self.obj_init_pos = obj_init_pos
        self.ag_init_pos = ag_init_pos

        self.fpsClock = pygame.time.Clock()

        pygame.init()
        # initialize surface
        self.display = pygame.display.set_mode(size=self.screen_size)
        self.caption = "Hands-Free" if self.env_type == 0 else "Hands-tied"
        pygame.display.set_caption(self.caption)

        # scaling factors
        self.ag_scale = (45, 45)
        self.obj_scale = (30, 40)
        self.pos_scale = 61.5

        # load images
        self.obj_img = pygame.image.load("2D_jp/images/battery.png")
        self.obj_img = pygame.transform.scale(self.obj_img, (50, 60))
        self.ag_img = pygame.image.load("2D_jp/images/agent.png")
        self.ag_img = pygame.transform.scale(self.ag_img, (45, 45))
        self.sp_img = pygame.image.load("2D_jp/images/max.png")
        self.sp_img = pygame.transform.scale(self.sp_img, (45, 45))
        self.rec_img = pygame.image.load("2D_jp/images/qingyi.png")
        self.rec_img = pygame.transform.scale(self.rec_img, (45, 45))
        self.ag_imgs = [self.rec_img, self.sp_img]

        # initialize background and positions
        self.reset()

    def reset(self):
        self.display.fill(Color.L_GREY.value)

        # pygame.draw.line(self.display, (0, 0, 0), (650, 300), (650, 450), 10)

        for i, ag_pos in enumerate(self.ag_init_pos):
            self.display.blit(
                self.ag_imgs[i],
                list(x * self.pos_scale for x in reversed(ag_pos)))

        for obj_pos in self.obj_init_pos:
            self.display.blit(
                self.obj_img,
                list(x * self.pos_scale for x in reversed(obj_pos)))

            # while True:
            #     pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

    def draw_grid(self, line_dis=60):
        # vertical
        num_lines = self.screen_size[0] // line_dis
        for i in range(num_lines):
            pygame.draw.line(self.display, Color.BLACK.value,
                             (line_dis * i, 0),
                             (line_dis * i, self.screen_size[1]), 1)

        # horizontal
        pygame.draw.line(self.display, Color.BLACK.value,
                         (0, self.screen_size[1] / 2),
                         (self.screen_size[0], self.screen_size[1] / 2), 1)

    def draw_wall(self):
        pygame.draw.line(self.display, Color.BLACK.value,
                         (self.screen_size[0] - 60, 0),
                         (self.screen_size[0] - 60, self.screen_size[1] / 2),
                         5)

        pygame.draw.line(self.display, Color.BLACK.value,
                         (self.screen_size[0] - 60, self.screen_size[1] / 2),
                         (self.screen_size[0], self.screen_size[1] / 2), 5)

    def __call__(self, states):
        # erase
        self.display.fill(Color.L_GREY.value)
        self.draw_grid()
        if self.env_type == 1:
            self.draw_wall()
        # if self.env_type == 1:
        #     pygame.draw.line(self.display, (0, 0, 0), (1025, 0), (1025, 75),
        #                      10)
        #     pygame.draw.line(self.display, (0, 0, 0), (1025, 75), (1100, 75),
        #                      10)

        # extract positions
        rec_pos, sp_pos, rec_store, sp_store = states
        self.display.blit(self.sp_img,
                          list(x * self.pos_scale for x in reversed(sp_pos)))
        self.display.blit(self.rec_img,
                          list(x * self.pos_scale for x in reversed(rec_pos)))

        # extract storage
        for obj_i in range(len(self.obj_init_pos)):
            if obj_i not in sp_store.union(rec_store):
                self.display.blit(
                    self.obj_img,
                    list(x * self.pos_scale
                         for x in reversed(self.obj_init_pos[obj_i])))

        pygame.display.update()


# render = Render()
# render.reset()
# states = ((1, 3), (0, 7), set(), set())
# while True:
#     render(states)
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             pygame.quit()
