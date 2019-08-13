import os
import sys
import pygame

import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FLOOR_COLOR = (253, 214, 146)


class Battery(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(Battery, self).__init__()
        self.scale = scale
        self.image = pygame.image.load(os.path.join('images', 'battery.png')).convert()
        self.image = pygame.transform.scale(self.image, self.scale)
        self.rect = self.image.get_rect()

class Agent(pygame.sprite.Sprite):
    def __init__(self, scale, screen_size, ID):
        super(Agent, self).__init__()
        self.scale = scale
        self.screen_size = screen_size
        self.id = ID
        self.image = pygame.image.load(os.path.join('images', 'smiley_face.png'))
        self.image = pygame.transform.scale(self.image, self.scale)
        self.rect = self.image.get_rect()
    
    # TODO.
    def update(self, all_action):
        new_pos = [self.rect.x + all_action[self.id][0], self.rect.y + all_action[self.id][1]]
        if 0 < new_pos[0] < self.screen_size[0] - self.scale[0] and 0 < new_pos[1] < self.screen_size[1] - 2 * self.scale[1]:
            self.rect.x = new_pos[0]
            self.rect.y = new_pos[1]


class Policy:
    def __init__(self, step_size):
        self.step_size = step_size
    
    def __call__(self, action_seq, agent):
        '''
            action_seq: [["up/down/right/left", #steps], ...]
        '''
        action_list = list()
        for action in action_seq:
            if action[1] <= 0:
                continue

            if action[0] == "up":
                a = [[0, -self.step_size], [0, -self.step_size]]
            elif action[0] == "down":
                a = [[0, self.step_size], [0, self.step_size]] 
            elif action[0] == "right":
                a = [[self.step_size, 0], [self.step_size, 0]]
            elif action[0] == "left":
                a = [[-self.step_size, 0], [-self.step_size, 0]] 
            else:
                a = [[0, 0], [0, 0]] 
            
            if agent == "speaker":
                a[1] = [0, 0]
            elif agent == "listener":
                a[0] = [0, 0]

            for _ in range(action[1]):
                action_list.append(a)

        return action_list

