
import random
import numpy as np
import itertools as it

class Env_1D:
    def __init__(self, env_length, goal):
        self.env_length = env_length
        
        self.actions = [-1, 0, 1]
        self.action_space = list(tuple(it.product(*[self.actions, self.actions])))

        self.states = list(range(self.env_length))
        self.state_space = list(tuple(it.product(*[self.states, self.states])))

        self.goal_poss = [0, self.env_length - 2]
        # self.goal_space = [[0], [1], [0, 1], [None]]
        # self.goal = random.sample(self.goal_space, 1)[0]
        self.goal = goal

        self.action_cost = 1
        self.target_reward = 1

        self.collection = []


    def transition(self, states, actions):
        next_states = list(np.add(states, actions))
        for i, ns in enumerate(next_states):
            if ns not in self.states:
                next_states[i] = states[i]

        return tuple(next_states)


    def reward_2g(self, states, actions):
        next_states = self.transition(states, actions)
        
        reward = 0
        # action cost
        for action in actions:
            if action != 0:
                reward -= self.action_cost

        if self.done_2g(next_states):
            reward += self.target_reward

        return reward

        
    def done_2g(self, states):
        for s in states:
            for g in self.goal_poss:
                if s == g:
                    return True

        return False


    def reward(self, states, actions):
        next_states = self.transition(states, actions)
        
        reward = 0
        # action cost
        for action in actions:
            if action != 0:
                reward -= self.action_cost

        # target reward
        for ns in next_states:
            for g in self.goal_poss:
                if ns == g:
                    self.collection.append(self.goal_poss.index(g))


        # if self.goal == [None]:
        #     reward += self.target_reward if self.collection else 0
        # else:
        #     reward += self.target_reward if set(self.goal) == set(self.collection) else 0

        # reward += self.target_reward if self.done() else 0
        if self.done():
            reward += self.target_reward
            self.collection = []

        return reward


    def done(self):
        if self.goal == [None]:
            if self.collection:
                return True
        else:
            if set(self.goal) == set(self.collection):
                return True
        
        return False

    def set_goal(self, goal):
        self.goal = goal


class Env_1D_single:
    def __init__(self, env_length, goal_index):
        self.env_length = env_length
        self.goal_index = goal_index

        self.goal_poss = [0, self.env_length - 2]
        self.state_space = list(range(self.env_length))
        self.state_space_no_terminal = [x for x in self.state_space if x not in self.goal_poss]
        self.action_space = [-1, 0, 1]

        self.target_reward = 10
        self.action_cost = 1

    def transition(self, state, action):
        # print(state, action)
        next_state = state + action
        if next_state not in self.state_space:
            next_state = state

        return next_state

    def reward(self, state, action):
        next_state = self.transition(state, action)
        # next_state = state + action
        reward = 0
        if next_state == self.goal_poss[self.goal_index]:
            reward += self.target_reward

        if action != 0:
            reward -= self.action_cost

        return reward

    def done(self, state):
        for g_pos in self.goal_poss:
            if state == g_pos:
                return True
        
        return False


class Env_1D_full:
    def __init__(self, env_length, goal_index):
        self.env_length = env_length
        self.goal_index = goal_index

        self.goal_poss = [0, self.env_length - 2]
        self.state_space = list(range(self.env_length))
        self.action_space = [-1, 0, 1]

        self.target_reward = 1
        self.action_cost = 10

    def transition(self, state, action):
        # print(state, action)
        next_state = state + action
        if next_state not in self.state_space:
            next_state = state

        return next_state

    def reward(self, state, action):
        next_state = self.transition(state, action)
        # next_state = state + action
        reward = 0
        if next_state == self.goal_poss[self.goal_index]:
            reward += self.target_reward

        # Hack!!
        if next_state == self.goal_poss[1]:
            reward -= 100

        if action != 0:
            reward -= self.action_cost

        return reward

    def done(self, state):
        for g_pos in self.goal_poss:
            if state == g_pos:
                return True
        
        return False








# env_length = 10
# env = Env_1D(env_length)


