import copy
import numpy as np
import itertools as it
from util import GOAL_REWARD, ACTION_COST


class Env:
    def __init__(self, env_length, goal, goal_space, env_type):
        self.env_length = env_length
        self.goal = goal
        self.env_type = env_type
        self.goal_space = goal_space

        # action space
        self.actions = [-1, 0, 1]
        self.action_space = list(it.product(*[self.actions, self.actions]))
        # state space
        self.poss = list(range(self.env_length))
        self.collection_space = [{0}, {1}, {0, 1}, set()]
        self.state_space = list(
            it.product(*[self.poss, self.poss, self.collection_space]))

        self.action_cost = ACTION_COST
        self.get_obj_reward = 0
        self.target_reward = GOAL_REWARD

        # maybe not useful
        self.goal_poss = [0, self.env_length - 2]

        # get non terminal states
        self.state_space_no_terminal = [
            s for s in self.state_space if not self.is_terminal_state(s)
        ]
        # print(self.state_space_no_terminal)

    def transition(self, states, actions):
        # first transition positions
        next_states = list(np.add(states[:2], actions))
        for i, ns in enumerate(next_states):
            if ns not in self.poss:
                next_states[i] = states[i]

        # add env restrictions
        # 0: agent 1 can make 0 move
        # if self.env_type == 0:
        #     next_states[1] = states[1]
        # elif self.env_type == 1:
        #     if (next_states[1] != self.env_length - 1) and (
        #             next_states[1] != self.env_length - 2):
        #         next_states[1] = states[1]
        # elif self.env_type == 2
        if self.env_type != 3 and next_states[1] not in list(
                range(self.env_length - self.env_type - 1,
                      self.env_length - 1)):
            next_states[1] = states[1]

        # then transition collections
        curr_collection = copy.copy(states[-1])
        for i, gp in enumerate(self.goal_poss):
            for ap in next_states:
                if gp == ap and (i not in curr_collection):
                    curr_collection.add(i)

        # combine into next state
        next_states.append(curr_collection)

        return tuple(next_states)

    def reward(self, states, actions):
        next_states = self.transition(states, actions)
        curr_collection = next_states[-1]

        reward = 0
        for action in actions:
            if action != 0:
                reward -= self.action_cost

        if self.goal == set() and len(curr_collection) == 1:
            reward += self.target_reward
        elif self.goal != set() and self.goal == curr_collection:
            reward += self.target_reward

        return reward

    def set_goal(self, goal):
        self.goal = goal

        # non-terminal states differ when goal changes
        self.state_space_no_terminal = [
            s for s in self.state_space if not self.is_terminal_state(s)
        ]

    def is_terminal_state(self, state):
        curr_collection = state[2]
        # print("curr collection", curr_collection)

        # if matches the goal --> terminal
        if self.goal != set() and curr_collection == set(self.goal):
            return True

        # early termination if wrong things in collection already
        if self.goal == {0}:
            if 1 in curr_collection:
                return True
        elif self.goal == {0}:
            if 0 in curr_collection:
                return True
        elif self.goal == set():
            if len(curr_collection) == 1 or len(curr_collection) == 2:
                return True

        return False


class PassingEnv(Env):
    def __init__(self, env_length, goal, goal_space, env_type):
        self.terminal = env_length - 3
        super(PassingEnv, self).__init__(env_length=env_length,
                                         goal=goal,
                                         goal_space=goal_space,
                                         env_type=env_type)
        # self.action_space = list(it.product(*[self.actions, [0]]))

        # self.state_space = list(
        #     it.product(*[self.poss, [7], self.collection_space]))

        # self.state_space_no_terminal = [
        #     s for s in self.state_space if not self.is_terminal_state(s)
        # ]

    def is_terminal_state(self, state):
        lis_state = state[0]
        return (super().is_terminal_state(state)
                and (lis_state == self.terminal))

    def reward(self, states, actions):
        next_states = self.transition(states, actions)
        curr_collection = next_states[-1]

        reward = 0
        for action in actions:
            if action != 0:
                reward -= self.action_cost

        if self.goal == set() and len(curr_collection) == 1:
            if next_states[0] == self.terminal:
                reward += self.target_reward
            else:
                reward += self.get_obj_reward
        elif self.goal != set() and self.goal == curr_collection:
            if next_states[0] == self.terminal:
                reward += self.target_reward
            else:
                reward += self.get_obj_reward

        return reward


# goal_space = [{0}, {1}]
# env = PassingEnv(env_length=8, goal={0}, goal_space=goal_space)
# states = (4, 7, {0})
# actions = (1, 0)
# print(env.reward(states, actions))
