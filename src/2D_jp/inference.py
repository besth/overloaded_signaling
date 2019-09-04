import os
import copy
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from env import Env2D
from algo import one_step_lookahead
from util import SIGNAL_DICT, WORLD_DICT, GOAL_DICT, SIGNAL_DICT, WORLD_DICT, GOAL_DICT, GOAL_REWARD, SIGNAL_REW_DICT, GOAL_SPACE


def softmax(values: list, temp=1.0):
    reweighted = np.asarray(values) / temp
    exp_values = np.exp(reweighted)
    sm_values = exp_values / np.sum(exp_values)

    return sm_values


def log_softmax(values, temp):
    return np.log(softmax(values, temp))


class GoalInference:
    def __init__(self, beta, temp, env):
        self.beta = beta
        self.temp = temp
        self.env = env

        self.num_signals = len(SIGNAL_DICT)
        self.num_worlds = len(WORLD_DICT)
        self.num_goals = len(GOAL_DICT)

        self.signals = list(range(self.num_signals))
        self.worlds = list(range(self.num_worlds))
        self.goals = list(range(self.num_goals))

        self.lexicon = self.create_lexicon()
        self.goal_prior = self.get_goal_prior()

        self.actions = self.env.actions
        if self.env.env_type == WORLD_DICT["hands-tied"]:
            self.action_sigs = list(it.product(*[[(0, 0)], self.signals]))
        else:
            self.action_sigs = list(it.product(*[self.actions, self.signals]))

        # prune action_sigs
        self.action_sigs_pruned = copy.deepcopy(self.action_sigs)
        new_list = []
        for a_s in self.action_sigs_pruned:
            a, s = a_s
            # if a != (0, 0) and s in range(4):
            #     continue
            # if a == (0, 0) and s in range(4, 6):
            #     continue
            if a != (0, 0) and s in range(1):
                continue
            if a == (0, 0) and s in range(1, self.num_signals):
                continue

            new_list.append(a_s)

        self.action_sigs_pruned = new_list

    def get_goal_prior(self):
        return [1 / len(self.goals) for _ in self.goals]

    def create_lexicon(self):
        lex = np.zeros((len(self.signals), len(self.goals)))

        # sig = "help" --> [0.5, 0.5]
        lex[SIGNAL_DICT["help"]][:] = 1 / len(self.goals)
        # lex[SIGNAL_DICT["help"]][:] = 1

        # sig = "help-A" --> [1, 0]
        # lex[SIGNAL_DICT["help-A"]][GOAL_DICT['A']] = 1
        # lex[SIGNAL_DICT["help-A"]][GOAL_DICT['B']] = 0
        # lex[SIGNAL_DICT["help-A"]][GOAL_DICT["Any"]] = 0

        # sig = "help-B" --> [0, 1]
        # lex[SIGNAL_DICT["help-B"]][GOAL_DICT['A']] = 0
        # lex[SIGNAL_DICT["help-B"]][GOAL_DICT['B']] = 1
        # lex[SIGNAL_DICT["help-B"]][GOAL_DICT["Any"]] = 0

        # sig = "help-Any" --> [0.5, 0.5]
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['A']] = 0
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['B']] = 0
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['A']] = 0.5
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['B']] = 0.5
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT["Any"]] = 1
        # lex[SIGNAL_DICT["help-Any"]][:] = 1

        # sig = "get-A" --> [1, 0]
        lex[SIGNAL_DICT["get-A"]][GOAL_DICT['A']] = 1
        lex[SIGNAL_DICT["get-A"]][GOAL_DICT['B']] = 0
        # lex[SIGNAL_DICT["get-A"]][GOAL_DICT["Any"]] = 0

        # sig = "get-B" --> [0, 1]
        lex[SIGNAL_DICT["get-B"]][GOAL_DICT['A']] = 0
        lex[SIGNAL_DICT["get-B"]][GOAL_DICT['B']] = 1
        # lex[SIGNAL_DICT["get-B"]][GOAL_DICT["Any"]] = 0

        return lex

    def reward_signal(self, signal: int):
        return SIGNAL_REW_DICT[signal]

    def reward_goal(self, goal, action_sig: tuple, curr_state, q_values):
        action, signal = action_sig
        # print(action_sig)

        q_values_given_a_2 = [
            q for i, q in enumerate(q_values)
            if self.env.action_space[i][1] == action
        ]
        # print(action_sig, max(q_values_given_a_2))

        # goal_value = max(log_softmax(q_values_given_a_2, temp=self.temp))
        reward = max(q_values_given_a_2)

        # reward = self.lexicon[signal][goal] * max(q_values_given_a_2)
        # if signal == 4:
        #     print(self.lexicon[signal][goal], reward, "signal is 4, get A")

        return reward

    def compute_likelihood(self, goal, world, curr_state, q_values):

        reward_goals = [
            self.reward_goal(goal, action_sig, curr_state, q_values)
            for action_sig in self.action_sigs_pruned
        ]
        # print(list(zip(self.action_sigs_pruned, reward_goals)))
        # exit()

        if len(reward_goals) != 1:
            reward_goals = softmax(reward_goals, self.temp)
        print(list(zip(self.action_sigs_pruned, reward_goals)))
        # exit()

        llhs = [
            np.exp(self.beta *
                   (self.reward_signal(action_sig[1]) +
                    self.lexicon[action_sig[1]][goal] * reward_goals[i]))
            for i, action_sig in enumerate(self.action_sigs_pruned)
        ]

        if len(llhs) != 1:
            llhs = np.asarray(llhs) / np.sum(llhs)
        # print(normalized_llhs)
        # exit()

        return llhs

    def __call__(self, action_sig, world, curr_state, q_values_list):
        print(q_values_list)

        goal_dist = []
        for i, goal in enumerate(self.goals):
            q_values = q_values_list[i]

            goal_prob = self.compute_likelihood(
                goal, world, curr_state,
                q_values)[self.action_sigs_pruned.index(action_sig)]
            print(goal_prob)
            goal_dist.append(goal_prob)

        # normalize
        goal_dist = np.array(goal_dist) / np.sum(goal_dist)

        return list(goal_dist)


# env_size = (2, 8)
# env_type = 0
# goal_ind = 0
# goal = GOAL_SPACE[goal_ind]
# obj_poss = [(0, 0), (0, 6)]

# terminal_pos = (0, env_size[1] - 1)
# env = Env2D(env_size=env_size,
#             env_type=env_type,
#             goal=goal,
#             obj_poss=obj_poss,
#             terminal_pos=terminal_pos)

# beta = 0.1
# GI = GoalInference(beta, env, 0, 0)