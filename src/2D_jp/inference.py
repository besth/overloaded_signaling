import os
import copy
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from env import Env2D
from util import SIGNAL_DICT, WORLD_DICT, GOAL_DICT, SIGNAL_DICT, WORLD_DICT, GOAL_DICT, GOAL_REWARD, SIGNAL_REW_DICT, GOAL_SPACE


class GoalInference:
    def __init__(self, beta, env, vi_obj, v_table):
        self.beta = beta
        self.env = env
        self.vi_obj = vi_obj
        self.v_table = v_table

        self.num_signals = len(SIGNAL_DICT)
        self.num_worlds = len(WORLD_DICT)
        self.num_goals = len(GOAL_DICT)

        self.signals = list(range(self.num_signals))
        self.worlds = list(range(self.num_worlds))
        self.goals = list(range(self.num_goals))

        self.lexicon = self.create_lexicon()
        self.goal_prior = self.get_goal_prior()

        self.actions = self.env.actions
        self.action_sigs = list(it.product(*[self.actions, self.signals]))

        # prune action_sigs
        self.action_sigs_pruned = copy.deepcopy(self.action_sigs)
        new_list = []
        for a_s in self.action_sigs_pruned:
            a, s = a_s
            if a != (0, 0) and s in range(4):
                continue
            if a == (0, 0) and s in range(4, 6):
                continue

            new_list.append(a_s)

        self.action_sigs_pruned = new_list
        # print(self.action_sigs, len(self.action_sigs))
        # print(self.action_sigs_pruned, len(self.action_sigs_pruned))

    def get_goal_prior(self):
        return [1 / len(self.goals) for _ in self.goals]

    def create_lexicon(self):
        lex = np.zeros((len(self.signals), len(self.goals)))

        # sig = "help" --> [0.5, 0.5]
        lex[SIGNAL_DICT["help"]][:] = 1 / len(self.goals)

        # sig = "help-A" --> [1, 0]
        lex[SIGNAL_DICT["help-A"]][GOAL_DICT['A']] = 1
        lex[SIGNAL_DICT["help-A"]][GOAL_DICT['B']] = 0
        # lex[SIGNAL_DICT["help-A"]][GOAL_DICT["Any"]] = 0

        # sig = "help-B" --> [0, 1]
        lex[SIGNAL_DICT["help-B"]][GOAL_DICT['A']] = 0
        lex[SIGNAL_DICT["help-B"]][GOAL_DICT['B']] = 1
        # lex[SIGNAL_DICT["help-B"]][GOAL_DICT["Any"]] = 0

        # sig = "help-Any" --> [0.5, 0.5]
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['A']] = 0
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['B']] = 0
        lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['A']] = 0.5
        lex[SIGNAL_DICT["help-Any"]][GOAL_DICT['B']] = 0.5
        # lex[SIGNAL_DICT["help-Any"]][GOAL_DICT["Any"]] = 1

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

    def reward_goal(self, goal, action_sig: tuple, curr_state):
        action, signal = action_sig
        q_values = self.vi_obj.one_step_lookahead(self.v_table, curr_state)
        # print(q_values)
        print(list(zip(self.env.action_space, q_values)))
        q_values_given_a_2 = [
            q for i, q in enumerate(q_values)
            if self.env.action_space[i][1] == action
        ]
        print("max q value", max(q_values_given_a_2))
        print(signal, goal, self.lexicon[signal][goal])

        reward = self.lexicon[signal][goal] * max(q_values_given_a_2)

        return reward

        # q_values = self.vi_obj.one_step_lookahead(self.v_table, curr_state)
        # reward = 0
        # for goal in self.goals:
        #     q_values = self.vi_obj[goal].one_step_lookahead(
        #         self.v_table[goal], curr_state)
        # q_values_given_a_2 = []
        # for i, joint_a in enumerate(self.env.action_space):
        #     if joint_a[1] != action:
        #         continue
        #     q_values_given_a_2.append(q_values[i])

        # q_values_given_a_2 = [
        #     q for i, q in enumerate(q_values)
        #     if self.env.action_space[i][1] == action
        # ]

        # reward += self.lexicon[signal][goal] * max(q_values_given_a_2)

        # return reward

    def compute_likelihood(self, goal, world, curr_state):
        llhs = [
            np.exp(self.beta *
                   (self.reward_signal(action_sig[1]) +
                    self.reward_goal(goal, action_sig, curr_state)))
            for action_sig in self.action_sigs_pruned
        ]

        normalized_llhs = np.asarray(llhs) / np.sum(llhs)

        return normalized_llhs

    def __call__(self, action_sig, world, curr_state):
        goal_dist = [
            self.compute_likelihood(
                goal, world,
                curr_state)[self.action_sigs_pruned.index(action_sig)] *
            self.goal_prior[goal] for goal in self.goals
        ]

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