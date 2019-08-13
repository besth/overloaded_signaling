import numpy as np
from util import SIGNAL_DICT, WORLD_DICT, GOAL_DICT, SIGNAL_DICT, WORLD_DICT, GOAL_DICT, GOAL_REWARD, SIGNAL_REW_DICT


class GoalInference:
    def __init__(self, beta):
        self.beta = beta

        self.signals = list(range(4))
        self.worlds = list(range(2))
        self.goals = list(range(2))

        self.lexicon = self.create_lexicon()
        self.goal_prior = self.get_goal_prior()

    def get_goal_prior(self):
        # return [0.23, 0.77]
        return [1 / len(self.goals) for _ in self.goals]

    def create_lexicon(self):
        lex = np.zeros((len(self.signals), len(self.goals)))

        # sig = "help" --> [0.5, 0.5]
        lex[SIGNAL_DICT["help"]][:] = 1 / len(self.goals)

        # sig = "help-A" --> [1, 0]
        lex[SIGNAL_DICT["help-A"]][GOAL_DICT['A']] = 1
        lex[SIGNAL_DICT["help-A"]][GOAL_DICT['B']] = 0

        # sig = "help-B" --> [0, 1]
        lex[SIGNAL_DICT["help-B"]][GOAL_DICT['A']] = 0
        lex[SIGNAL_DICT["help-B"]][GOAL_DICT['B']] = 1

        # sig = "help-Any" --> [0.5, 0.5]
        lex[SIGNAL_DICT["help-Any"]][:] = 1 / len(self.goals)

        # sig = "get-A" --> [1, 0]
        lex[SIGNAL_DICT["get-A"]][GOAL_DICT['A']] = 1
        lex[SIGNAL_DICT["get-A"]][GOAL_DICT['B']] = 0

        # sig = "get-B" --> [0, 1]
        lex[SIGNAL_DICT["get-B"]][GOAL_DICT['A']] = 0
        lex[SIGNAL_DICT["get-B"]][GOAL_DICT['B']] = 1

        return lex

    def reward_signal(self, signal, world):
        if signal != "get-A" and signal != "get-B":
            return SIGNAL_REW_DICT[signal]
        else:
            return SIGNAL_REW_DICT[signal][world]
        # if world == WORLD_DICT["hands-free"]:
        #     if signal == SIGNAL_DICT["help"]:
        #         return -1
        #     elif signal == SIGNAL_DICT["get-A"]:
        #         return -2
        #     elif signal == SIGNAL_DICT["get-B"]:
        #         return -2
        #     elif signal == SIGNAL_DICT["get-Any"]:
        #         return -3
        # elif world == WORLD_DICT["hands-tied"]:
        #     if signal == SIGNAL_DICT["help"]:
        #         return -1
        #     elif signal == SIGNAL_DICT["get-A"]:
        #         return -2
        #     elif signal == SIGNAL_DICT["get-B"]:
        #         return -2
        #     elif signal == SIGNAL_DICT["get-Any"]:
        #         return -3

    def reward_goal(self, goal, gt_goal):
        return (goal == gt_goal) * GOAL_REWARD

    def compute_likelihood(self, signal, goal, world):
        goal_utility = np.sum([
            self.lexicon[signal][g] * self.reward_goal(g, goal)
            for g in self.goals
        ])
        utility = self.reward_signal(signal, world) + goal_utility

        llh = np.exp(self.beta * utility)
        return llh

    def normalized_likelihood(self, goal, world):
        unnormalized = [
            self.compute_likelihood(signal, goal, world)
            for signal in self.signals
        ]

        normalized = np.array(unnormalized) / np.sum(unnormalized)

        return normalized

    def __call__(self, signal, world):
        goal_dist = [
            self.normalized_likelihood(goal, world)[signal] *
            self.goal_prior[goal] for goal in self.goals
        ]

        # normalize
        goal_dist = np.array(goal_dist) / np.sum(goal_dist)

        return list(goal_dist)


def test():
    GI = GoalInference(beta=0.1)
    signal = SIGNAL_DICT["help"]
    world = WORLD_DICT["hands-free"]

    goal_dist = GI(signal, world)
    print(goal_dist)


if __name__ == "__main__":
    test()
