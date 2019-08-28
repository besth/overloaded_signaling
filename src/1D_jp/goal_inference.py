import os
import numpy as np
import matplotlib.pyplot as plt
from util import SIGNAL_DICT, WORLD_DICT, GOAL_DICT, SIGNAL_DICT, WORLD_DICT, GOAL_DICT, GOAL_REWARD, SIGNAL_REW_DICT


class GoalInference:
    def __init__(self, beta):
        self.beta = beta

        self.num_signals = len(SIGNAL_DICT)
        self.num_worlds = len(WORLD_DICT)
        self.num_goals = len(GOAL_DICT)

        self.signals = list(range(self.num_signals))
        self.worlds = list(range(self.num_worlds))
        self.goals = list(range(self.num_goals))

        self.lexicon = self.create_lexicon()
        self.goal_prior = self.get_goal_prior()

    def get_goal_prior(self):
        return [0.23, 0.77]
        # return [1 / len(self.goals) for _ in self.goals]

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
        if signal != SIGNAL_DICT["get-A"] and signal != SIGNAL_DICT["get-B"]:
            return SIGNAL_REW_DICT[signal]
        else:
            return SIGNAL_REW_DICT[signal][world]

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


def plot(path="plot/"):
    worlds = list(WORLD_DICT.keys())
    worlds_ind = [WORLD_DICT[w] for w in worlds]

    signals = ["help"]
    signals_ind = [SIGNAL_DICT[s] for s in signals]

    betas = np.arange(0, 0.5, 0.1).round(2)
    # p_fars = np.zeros((len(signals), len(betas)))
    fig, axs = plt.subplots(len(signals), len(betas), sharey=True)

    p_far_gt = [0.49, 0.23]
    error = [0.29, 0.34]

    for b_i in range(len(betas)):
        GI = GoalInference(beta=betas[b_i])
        for s_i in range(len(signals)):
            p_far = [(GI(SIGNAL_DICT[signals[s_i]], w_ind))[0]
                     for w_ind in worlds_ind]

            bar_width = 0.25
            x1 = np.arange(len(p_far))
            x2 = [x + bar_width for x in x1]

            # curr = axs[s_i, b_i]
            curr = axs[b_i]
            line_1 = curr.bar(x1,
                              p_far_gt,
                              color=[(0.2, 0.4, 0.6, 0.9),
                                     (0.6, 0.4, 0.2, 0.9)],
                              yerr=error,
                              width=bar_width)
            line_2 = curr.bar(x2,
                              p_far,
                              color=[(0.2, 0.4, 0.6, 0.5),
                                     (0.6, 0.4, 0.2, 0.5)],
                              width=bar_width)

            curr.set_title("beta= {}".format(betas[b_i]), fontsize=12)
            # curr.set_xticks([0, 1], ["HF", "HT"])
            if b_i == 0:
                curr.set(ylabel=signals[s_i])
                curr.yaxis.get_label().set_fontsize(12)

    plt.figlegend((line_1, line_2), ("Human result", "Model prediction"),
                  loc="upper left",
                  fontsize=12)
    plt.setp(axs, xticks=[0, 1], xticklabels=['HF', 'HT'])
    if not os.path.exists(path):
        os.mkdir(path)

    title = "goal_dist.png"
    plt.tight_layout()
    plt.savefig(os.path.join(path, title))
    plt.show()


def plot_paper_results():
    plt.figure()
    x = ['F', 'T']
    y = [0.49, 0.23]
    error = [0.29, 0.34]

    plt.bar(x, y, yerr=error, color=["blue", "red"])
    plt.show()


def test():
    GI = GoalInference(beta=0.2)
    signal = SIGNAL_DICT["help-Any"]
    world = WORLD_DICT["hands-tied"]
    print(world)
    goal_dist = GI(signal, world)
    print(goal_dist)


if __name__ == "__main__":
    plot()
    # test()
    # plot_paper_results()
