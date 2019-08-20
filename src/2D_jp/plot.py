import os
import numpy as np
import matplotlib.pyplot as plt

from util import GOAL_SPACE, ENV_TYPES

num_goals = len(GOAL_SPACE)
num_envs = len(ENV_TYPES)


def plot_avg_steps():
    # create empty storage
    num_steps = np.empty((num_envs, num_goals))
    goals, envs, opts, avgs, errs = [], [], [], [], []
    data = {}

    # read from file
    with open("data/num_steps.txt") as f:
        for line in f:
            # print(line.split()[:2])
            goal_ind, env_ind = [int(x) for x in line.split()[:2]]
            opt, avg, err = [float(x) for x in line.split()[2:]]
            data[(goal_ind, env_ind)] = (opt, avg, err)
            goals.append(goal_ind)
            envs.append(env_ind)
            opts.append(opt)
            avgs.append(avg)
            errs.append(err)

    # plot
    fig, axs = plt.subplots(num_goals, num_envs)
    fig.set_size_inches(16.5, 12.5)
    fig.suptitle(
        "Average number of steps to terminal state (softmax temp = {})".format(
            0.2),
        fontweight="bold",
        y=1)
    bar_width = 0.25
    goal_name = ["far", "near", "either"]
    for g_i in range(num_goals):
        for e_i in range(num_envs):
            perf = data[(g_i, e_i)]
            curr = axs[g_i, e_i]
            x = np.arange(2)
            steps = [perf[0], perf[1]]

            curr.bar(x, steps, yerr=[0, perf[2]], width=bar_width)
            for i, v in enumerate(steps):
                curr.text(i - 0.1, v + 0.1, str(v), fontweight='bold')

            if g_i == 0:
                curr.set_title("Env: {}".format(ENV_TYPES[e_i]), fontsize=10)
            if e_i == 0:
                curr.set(ylabel="goal: {}".format(goal_name[g_i]))

    plt.setp(axs, xticks=[0, 1], xticklabels=['optimal', 'model'])
    path = "plot/"
    if not os.path.exists(path):
        os.mkdir(path)

    title = "avg_num_steps.png"
    plt.tight_layout()
    plt.savefig(os.path.join(path, title))
    plt.show()


plot_avg_steps()