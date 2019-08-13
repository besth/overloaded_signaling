import numpy as np
import itertools as it
import matplotlib.pyplot as plt


def compute_p_a_signal(signal):
    signals = ["i", "you"]
    signal_dict = {"i": 0, "you": 1}

    # Global variable
    ALPHA = 1

    FLAG_EITHER = False
    FLAG_HANDS_FREE = False

    # goal/desire
    ind_near = 0
    ind_far = 1
    ind_either = 2
    goals = ["near", "far"]
    if FLAG_EITHER:
        goals = ["near", "far", "either"]

    # joint action
    targets = goals[:2]
    action_space = list(it.product(
        signals, targets))  # [(i, near), (i, far), (you, near), (you, far)]
    action_costs = [-1, -10, -5, -5]
    if not FLAG_HANDS_FREE:
        action_costs = [-500, -500, -5, -5]

    #  action         goal=near    goal=far
    # (i, near)         9           -1
    # (i, far)          -10         0
    # (you, near)       5           -5
    # (you, far)        -5          5
    util_mat = np.ndarray((len(action_space), (len(goals))))
    for row_i in range(len(util_mat)):
        util_mat[row_i:] = action_costs[row_i]

    reward = +10
    for ind in [0, 2]:
        util_mat[ind][ind_near] += reward
        util_mat[ind + 1][ind_far] += reward

    if FLAG_EITHER:
        util_mat[:, 2] += 10

    # P(g): [p(near), p(far)]
    goal_priors = [0.77, 0.23]
    # goal_priors = [0.5, 0.5]
    if FLAG_EITHER:
        goal_priors = [0.3, 0.3, 0.1]

    # P(a|g) ~ exp(a * U)
    p_a_g = np.exp(ALPHA * util_mat)
    p_a_g_normed = p_a_g / np.sum(p_a_g, axis=0)

    # P(sig|a, g)
    p_sig_a_g = np.ones((len(signals), len(action_space), len(goals)))
    # when signal = 'you'
    p_sig_a_g[signal_dict["you"]][:2][:] = 0
    # when signal = 'i'
    p_sig_a_g[signal_dict["i"]][2:][:] = 0

    # P(a, g|signal)
    p_a_g_sig = np.zeros_like(p_sig_a_g)
    for i in signal_dict.values():
        p_a_g_sig[i] = p_sig_a_g[i] * p_a_g_normed * goal_priors

    # P(a | signal)
    p_a_sig = np.zeros((len(signals), len(action_space)))
    for i, p in enumerate(p_a_g_sig):
        p_a_sig[i] = np.sum(p_a_g_sig[i], axis=1)

    p_a_sig_normed = p_a_sig / np.sum(p_a_sig, axis=1)[:, np.newaxis]

    return p_a_sig_normed[signal_dict[signal]]


# def plot_dist(p_dist):
#     plt.figure(figsize=(20, 10))
#     x = range(len(action_space))
#     plt.xlabel("joint action")

#     plt.subplot(121)
#     plt.xticks(x, action_space)
#     plt.title("P(a | signal='i')")
#     plt.bar(x, p_dist[ind_i], width=0.3)

#     plt.subplot(122)
#     plt.xticks(x, action_space)
#     plt.title("P(a | signal='you')")
#     plt.bar(x, p_dist[ind_you], width=0.3)

#     plt.show()

print(compute_p_a_signal("you"))
# plot_dist(p_a_sig_normed)
