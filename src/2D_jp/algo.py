import copy
import numpy as np
from util import StateEncoding, GOAL_SPACE


def softmax(values: list, temp=1.0):
    reweighted = np.asarray(values) / temp
    exp_values = np.exp(reweighted)
    sm_values = exp_values / np.sum(exp_values)

    return sm_values


def one_step_lookahead(env, s_values, curr_s, gamma):
    q_values = []
    for a in env.action_space:
        next_state = env.transition(curr_s, a)
        reward = env.reward(curr_s, a)

        next_state_ind = env.get_state_index(next_state)

        q_values.append(reward + gamma * s_values[next_state_ind])

    return q_values


class ValueIteration:
    def __init__(self, gamma, epsilon, tau, env):
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.env = env

    def select_action(self, v_list, s, method="softmax"):
        q_values = one_step_lookahead(self.env, v_list, s, self.gamma)

        if method == "softmax":
            q_probs = softmax(q_values, self.tau)
            action_index = np.random.choice(len(q_probs), p=q_probs)
        elif method == "max":
            a_indices = [
                i for i in range(len(q_values))
                if q_values[i] == np.max(q_values)
            ]
            action_index = np.random.choice(a_indices, 1)[0]

        action = self.env.action_space[action_index]

        return action

    def __call__(self, goal=None, path=None):
        if goal is not None:
            self.env.set_goal(goal)

        # initialize value for all states(including terminal states)
        s_values = np.zeros_like(self.env.state_space_encoding)
        total_states = np.size(s_values)
        curr_iter = 0
        while True:
            print("current iteration:", curr_iter, ", current goal:",
                  self.env.goal)
            error = 0

            for s in self.env.state_space:
                s_ind = self.env.get_state_index(s)
                # proceed if non-terminal and valid state:
                if self.env.state_space_encoding[
                        s_ind] == StateEncoding.GOOD.value:
                    curr_v = s_values[s_ind]
                    q_values = one_step_lookahead(self.env, s_values, s,
                                                  self.gamma)
                    max_q = max(q_values)
                    s_values[s_ind] = max_q
                    error = max(error, abs(curr_v - max_q))

            print("error", error)
            if error < self.epsilon:
                break
            curr_iter += 1

        # save results for multi-threading
        if path is None:
            raise AssertionError

        goal_ind = GOAL_SPACE.index(self.env.goal)
        env_type = self.env.env_type
        np.save(path, s_values)

        return s_values