import copy
import numpy as np
from util import StateEncoding


def softmax(values: list, temp=1.0):
    reweighted = np.asarray(values) / temp
    exp_values = np.exp(reweighted)
    sm_values = exp_values / np.sum(exp_values)

    return sm_values


class ValueIteration:
    def __init__(self, gamma, epsilon, tau, env):
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.env = env

    def one_step_lookahead(self, s_values, s):
        q_values = []
        for a in self.env.action_space:
            next_state = self.env.transition(s, a)
            reward = self.env.reward(s, a)

            next_state_ind = self.env.get_state_index(next_state)

            q_values.append(reward + self.gamma * s_values[next_state_ind])

        return q_values

    def select_action(self, v_list, s, method="softmax"):
        q_values = self.one_step_lookahead(v_list, s)

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

    def __call__(self, goal):
        self.env.set_goal(goal)

        # initialize value for all states(including terminal states)
        s_values = np.zeros_like(self.env.state_space_encoding)
        total_states = np.size(s_values)
        curr_iter = 0
        while True:
            print("current iteration:", curr_iter)
            error = 0

            for s in self.env.state_space:
                s_ind = self.env.get_state_index(s)
                # proceed if non-terminal and valid state:
                if self.env.state_space_encoding[
                        s_ind] == StateEncoding.GOOD.value:
                    curr_v = s_values[s_ind]
                    q_values = self.one_step_lookahead(s_values, s)
                    max_q = max(q_values)
                    s_values[s_ind] = max_q
                    error = max(error, abs(curr_v - max_q))

            print("error", error)
            if error < self.epsilon:
                break
            curr_iter += 1

        return s_values