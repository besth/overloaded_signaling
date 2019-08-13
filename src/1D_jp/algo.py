import click
import numpy as np

from env import Env

# np.random.seed(0)


def softmax(values: list):
    exp_values = np.exp(values)

    sm_values = exp_values / np.sum(exp_values)
    # print(exp_values, np.sum(exp_values), sm_values, np.sum(sm_values))

    return sm_values


class ValueIteration:
    def __init__(self, gamma, epsilon, env):
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def one_step_lookahead(self, v_list, s):
        q_values = []
        for a in self.env.action_space:
            next_state = self.env.transition(s, a)
            next_state_ind = self.env.state_space.index(next_state)
            reward = self.env.reward(s, a)
            q_values.append(reward + self.gamma * v_list[next_state_ind])

        return q_values

    def select_action(self, v_list, s, method="softmax"):
        q_values = self.one_step_lookahead(v_list, s)
        if method == "softmax":
            q_probs = softmax(q_values)
            # print(self.env.action_space)
            # print(q_probs)
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
        v_list = [0 for s in self.env.state_space]
        while True:
            error = 0

            for i, s in enumerate(self.env.state_space):
                if s in self.env.state_space_no_terminal:
                    curr_v = v_list[i]
                    q_values = self.one_step_lookahead(v_list, s)
                    v_list[i] = max(q_values)
                    error = max(error, abs(curr_v - v_list[i]))

            if error < self.epsilon:
                break

        return v_list


class InferGoal:
    def __init__(self, beta, vi, env):
        self.beta = beta
        self.vi = vi
        self.env = env
        self.goal_space = self.env.goal_space

        # print(self.goal_space)
        self.v_tables = [self.vi(goal) for goal in self.goal_space]
        # print(self.v_tables)
        # print(len(self.v_tables))
        # print("test", self.env.transition((1, 7, set()), (-1, 0)))

    def goal_prior(self):
        return [1 / len(self.goal_space) for _ in range(len(self.goal_space))]

    def goal_llh_one_step(self, s, next_s, goal):
        res = 0
        # find action values by one step lookahead.
        action_values = self.vi.one_step_lookahead(
            self.v_tables[self.goal_space.index(goal)], s)

        # print("test2", s, self.env.transition(s, (-1, 0)))
        # print(goal, self.env.action_space, action_values)
        for i, a in enumerate(self.env.action_space):
            curr_a_value = action_values[i]
            res += (self.env.transition(s, a) == next_s) * np.exp(
                self.beta * curr_a_value)

        return res

    def goal_llh_seq(self, s_seq, goal):
        # print(s_seq)
        self.env.set_goal(goal)
        res = 1
        for i in range(len(s_seq) - 1):
            s = s_seq[i]
            next_s = s_seq[i + 1]
            res *= self.goal_llh_one_step(s, next_s, goal)

        # print(goal, res)
        return res

    def __call__(self, s_seq):
        g_dist = [
            self.goal_llh_seq(s_seq, goal) * self.goal_prior()[i]
            for i, goal in enumerate(self.goal_space)
        ]

        # Normalize
        g_dist = list(np.asarray(g_dist) / np.sum(g_dist))
        return g_dist


ENV_TYPE = [0, 1, None]


@click.command()
@click.option('--goal', default=0, help='Which goal: [0], [1], [0, 1], []')
@click.option('--inferred-goal',
              default=0,
              help='Which goal: {0:[0], 1, 2, 3}')
@click.option('--env-type', default=2, help='Which goal: [0], [1], [0, 1], []')
def test(goal, inferred_goal, env_type):
    goal_space = [{0}, {1}, set()]
    beta = 0.1
    env_length = 7
    env = Env(env_length, goal=goal_space[goal], env_type=ENV_TYPE[env_type])

    gamma = 0.9
    epsilon = 0.0001
    vi_jp = ValueIteration(gamma=gamma, epsilon=epsilon, env=env)

    infer_goal = InferGoal(beta, vi_jp, env)

    # testing state_seq
    state_seq = [(3, 6, set()), (3, 5, {1})]

    g_dist = infer_goal(state_seq)
    print(g_dist)


if __name__ == "__main__":
    test()