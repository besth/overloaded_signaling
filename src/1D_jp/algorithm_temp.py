import random

class ValueIteration:
    def __init__(self, gamma, epsilon, env):
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def select_action(self, v_dict, curr_s):
        q_values = [self.env.reward(curr_s, a) + self.gamma * v_dict[self.env.transition(curr_s, a)] for a in self.env.action_space]
        print(q_values, max(q_values))
        
        # find all actions corres. to the optimal value
        actions = [index for index, v in enumerate(q_values) if v == max(q_values)]

        return self.env.action_space[random.sample(actions, 1)[0]]

    def __call__(self):
        v_dict = {s: 0 for s in self.env.state_space}

        while True:
            error = 0
            for s in self.env.state_space_no_terminal:
                v = v_dict[s]

                q_values = [self.env.reward(s, a) + self.gamma * v_dict[self.env.transition(s, a)] for a in self.env.action_space]
                v_dict[s] = max(q_values)
                error = max(error, abs(v - v_dict[s]))
            
            if error < self.epsilon:
                break

        return v_dict




