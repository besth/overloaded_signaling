import itertools as it
import numpy as np

from pomdp import POMDP

class BeliefReward:
    def __init__(self, pomdp):
        self.reward_func = pomdp.reward_func
        self.state_space = pomdp.state_space

    def __call__(self, belief, action):
        belief_reward = np.sum([belief(s) * self.reward_func(s, action) for s in self.state_space])




def one_step_lookahead(state, value_dict, pomdp, gamma):
    action_values = np.zeros_like(pomdp.action_space)
    for a_i, a in enumerate(pomdp.action_space):
        next_state = pomdp.transition_func(state, a)
        curr_value = pomdp.reward_func(state, a) + gamma * value_dict[next_state]
        action_values.append(curr_value)

    return action_values


class ValueIteration:
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, pomdp):
        value_dict = {state: 0 for state in pomdp.state_space}
        done = False
        while not done:
            prev_value_dict = dict(value_dict)
            for s_i, s in enumerate(pomdp.state_space):
                action_values = one_step_lookahead(s, value_dict, pomdp, self.gamma)
                value_dict[s] = max(action_values)

            # check terminal condition
            if all([abs(value_dict[state] - prev_value_dict[state]) for state in pomdp.state_space]) < self.epsilon:
                print("Value function converged.")
                break
            
        return value_dict



def test():
    env_size = [5, 10]
    goal_space = ["left", "right", "either"]
    receiver_action_space = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    speaker_action_space = receiver_action_space + ["speaker", "receiver"]
    action_space = list(it.product(*([speaker_action_space, receiver_action_space])))
    # state_space = [[x, y, g] for x in range(env_size[0]) for y in range(env_size[1]) for g in goal_space]
    state_space = list(it.product(*[range(env_size[0]), range(env_size[1]), goal_space]))
    obs_space = list(it.product(*[range(env_size[0]), range(env_size[1]), ["speaker", "receiver", "none"]]))

    agent_index_dict = {"speaker": 0, "receiver": 1}
    initial_agent_pos = [[0, 0], [0, 0]]
    initial_battery_pos_dict = {"left": [2, 2], "right": [2, 4]}
    target_name = "either"

    # make POMDP
    self.pomdp = POMDP(env_size=env_size,
                state_space=state_space,
                action_space=action_space,
                goal_space=goal_space,
                obs_space=obs_space,
                agent_index_dict=agent_index_dict,
                initial_agent_pos=initial_agent_pos,
                initial_battery_pos_dict=initial_battery_pos_dict,
                target_name=target_name
                )

if __name__ == "__main__":
    test()