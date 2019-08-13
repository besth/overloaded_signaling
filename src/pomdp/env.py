import numpy as np
import itertools as it

class Env:
    def __init__(self, 
                env_size:list, 
                state_space:list, 
                action_space:list, 
                goal_space:list, 
                obs_space:list,
                agent_index_dict:dict,
                initial_agent_pos:list, 
                initial_battery_pos_dict:dict, 
                target_name:str
                ):
        # parameters for Env
        self.env_size = env_size
        self.state_space = state_space
        self.action_space = action_space
        self.goal_space = goal_space
        self.obs_space = obs_space

        # specific setup
        self.battery_pos_dict = initial_battery_pos_dict
        self.agent_pos = initial_agent_pos
        self.target_name = target_name

        # TODO:assign index to agents
        self.speaker_index = agent_index_dict["speaker"]
        self.receiver_index = agent_index_dict["receiver"]

        # TODO: initialize agent objects
        
    
    def transition_func(self, state, action):
        speaker_action = action[self.speaker_index]
        receiver_action = action[self.receiver_index]
        def in_bound(new_pos):
            if 0 <= new_pos[0] < self.env_size[0] and 0 <= new_pos[1] < self.env_size[1]:
                return True
            else:
                return False

        next_state = state
        receiver_state = state[self.receiver_index]
        speaker_state = state[self.speaker_index]
        # receiver new state
        receiver_new_state = list(np.add(receiver_state, receiver_action))
        if in_bound(receiver_new_state):
            next_state[self.receiver_index] = receiver_new_state

        # speaker new state
        if isinstance(speaker_action, str):
            return next_state
        speaker_new_state = list(np.add(speaker_state, speaker_action))
        if in_bound(speaker_new_state):
            next_state[self.speaker_index] = speaker_new_state

        return next_state


    def reward_func(self, state, action, next_state):
        action_cost = 1
        target_reward = 10
        
        reward = 0
        # action cost
        speaker_action = action[self.speaker_index]
        receiver_action = action[self.receiver_index]
        if speaker_action != [0, 0] and not isinstance(speaker_action, str):
            reward -= action_cost
        if receiver_action != [0, 0]:
            reward -= action_cost

        # reach target reward
        if self.target_name == "either":
            target_pos = list(self.initial_battery_pos_dict.values())
        else:
            target_pos = [self.initial_battery_pos_dict[self.target_name]]
        if set([tuple(s) for s in next_state[:2]]) & set([tuple(s) for s in target_pos]):
            reward += target_reward

        return reward
    
    def obs_func(self, new_state, action):
        speaker_state = new_state[self.speaker_index]
        receiver_state = new_state[self.receiver_index]

        speaker_action = action[self.speaker_index]

        if isinstance(speaker_action, str):
            obs = new_state + [speaker_action]
        else:
            obs = new_state + ["none"]


        return obs


def main():
    env_size = [2, 3]
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



    # make Env
    pomdp = Env(env_size=env_size,
                  state_space=state_space,
                  action_space=action_space,
                  goal_space=goal_space,
                  obs_space=obs_space,
                  agent_index_dict=agent_index_dict,
                  initial_agent_pos=initial_agent_pos,
                  initial_battery_pos_dict=initial_battery_pos_dict,
                  target_name=target_name
                  )

    state = [[1,1], [1,2], "left"]
    action = [[0,0], [0,0]]
    next_state = [[2,0], [1,4], target_name]

    reward = pomdp.reward_func(state, action, next_state)
    


if __name__ == "__main__":
    main()