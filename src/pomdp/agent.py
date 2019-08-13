import random
import numpy as np
import itertools as it

class Env:
    def __init__(self,
                env_size,
                num_speakers,
                num_receivers,
                state_space,
                signal_space,
                init_ag_pos,
                init_obj_pos,
                is_no_pref
                ):
        # attributes
        self.env_size = env_size
        self.state_space = state_space
        self.signal_space = signal_space
        self.init_ag_pos = init_ag_pos
        self.init_obj_pos = init_obj_pos
        self.is_no_pref = is_no_pref

        self.num_speakers = num_speakers
        self.num_receivers = num_receivers
        self.num_agents = self.num_speakers + self.num_receivers
        self.num_objects = len(self.init_obj_pos)

        # contents - 
        # agents
        self.speakers = [Speaker(env=self, ID=i) for i in range(self.num_speakers)]
        self.receivers = [Receiver(env=self, ID=i) for i in range(self.num_receivers)]
        
        # order might matter (speaker -> receiver) - use index to fix the order
        self.agents = self.speakers + self.receivers
        self.speaker_index = 0
        self.receiver_index = 1

        # objects
        self.objects = [Object(ID=i) for i in range(self.num_objects)]
        
        # goal space is all objects + 1(whichever is fine)

        # target
        self.num_targets = 1 
        if self.is_no_pref:
            self.num_targets = len(self.objects)

        self.targets = None

        # initialization
        self.reset()
    
    def reset(self):
        # reset positions (+ belief if agents)
        for i, agent in enumerate(self.agents):
            agent.pos = self.init_ag_pos[i]
            agent.belief = [1/len(self.state_space) for _ in self.state_space]
        
        for i, obj in enumerate(self.objects):
            obj.pos = self.init_obj_pos[i]

        # random sample new target
        self.targets = random.sample(self.objects, self.num_targets)
        

    def transition(self, state, action):
        '''
            state: joint state of speaker and receiver: [[[s1x, s1y], ..., [snx, sny]],[[r1x, r1y], ..., [rnx, rny]]]
            action: joint action. same format as state. Speaker might have action ["<signal>"]

            return:
            next_state: same format as state
        '''
        speaker_actions = action[self.speaker_index]
        receiver_actions = action[self.receiver_index]
        def in_bound(new_pos):
            if 0 <= new_pos[0] < self.env_size[0] and 0 <= new_pos[1] < self.env_size[1]:
                return True
            else:
                return False

        next_state = state
        receiver_states = state[self.receiver_index]
        speaker_states = state[self.speaker_index]
        # receiver new state
        receiver_new_states = [np.add(receiver_states[i], receiver_actions[i]) for i in range(self.num_receivers)]
        for i in range(self.num_receivers):
            curr_action = receiver_actions[i]
            curr_state = receiver_states[i]
            new_state = list(np.add(curr_action, curr_state))
            if in_bound(new_state):
                next_state[self.receiver_index][i] = new_state

        # speaker new state
        for i in range(self.num_speakers):
            curr_action = speaker_actions[i]
            curr_state = speaker_states[i]
            # check if action is signal
            # if not isinstance(curr_action, str):
            if not isinstance(curr_action, int):
                new_state = list(np.add(curr_state, curr_action))
                if in_bound(new_state):
                    next_state[self.speaker_index][i] = new_state
        

        return next_state

    def reward(self, state, action, next_state):
        '''
            state, action, next_state same as transition
            return:
            reward : float
        '''
        action_cost = 1
        target_reward = 10
        signal_cost = 0

        reward = 0
        # action cost
        for agent in self.agents:
            if isinstance(agent, Receiver):
                if action[self.receiver_index][agent.ID] != [0, 0]:
                    reward -= action_cost
            else:
                sp_action = action[self.speaker_index][agent.ID]
                if isinstance(sp_action, int):
                    reward -= signal_cost
                elif sp_action != [0, 0]:
                    reward -= action_cost

        # target reward
        # TODO: if two target touched at the same time, currently getting double rewards
        all_next_state = next_state[self.speaker_index] + next_state[self.receiver_index]
        for n_s in all_next_state:
            for target_pos in [target.pos for target in self.targets]:
                if n_s == target_pos:
                    reward += target_reward

        return reward


class Agent:
    def __init__(self, env):
        self.env = env
        self.pos = None
        self.belief = None
        self.cardinal_actions = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]

    
    def obs(self, action, next_state):
        # check whether there is signal action
        speaker_actions = action[self.env.speaker_index]
        signals = [a for a in speaker_actions if isinstance(a, int)]

        # extract positions form state
        positions = next_state[:2]
        obs = positions
        obs.append(signals)

        return obs

    def belief_update(self):
        raise NotImplementedError


class Speaker(Agent):
    def __init__(self, env, ID):
        super().__init__(env)
        self.ID = ID
        self.signals = self.env.signal_space
        self.action_space = self.cardinal_actions + self.signals
    
    def belief_update(self):
        pass



class Receiver(Agent):
    def __init__(self, env, ID):
        super().__init__(env)
        self.env = env
        self.ID = ID
        self.action_space = self.cardinal_actions

    def belief_update(self):
        pass


class Object:
    def __init__(self, ID):
        self.pos = None


def test():
    env_size = [2, 2]
    num_speakers = 1
    num_receivers = 1
    initial_agent_pos = [[0, 0], [0, 0]]
    initial_battery_pos = [[2, 2],[2, 4]]
    num_objects = len(initial_battery_pos)
    # state_space = [[x, y, g] for x in range(env_size[0]) for y in range(env_size[1]) for g in goal_space]
    signal_space = list(range(num_speakers + num_receivers))
    goal_space = range(num_objects)
    latent_state_space = list(it.product(*[signal_space, goal_space]))
    positions = [list(pos) for pos in list(it.product(*[range(env_size[0]), range(env_size[1])]))]
    state_space = [list(s) for s in list(it.product(*[positions, latent_state_space]))]

    # obs_space = list(it.product(*[range(env_size[0]), range(env_size[1]), ["speaker", "receiver", "none"]]))

    # if the target is either one.
    is_no_pref = False
    env = Env(
            env_size=env_size,
            num_speakers=num_speakers, 
            num_receivers=num_receivers,
            state_space=state_space,
            signal_space = signal_space,
            init_ag_pos=initial_agent_pos,
            init_obj_pos=initial_battery_pos,
            is_no_pref=is_no_pref)


if __name__ == "__main__":
    test()
    
    