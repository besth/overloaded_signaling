import sys
sys.path.append('..')

import itertools as it
import unittest
from ddt import ddt, data, unpack

from agent import Env 

@ddt
class TestEnv(unittest.TestCase):
    def setUp(self):
        env_size = [5, 5]
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
        self.env = Env(
                env_size=env_size,
                num_speakers=num_speakers, 
                num_receivers=num_receivers,
                state_space=state_space,
                signal_space = signal_space,
                init_ag_pos=initial_agent_pos,
                init_obj_pos=initial_battery_pos,
                is_no_pref=is_no_pref)


    # to test multiple sp/re - need to redefine env, modifying initial pos list
    @data(
        ([[[1, 1]], [[2, 2]], [0, 0]], [[[1, 0]], [[-1, 1]]], [[[2, 1]], [[1, 3]], [0, 0]]),
        ([[[1, 1]], [[2, 2]], [0, 0]], [[[-2, 0]], [[-1, 1]]], [[[1, 1]], [[1, 3]], [0, 0]]),
        ([[[4, 4]], [[2, 2]], [0, 0]], [[[1, 0]], [[-1, 1]]], [[[4, 4]], [[1, 3]], [0, 0]]),
        ([[[4, 9]], [[2, 2]], [0, 0]], [[[1, 1]], [[-1, 1]]], [[[4, 9]], [[1, 3]], [0, 0]]),
        ([[[2, 3]], [[2, 2]], [0, 0]], [[0], [[-1, 1]]], [[[2, 3]], [[1, 3]], [0, 0]]),
        ([[[3, 6]], [[2, 2]], [0, 0]], [[1], [[-1, 1]]], [[[3, 6]], [[1, 3]], [0, 0]]),
        ([[[1, 1], [4, 4]], [[2, 2], [0, 0]]], [[0, [2, 2]], [[-1, 1], [-1, -1]]], [[[1, 1], [4, 4]], [[1, 3], [0, 0]]])
    )
    @unpack
    def test_transition(self, state, action, gt_next_state):
        next_state = self.env.transition(state, action)
        for i in range(len(next_state)):
            self.assertListEqual(next_state[i], gt_next_state[i])


    # to test multiple sp/re - need to redefine env, modifying initial pos list
    # TODO: not tested for signal cost != 0
    @data(
        ([[[1, 1]], [[2, 2]], [0, 1]], [[[1, 0]], [[-1, 1]]], [[[2, 1]], [[2, 4]], [0, 0]], -2),
        ([[[1, 1]], [[2, 2]], [0, 0]], [[[-2, 0]], [[-1, 1]]], [[[2, 2]], [[2, 2]], [0, 0]], 18),
        ([[[4, 4]], [[2, 2]], [0, 1]], [[[1, 0]], [[-1, 1]]], [[[4, 4]], [[1, 3]], [0, 0]], -2),
        ([[[4, 9]], [[2, 2]], [0, 0]], [[[1, 1]], [[-1, 1]]], [[[2, 2]], [[1, 3]], [0, 0]], 8),
        ([[[2, 3]], [[2, 2]], [0, 1]], [[0], [[-1, 1]]], [[[2, 3]], [[1, 3]], [0, 0]], -1),
        ([[[3, 6]], [[2, 2]], [0, 0]], [[1], [[-1, 1]]], [[[3, 6]], [[2, 2]], [0, 0]], 9),
        # ([[[1, 1], [4, 4]], [[2, 2], [0, 1]], [0, 0]], [[0, [2, 2]], [[-1, 1], [-1, -1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]], [0, 0]], 37)
    )
    @unpack
    def test_reward(self, state, action, next_state, gt_reward):
        # only depending on next state. So the s,a->s' here might not be the true transition.
        self.env.targets = [self.env.objects[0]]

        reward = self.env.reward(state, action, next_state)
        self.assertEqual(reward, gt_reward)


    # to test multiple sp/re - need to redefine env, modifying initial pos list
    @data(
        ([[[1, 0]], [[-1, 1]]], [[[2, 1]], [[2, 4]], [0, 0]], [[[2, 1]], [[2, 4]], []]),
        ([[[-2, 0]], [[-1, 1]]], [[[2, 2]], [[2, 2]], [0, 0]], [[[2, 2]], [[2, 2]], []]),
        ([[[1, 0]], [[-1, 1]]], [[[4, 4]], [[1, 3]], [0, 0]], [[[4, 4]], [[1, 3]], []]),
        ([[[1, 1]], [[-1, 1]]], [[[2, 2]], [[1, 3]], [0, 0]], [[[2, 2]], [[1, 3]], []]),
        ([[0], [[-1, 1]]], [[[2, 3]], [[1, 3]], [0, 0]], [[[2, 3]], [[1, 3]], [0]]),
        ([[1], [[-1, 1]]], [[[3, 6]], [[2, 2]], [0, 0]], [[[3, 6]], [[2, 2]], [1]]),
        #[0, 1]], [0, 0]], [[0, [2, 2]], [[-1, 1], [-1, -1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]], [0, 0]], 37)
    )
    @unpack
    def test_obs(self, action, next_state, gt_obs):
        agent = self.env.agents[0]

        obs = agent.obs(action, next_state)
        for i in range(len(obs)):
            self.assertListEqual(obs[i], gt_obs[i])

    


    

    


if __name__=="__main__":
    unittest.main()



