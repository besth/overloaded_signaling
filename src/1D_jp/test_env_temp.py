import unittest
from ddt import data, unpack, ddt
from env import Env_1D, Env_1D_single


@ddt
class testEnv(unittest.TestCase):
    def setUp(self):
        self.env_length = 10
        self.env = Env_1D(self.env_length, [None])

    @data(
        ((1, 5), (1, -1), (2, 4)),
        ((0, 8), (-1, 0), (0, 8)),
        ((5, 9), (1, 1), (6, 9))
    )
    @unpack
    def test_transition(self, state, action, gt_next_state):
        next_state = self.env.transition(state, action)
        self.assertEqual(next_state, gt_next_state)

    @data(
        ((1, 1), (0, 0), [None], 0),
        ((4, 4), (1, -1), [0], -2),
        ((1, 1), (-1, 1), [0], 8),
        ((1, 9), (-1, -1), [0, 1], 8),
        ((1, 9), (-1, -1), [None], 8),
        ((1, 9), (-1, -1), [0], -2)
    )   
    @unpack 
    def test_reward(self, state, action, goal, gt_reward):
        self.env = Env_1D(self.env_length, goal)
        reward = self.env.reward(state, action)
        self.assertEqual(reward, gt_reward)

    @data(
        ((6, -1, -1)),
        ((6, 0, 0)),
        ((6, 1, -1))
    )
    @unpack
    def test_reward_single(self, state, action, gt_reward):
        self.env = Env_1D_single(7, 0)
        reward = self.env.reward(state, action)
        self.assertEqual(reward, gt_reward)



if __name__ == "__main__":
    unittest.main()