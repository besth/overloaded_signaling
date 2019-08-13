import unittest
from ddt import data, unpack, ddt

from env import Env_1D
from algo import ValueIteration as VI, softmax


@ddt
class testAlgo(unittest.TestCase):
    def setUp(self):
        env_length = 7
        self.env = Env_1D(env_length)

        gamma = 0.99
        epsilon = 0.05
        self.VI = VI(gamma, epsilon, self.env)

    def test_select_action(self):
        # TODO: hand-craft value dict. maybe reduce the env size

        pass

    def test_vi(self, gt_v_dict):
        pass

    @data(
        ([10, 20, 20, 10, 40, 50], [])
    )
    @unpack
    def test_sm(self, values, gt_probs):
