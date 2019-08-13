import sys
sys.path.append('..')

import numpy as np
import itertools as it
import unittest
from ddt import ddt, data, unpack

from pomdp import POMDP


@ddt
class TestPOMDP(unittest.TestCase):
    def setUp(self):
        env_size = [5, 10]
        goal_space = ["left", "right", "either"]
        receiver_action_space = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
        speaker_action_space = receiver_action_space + ["speaker", "receiver"]
        action_space = list(it.product(*([speaker_action_space, receiver_action_space])))
        # state_space = [[x, y, g] for x in range(env_size[0]) for y in range(env_size[1]) for g in goal_space]
        state_space = list(it.product(*[range(env_size[0]), range(env_size[1]), goal_space]))
        obs_space = list(it.product(*[range(env_size[0]), range(env_size[1]), ["speaker", "receiver", "none"]]))

        agent_index_dict = {"speaker": 0, "receiver": 1}
        agent_pos = [[0, 0], [0, 0]]
        battery_pos_dict = {"left": [2, 2], "right": [2, 4]}
        target_name = "either"

        # make POMDP
        self.pomdp = POMDP(env_size=env_size,
                    state_space=state_space,
                    action_space=action_space,
                    goal_space=goal_space,
                    obs_space=obs_space,
                    agent_index_dict=agent_index_dict,
                    agent_pos=agent_pos,
                    battery_pos_dict=battery_pos_dict,
                    target_name=target_name
                    )

    @data(([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[0, 0], [1, 1], "left"], "left", 0),
          ([[0, 0], [1, 1], "left"], [[1, 0], [0, 0]], [[0, 0], [1, 1], "left"], "left", -1),
          ([[0, 0], [1, 1], "left"], [[0, 0], [1, 0]], [[0, 0], [1, 1], "left"], "left", -1),
          ([[0, 0], [1, 1], "left"], [[1, 1], [1, -1]], [[0, 0], [1, 1], "left"], "left", -2),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[2, 2], [1, 1], "left"], "left", 10),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[1, 1], [2, 4], "left"], "left", 0),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[1, 1], [2, 4], "right"], "right", 10),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[2, 2], [1, 4], "right"], "right", 0),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[1, 1], [2, 4], "left"], "either", 10),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[2, 2], [1, 1], "left"], "either", 10),
          ([[0, 0], [1, 1], "left"], [[0, 0], [0, 0]], [[2, 2], [2, 4], "left"], "either", 10),
          ([[0, 0], [1, 1], "left"], ["speaker", [0, 0]], [[1, 1], [1, 1], "left"], "either", 0),
          ([[0, 0], [1, 1], "left"], ["speaker", [0, 1]], [[1, 1], [1, 1], "left"], "either", -1)
         )
    @unpack
    def test_reward(self, state, action, next_state, target_name, gt_reward):
        self.pomdp.target_name = target_name

        reward = self.pomdp.reward_func(state, action, next_state)
        self.assertEqual(reward, gt_reward)

    @data(
        ([[1, 1], [2, 2], "left"], [[1, 0], [-1, 1]], [[2, 1], [1, 3], "left"]),
        ([[1, 1], [2, 2], "left"], [[-2, 0], [-1, 1]], [[1, 1], [1, 3], "left"]),
        ([[4, 4], [2, 2], "left"], [[1, 0], [-1, 1]], [[4, 4], [1, 3], "left"]),
        ([[4, 9], [2, 2], "left"], [[1, 1], [-1, 1]], [[4, 9], [1, 3], "left"]),
        ([[2, 3], [2, 2], "left"], ["speaker", [-1, 1]], [[2, 3], [1, 3], "left"]),
        ([[3, 6], [2, 2], "left"], ["either", [-1, 1]], [[3, 6], [1, 3], "left"])
    )
    @unpack
    def test_transition(self, state, action, gt_next_state):
        next_state = self.pomdp.transition_func(state, action)
        self.assertListEqual(next_state, gt_next_state)

    @data(
        ([[3, 6], [2, 2], "left"], [[1, 1], [-1, 1]], [[3, 6], [2, 2], "left", "none"]),
        ([[3, 6], [2, 2], "left"], ["speaker", [-1, 1]], [[3, 6], [2, 2], "left", "speaker"]),
        ([[3, 6], [2, 2], "left"], ["receiver", [-1, 1]], [[3, 6], [2, 2], "left", "receiver"]),
    )
    @unpack
    def test_obs(self, new_state, action, gt_obs):
        obs = self.pomdp.obs_func(new_state, action)
        self.assertListEqual(obs, gt_obs)


if __name__=="__main__":
    unittest.main()

