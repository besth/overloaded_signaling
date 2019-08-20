import unittest
from ddt import ddt, data, unpack

from env import Env2D
from util import GOAL_SPACE, GOAL_REWARD, ACTION_COST, PICKUP_COST


@ddt
class testEnv(unittest.TestCase):
    def setUp(self):
        env_size = (2, 8)
        env_type = 1
        goal_ind = 0
        goal = GOAL_SPACE[goal_ind]
        obj_poss = [(0, 2), (1, 6)]

        # terminal_pos = [env_size[i] - 1 for i in range(len(env_size))]
        terminal_pos = (1, 7)
        self.env = Env2D(env_size=env_size,
                         env_type=env_type,
                         goal=goal,
                         obj_poss=obj_poss,
                         terminal_pos=terminal_pos)

    @data(
        # (((0, 0), (1, 7), set(), {1}), ((-1, 0), (1, 0)), ((0, 0), (1, 7), set(), {1})),
        # (((0, 0), (1, 7), set(), {0, 1}), ((0, -1), (0, 1)), ((0, 0), (1, 7), set(), {0, 1})),
        # (((0, 0), (1, 7), set(), set()), ((0, 1), (0, -1)), ((0, 1), (1, 6), set(), {1})),
        # (((0, 1), (1, 7), set(), set()), ((0, 1), (0, -1)), ((0, 2), (1, 6), {0}, {1})),
        # (((0, 1), (1, 7), set(), {0}), ((0, 1), (0, -1)), ((0, 2), (1, 6), set(), {0, 1})),
        # (((0, 1), (1, 7), {0, 1}, set()), ((0, 1), (0, -1)), ((0, 2), (1, 6), {0, 1}, set())),
        # (((0, 1), (1, 7), {1}, {0}), ((0, 1), (0, -1)),((0, 2), (1, 6), {1}, {0})),
        # (((1, 5), (1, 7), set(), set()), ((0, 1), (0, -1)),((1, 6), (1, 6), {1}, set())),
    )
    @unpack
    def test_transition(self, state, action, gt_next_state):
        next_state = self.env.transition(state, action)
        for i in range(len(next_state)):
            self.assertListEqual(list(next_state[i]), list(gt_next_state[i]))

    # @data(
    # (((0, 0), (1, 7), set(), {0}),
    #  ((-1, 0), (1, 0)), {0}, GOAL_REWARD - 2 * ACTION_COST),
    # (((0, 0), (1, 7), {0}, {1}), ((-1, 0), (1, 0)), {0}, -2 * ACTION_COST),
    # (((0, 1), (1, 7), set(), {1}),
    #  ((0, 1), (1, 0)), {0}, -2 * ACTION_COST),
    # (((0, 7), (1, 7), set(), {1}),
    #  ((1, 0), (1, 0)), {0, 1}, -2 * ACTION_COST),
    # (((0, 7), (1, 7), {0}, {1}),
    #  ((1, 0), (1, 0)), {0, 1}, -2 * ACTION_COST),
    # (((0, 7), (1, 7), set(), {1}),
    #  ((1, 0), (1, 0)), {0}, -2 * ACTION_COST),
    # (((0, 7), (1, 7), {0}, {1}),
    #  ((1, 0), (1, 0)), {0}, GOAL_REWARD - 2 * ACTION_COST),
    # (((0, 7), (1, 7), {0}, {1}),
    #  ((1, 0), (1, 0)), set(), GOAL_REWARD - 2 * ACTION_COST),
    # (((0, 7), (1, 7), set(), {1}),
    #  ((1, 0), (1, 0)), set(), GOAL_REWARD - 2 * ACTION_COST),
    # (((0, 7), (1, 7), set(), set()),
    #  ((1, 0), (1, 0)), set(), -2 * ACTION_COST),
    # (((0, 7), (1, 7), set(), {0, 1}),
    #  ((1, 0), (1, 0)), set(), -2 * ACTION_COST),
    # (((0, 4), (1, 7), set(), set()),
    #  ((1, 0), (0, -1)), {1}, -2 * ACTION_COST),
    # (((0, 4), (1, 7), set(), set()),
    #  ((1, 0), (0, -1)), {1}, -PICKUP_COST - 2 * ACTION_COST), )
    # @unpack
    # def test_reward(self, state, action, goal, gt_reward):
    #     self.env.set_goal(goal)
    #     reward = self.env.reward(state, action)
    #     self.assertEqual(reward, gt_reward)

    # @data(
    #     (((1, 1), (0, 0), set(), set()), set(), False),
    #     (((1, 1), (0, 0), {0}, {0}), {0}, False),
    #     (((1, 1), (1, 7), set(), set()), set(), False),
    #     (((1, 1), (1, 7), set(), {0}), set(), True),
    #     (((1, 1), (1, 7), set(), {1}), set(), True),
    #     (((1, 7), (1, 7), {0}, {1}), {1}, True),
    #     (((1, 7), (1, 7), {0}, set()), {1}, False),
    #     (((1, 1), (1, 7), set(), {0, 1}), set(), True),
    #     (((1, 5), (1, 3), {0}, {1}), {0, 1}, True),
    #     (((1, 5), (1, 3), {0}, {1}), set(), False),
    #     (((1, 5), (1, 3), {0}, {1}), {1}, False),
    # )
    # @unpack
    # def test_is_terminal_state(self, state, goal, gt):
    #     self.env.set_goal(goal)
    #     self.assertEqual(self.env.is_terminal_state(state), gt)


if __name__ == "__main__":
    unittest.main()
