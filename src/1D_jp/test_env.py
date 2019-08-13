import unittest
from ddt import ddt, data, unpack

from env import Env
from algo import ValueIteration


@ddt
class testEnv(unittest.TestCase):
    def setUp(self):
        self.env_length = 8
        goal_space = [{0}, {1}, set()]
        self.env = Env(self.env_length, {0},
                       env_type=None,
                       goal_space=goal_space)

        gamma = 0.9
        epsilon = 0.0001
        self.vi_jp = ValueIteration(gamma=gamma, epsilon=epsilon, env=self.env)

    # @data(((1, 1, {0}), [0], True), ((1, 1, {1}), [1], True),
    #       ((1, 1, {0, 1}), [0, 1], True), ((1, 1, set()), [], False),
    #       ((1, 1, {0}), [1], True), ((1, 1, {0, 1}), [1], True),
    #       ((1, 1, {0, 1}), [0], True), ((1, 1, {0, 1}), [1], True),
    #       ((1, 1, set()), [1], False), ((1, 1, {1}), [0, 1], False),
    #       ((1, 1, {0, 1}), [], True), ((1, 6, {1}), [0, 1], False))
    # @unpack
    # def test_is_term_state(self, state, goal, gt_is_terminal):
    #     self.env.goal = goal
    #     is_terminal = self.env.is_terminal_state(state)
    #     self.assertEqual(is_terminal, gt_is_terminal)

    @data(((1, 3, set()), (-1, 0), (0, 3, {0})),
          ((1, 7, set()), (-1, 0), (0, 7, {0})))
    @unpack
    def test_transition(self, states, actions, gt_next_states):
        next_state = self.env.transition(states, actions)
        self.assertTupleEqual(next_state, gt_next_states)

    # @data(
    #     ((3, 6, set()), [], (0, -1)),
    #     ((3, 6, set()), [0], (-1, 0)),
    #     ((3, 6, set()), [1], (0, -1)),
    #     ((3, 6, set()), [0, 1], (-1, 0)),  # need more attention!!!
    #     ((1, 6, set()), [0, 1], (-1, -1)),
    #     ((1, 6, set()), [0], (-1, 0)),
    #     ((1, 6, set()), [1], (0, -1)),
    #     ((1, 3, set()), [], (-1, 0)),
    #     ((1, 3, set()), [0], (-1, 0)),
    #     ((1, 3, set()), [1], (0, 1)),
    #     ((1, 3, set()), [0, 1], (0, 1)),  # need more attention!!!
    # )
    # @unpack
    # def test_first_step_action(self, start_states, goal, gt_actions):
    #     v_list = self.vi_jp(goal)
    #     actions = self.vi_jp.select_action(v_list, start_states)

    #     self.assertTupleEqual(actions, gt_actions)

    # @data(
    #     ((1, 6, set()), []), )
    # @unpack
    # def test_corner_cases_goal_either(self, start_states, goal):
    #     v_list = self.vi_jp(goal)
    #     action_counts = {}
    #     for _ in range(2000):
    #         actions = self.vi_jp.select_action(v_list, start_states)
    #         action_counts[actions] = action_counts.get(actions, 0) + 1

    #     self.assertEqual(len(action_counts), 2)
    #     expected_action_freq = 1 / len(action_counts)
    #     expected_action_freq = round(expected_action_freq, 1)

    #     single_act_count = list(action_counts.values())[0]
    #     total_counts = sum(list(action_counts.values()))
    #     action_freq = single_act_count / total_counts
    #     action_freq = round(action_freq, 1)
    #     self.assertAlmostEqual(expected_action_freq, action_freq)


if __name__ == "__main__":
    unittest.main()