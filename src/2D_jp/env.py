import copy
import numpy as np
import itertools as it

# local import
from util import GOAL_REWARD, ACTION_COST, StateEncoding, PICKUP_COST, WORLD_DICT


class Env2D:
    def _is_in_bound(self, pos):
        if pos[0] in range(self.env_size[0]) and pos[1] in range(
                self.env_size[1]):
            return True
        else:
            return False

    def __init__(self, env_size, env_type, goal, obj_poss: list,
                 terminal_pos: list):
        self.env_size = env_size
        self.env_type = env_type
        self.goal = goal
        self.obj_poss = obj_poss
        self.terminal_pos = terminal_pos

        self.num_agents = 2

        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.num_actions = len(self.actions)

        self.action_space = list(it.product(*[self.actions, self.actions]))

        self.positions = [(x, y) for x in range(self.env_size[0])
                          for y in range(self.env_size[1])]

        positions = copy.deepcopy(self.positions)

        self.num_positions = len(self.positions)

        self.collection_space = ({0}, {1}, {0, 1}, set())
        collection_space = copy.deepcopy(self.collection_space)
        self.num_collections = len(self.collection_space)

        self.state_space = list(
            it.product(
                *[positions, positions, collection_space, collection_space]))

        # prune state space to remove redundant collection cases
        self.prune_state_space()

        self.action_cost = ACTION_COST
        self.goal_reward = GOAL_REWARD

    def get_state_index(self, state):

        i0, i1 = [self.positions.index(state[i]) for i in range(2)]
        i2, i3 = [self.collection_space.index(state[i]) for i in range(2, 4)]

        return (i0, i1, i2, i3)

    def prune_state_space(self):
        self.state_space_encoding = np.ones(
            (self.num_positions, self.num_positions, self.num_collections,
             self.num_collections))

        for s in self.state_space:
            ind = self.get_state_index(s)
            if not self.is_valid_state(s):
                self.state_space_encoding[ind] = StateEncoding.BAD.value
            elif self.is_terminal_state(s):
                self.state_space_encoding[ind] = StateEncoding.TERMINAL.value

    def is_valid_state(self, s):
        p1, p2, c1, c2 = s

        condition_1 = (c1 == c2 and c1 != set())

        condition_2 = (c1 == {0, 1} and c2 != set()) or (c2 == {0, 1}
                                                         and c1 != set())

        condition_3 = any([c in c2 for c in c1])

        condition_4 = any([(p1 == self.obj_poss[o_i]) and (o_i not in c1)
                           and (o_i not in c2)
                           for o_i in range(len(self.obj_poss))])

        return False if any(
            [condition_1, condition_2, condition_3, condition_4]) else True

    def transition(self, state, action):
        # transition position
        prev_state = copy.deepcopy(state)
        next_state = list(tuple(e) for e in np.add(prev_state[:2], action))
        for i in range(len(next_state)):
            if not self._is_in_bound(next_state[i]):
                next_state[i] = prev_state[i]

        # transition collection
        c0 = copy.deepcopy(prev_state[-2])
        c1 = copy.deepcopy(prev_state[-1])
        curr_coll = [c0, c1]

        for obj_i, obj_pos in enumerate(self.obj_poss):
            for a_i, a_pos in enumerate(next_state):
                if (obj_pos == a_pos) and (obj_i not in curr_coll[0]) and (
                        obj_i not in curr_coll[1]):
                    curr_coll[a_i].add(obj_i)

        for c in curr_coll:
            next_state.append(c)

        # A2 cannot move if env_type is 0
        if self.env_type == WORLD_DICT["hands-tied"]:
            next_state[1] = state[1][:]
            next_state[3] = state[3]

        if self.env_type == 2:
            next_state[0] = state[0][:]
            next_state[2] = state[2]

        return tuple(next_state)

    def get_total_coll(self, c1, c2):
        if c1 == set():
            return c2
        elif c2 == set():
            return c1
        else:
            return c1.union(c2)

    def reward(self, state, action):
        next_state = self.transition(state, action)
        # print(state, action, next_state)
        poss = next_state[:2]
        colls = next_state[-2:]

        reward = 0
        # action cost
        for a in action:
            if a != (0, 0):
                reward -= self.action_cost

        # goal reward
        for i in range(self.num_agents):
            if self.goal != set() and self.goal == colls[i]:
                # print(self.goal, colls[i], poss[i])
                if poss[i] == self.terminal_pos:
                    reward += self.goal_reward

            elif self.goal == set() and len(colls[i]) == 1:
                if poss[i] == self.terminal_pos:
                    reward += self.goal_reward
                    break

        # pickup cost
        for i in range(len(colls)):
            if next_state[-2:][i] != state[-2:][i]:
                reward -= PICKUP_COST

        return reward

    def _is_terminal_state_single(self, pos, coll):
        # if matches the goal --> terminal
        if (self.goal != set()) and (coll == set(
                self.goal)) and (pos == self.terminal_pos):
            return True
        elif self.goal == set() and len(coll) == 1 and (
                pos == self.terminal_pos):
            return True
        else:
            return False

    def is_terminal_state(self, state):
        # check if one agent succeeds already
        for i in range(self.num_agents):
            pos = state[i]
            coll = state[i + self.num_agents]
            if self._is_terminal_state_single(pos, coll):
                return True

        # # early termination if wrong things in collection already
        # colls = state[-2:]
        # # goal is not both but both in some collection
        # if len(self.goal) != 2 and any([len(coll) == 2 for coll in colls]):
        #     return True
        # elif len(self.goal) == 2:
        #     if len(colls[0]) == 1 and len(colls[1]) == 1:
        #         return True

        return False

    def set_goal(self, goal):
        self.goal = goal

        self.prune_state_space()


def test():
    env_size = (2, 8)
    env_type = 0
    goal = set()
    obj_poss = [[0, 0], [0, 6]]
    env_size = (2, 4)
    obj_poss = [(0, 0), (0, 1)]
    terminal_pos = [env_size[i] - 1 for i in range(len(env_size))]
    env = Env2D(env_size=env_size,
                env_type=env_type,
                goal=goal,
                obj_poss=obj_poss,
                terminal_pos=terminal_pos)
    state = ((0, 2), (0, 2), set(), set())
    # print(env.is_valid_state(state))
    # actions = [([1, 0], [1, 0]), ([1, 0], [-1, 0]), ([1, 0], [0, 1]),
    #            ([1, 0], [0, -1])]
    action = ([1, 0], [0, -1])
    next_state = env.transition(state, action)
    print(next_state)


if __name__ == "__main__":
    test()