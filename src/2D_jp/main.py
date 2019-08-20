import os

import click
import numpy as np
import matplotlib.pyplot as plt

# local imports
from env import Env2D
from algo import ValueIteration
from rendering import Render2DGrid

from util import GOAL_SPACE


@click.command()
@click.option("--goal-ind", default=2, help="goal index: 0 or 1")
@click.option("--env-type",
              default=0,
              help="env type: 0(cannot move), 1(move)")
@click.option("--train", default=False, help="train flag")
@click.option("--test", default=True, help="test flag")
@click.option("--display",
              default=True,
              help="flag for turning on the renderer")
def main(goal_ind, env_type, train, test, display):
    env_size = (2, 8)
    goal = GOAL_SPACE[goal_ind]
    obj_poss = [(0, 0), (0, 6)]

    terminal_pos = (0, env_size[1] - 1)
    env = Env2D(env_size=env_size,
                env_type=env_type,
                goal=goal,
                obj_poss=obj_poss,
                terminal_pos=terminal_pos)

    gamma = 0.99
    epsilon = 0.01
    vi_jp = ValueIteration(gamma=gamma, epsilon=epsilon, env=env)

    render = Render2DGrid(env)

    # get value table
    if train:
        v_states = vi_jp(goal)
        print("value iteration finished, saving...")
        np.save("save_points/v_table_{}_{}.npy".format(goal_ind, env_type),
                v_states)

    if train and not test:
        exit()

    v_states = np.load("save_points/v_table_{}_{}.npy".format(
        goal_ind, env_type))

    start_state = ((1, 3), (0, 7), set(), set())

    max_episode_len = 20
    state = start_state
    curr_step = 0

    while True:
        if curr_step >= max_episode_len:
            break

        if display:
            render(state)

        print("curr step:", curr_step)
        if env.is_terminal_state(state):
            print("Reached terminal state")
            break

        action = vi_jp.select_action(v_states, state, method="max")

        next_state = env.transition(state, action)

        state = next_state

        curr_step += 1


if __name__ == "__main__":
    main()
