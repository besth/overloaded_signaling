import os

import time
import click
import pygame
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *

# local imports
from env import Env2D
from algo import ValueIteration
from rendering import Render2DGrid, Render
from inference import GoalInference

from util import GOAL_SPACE, WORLD_DICT


def get_perf_measure(vi, num_tests):
    goal_ind = GOAL_SPACE.index(vi.env.goal)
    env_type = vi.env.env_type

    v_tables = np.load("save_points/v_table_{}_{}.npy".format(
        goal_ind, env_type))

    start_state = ((1, 3), (0, 7), set(), set())

    max_episode_len = 20
    state = start_state
    curr_step = 0
    optimal_steps = 0

    while True:
        if curr_step >= max_episode_len:
            break

        if vi.env.is_terminal_state(state):
            break

        action = vi.select_action(v_tables, state, method="max")
        for a in action:
            if a != (0, 0):
                optimal_steps += 1

        next_state = vi.env.transition(state, action)

        state = next_state

        curr_step += 1

    # optimal_steps = curr_step

    steps = []
    for i in range(num_tests):
        print("test step", i)
        steps.append(0)
        state = start_state
        curr_step = 0
        while True:
            if curr_step >= max_episode_len:
                break

            if vi.env.is_terminal_state(state):
                break

            action = vi.select_action(v_tables, state, method="softmax")
            for a in action:
                if a != (0, 0):
                    steps[-1] += 1

            next_state = vi.env.transition(state, action)

            state = next_state

            curr_step += 1

        # steps.append(curr_step)

    avg_steps = np.mean(steps)
    std_steps = np.std(steps)

    return (optimal_steps, avg_steps, std_steps)


@click.command()
@click.option("--goal-ind", default=0, help="goal index: 0 or 1")
@click.option("--world",
              default="hands-free",
              help="env type: 0(cannot move), 1(move)")
@click.option("--train", default=False, help="train flag")
@click.option("--test", default=True, help="test flag")
@click.option("--display",
              default=False,
              help="flag for turning on the renderer")
@click.option("--save-perf",
              default=False,
              help="whether to save the testing performance")
@click.option("--num-tests",
              default=1000,
              help="number of tests to be averaged over")
def run(goal_ind, world, train, test, display, save_perf, num_tests):
    env_size = (2, 8)
    goal = GOAL_SPACE[goal_ind]
    obj_poss = [(0, 0), (0, 6)]

    terminal_pos = (0, env_size[1] - 1)
    env_type = WORLD_DICT[world]
    env = Env2D(env_size=env_size,
                env_type=env_type,
                goal=goal,
                obj_poss=obj_poss,
                terminal_pos=terminal_pos)

    gamma = 0.99
    epsilon = 0.01
    tau = 0.2
    vi_jp = ValueIteration(gamma=gamma, epsilon=epsilon, tau=tau, env=env)

    if display:
        # render = Render2DGrid(env)
        render = Render(env_type=env_type)

    # get value table
    if train:
        v_states = vi_jp(goal)
        print("value iteration finished.")
        # np.save(
        #     "save_points/v_table_{}_{}_test.npy".format(goal_ind, env_type),
        #     v_states)

    if test:
        v_states = np.load("save_points/v_table_{}_{}_test.npy".format(
            goal_ind, env_type))

        # perform goal inference after we get v_tables
        beta = 0.1
        GI = GoalInference(beta=beta, env=env, vi_obj=vi_jp, v_table=v_states)

        start_state = ((1, 3), (0, 7), set(), set())

        # llh = GI.compute_likelihood(goal_ind, env_type, start_state)
        # print(llh, np.argmax(llh), GI.action_sigs_pruned[np.argmax(llh)])

        goal_d = GI(((0, 0), 0), env_type, start_state)
        print("goal_d", goal_d)

        exit()
        while True:
            max_episode_len = 20
            state = start_state
            curr_step = 0

            while True:
                if curr_step >= max_episode_len:
                    break

                if display:
                    render(state)
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()

                    time.sleep(0.3)

                print("curr step:", curr_step)
                if env.is_terminal_state(state):
                    print("Reached terminal state")
                    if display:
                        render(state)
                        # while True:
                        #     render(state)
                        #     for event in pygame.event.get():
                        #         if event.type == QUIT:
                        #             pygame.quit()
                    break

                action = vi_jp.select_action(v_states, state, method="max")

                next_state = env.transition(state, action)

                state = next_state

                curr_step += 1

    if save_perf:
        opt, avg, error = get_perf_measure(vi=vi_jp, num_tests=num_tests)

        f = open("data/num_steps.txt", "a+")
        print("{} {} {} {} {}".format(goal_ind, env_type, opt, avg, error),
              file=f)


# if __name__ == "__main__":
#     run()
