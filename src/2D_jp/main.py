import os

import time
import click
import pygame
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *

# local imports
from env import Env2D
from algo import ValueIteration, one_step_lookahead
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
              default="hands-tied",
              help="two worlds: 'hands-free' or 'hands-tied'")
@click.option("--train", default=False, help="train flag")
@click.option("--test", default=True, help="test flag")
@click.option("--display",
              default=True,
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

    if not os.path.exists("save_points"):
        os.makedirs("save_points")
    save_path = "save_points/v_table_{}_{}_2_goals_test.npy".format(
        goal_ind, env_type)

    if display:
        # render = Render2DGrid(env)
        render = Render(env_type=env_type)

    # get value table
    if train:
        v_states = vi_jp(goal=goal, path=save_path)
    #     print("value iteration finished.")

    if test:
        v_states_list = []
        envs = []
        for goal_i in range(len(GOAL_SPACE)):
            load_path = "save_points/v_table_{}_{}_2_goals_test.npy".format(
                goal_i, env_type)
            v_states_list.append(np.load(load_path))
            envs.append(
                Env2D(env_size=env_size,
                      env_type=env_type,
                      goal=GOAL_SPACE[goal_i],
                      obj_poss=obj_poss,
                      terminal_pos=terminal_pos))

        beta = 10
        temp = 0.01

        GI = GoalInference(beta=beta, temp=temp, env=env)
        start_state = ((1, 3), (0, 7), set(), set())

        # select first step speaker action
        v_states_gt = v_states_list[goal_ind]
        q_values_gt = one_step_lookahead(env, v_states_gt, start_state, gamma)
        act_sig_dist = GI.compute_likelihood(goal_ind, world, start_state,
                                             q_values_gt)
        act_sig = GI.action_sigs_pruned[np.argmax(act_sig_dist)]
        print("selected", act_sig)

        # infer goal
        q_values_list = []
        for i in range(len(GOAL_SPACE)):
            q_values_list.append(
                one_step_lookahead(envs[i], v_states_list[i], start_state,
                                   gamma))
        goal_dist = GI(act_sig, world, start_state, q_values_list)

        inferred_goal_ind = np.argmax(goal_dist)
        print("inferred goal", inferred_goal_ind)
        v_states_inferred = v_states_list[inferred_goal_ind]

        # reset env goal
        env.set_goal(GOAL_SPACE[inferred_goal_ind])

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

                action = vi_jp.select_action(v_states_inferred,
                                             state,
                                             method="max")

                next_state = env.transition(state, action)

                state = next_state

                curr_step += 1

    if save_perf:
        opt, avg, error = get_perf_measure(vi=vi_jp, num_tests=num_tests)

        f = open("data/num_steps.txt", "a+")
        print("{} {} {} {} {}".format(goal_ind, env_type, opt, avg, error),
              file=f)


if __name__ == "__main__":
    run()
