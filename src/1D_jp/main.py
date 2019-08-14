import os

import click
import numpy as np
import matplotlib.pyplot as plt

# local imports
from env import Env, PassingEnv
from algo import ValueIteration
from goal_inference import GoalInference
from rendering import Render1DGrid
from util import ENV_TYPE


@click.command()
@click.option('--goal-ind', default=0, help='Which goal: [0], [1], [0, 1], []')
@click.option('--signal', default=0, help='signal type: 0 --> "help')
@click.option('--env-type', default=0, help='0, 1, 2, 3')
def main(goal_ind, signal, env_type):
    # initialization
    goal_space = [{0}, {1}, set()]
    env_length = 8
    # env = Env(env_length,
    #           goal=goal_space[goal_ind],
    #           goal_space=goal_space,
    #           env_type=ENV_TYPE[env_type])

    env = PassingEnv(env_length=env_length,
                     goal=goal_space[goal_ind],
                     goal_space=goal_space,
                     env_type=ENV_TYPE[env_type])

    beta = 0.04
    gamma = 0.9
    epsilon = 0.0001
    vi_jp = ValueIteration(gamma=gamma, epsilon=epsilon, env=env)

    render = Render1DGrid(env)

    # goal inference
    print("Inferring goal ===>")
    goal_inference = GoalInference(beta=beta)
    signal = signal
    if env_type == 0:
        world = 1  # hands-tied
    elif env_type == 1:
        world = 0  # hands-free

    inferred_goal_dist = goal_inference(signal, world)
    # if uniform distribution
    no_goal_pref = (inferred_goal_dist.count(
        inferred_goal_dist[0]) == len(inferred_goal_dist))
    if no_goal_pref:
        inferred_goal = set()
    else:
        # inferred_goal = np.random.choice(goal_space[:2], p=inferred_goal_dist)
        inferred_goal = {np.argmax(inferred_goal_dist)}
    print("Goal Distribution:", inferred_goal_dist, ", Inferred goal:",
          inferred_goal)

    # Learning the value dict using value iteration
    v_list_jp = vi_jp(inferred_goal)

    # running parameters
    start_states = (3, 7, set())

    max_episode_length = 10
    states = start_states
    curr_step = 0

    print("Planning and control ===>")
    state_seq = [start_states]
    while True:
        if curr_step >= max_episode_length:
            break

        render(states)

        print("Current step:", curr_step)

        actions = vi_jp.select_action(v_list_jp, states, method="softmax")
        next_states = env.transition(states, actions)
        reward = env.reward(states, actions)

        states = next_states
        state_seq.append(states)
        if env.is_terminal_state(states):
            render(states)
            break

        curr_step += 1

    print("state sequence", state_seq)


def plot_dist(x, y, x_name, y_name, title, path="./plot/"):
    plt.figure()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    plt.bar(x, y)

    save_path = os.path.join(path, title)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(save_path)


if __name__ == "__main__":
    main()