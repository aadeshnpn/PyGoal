"""Learn policy for MDP world using Modified Qlearning algorithm."""

import numpy as np

from pygoal.lib.mdp import GridMDP, orientations, turns, dictmax, create_policy
from pygoal.lib.genrecprop import GenRecPropMDP, GenRecPropMDPNear
from pygoal.utils.bt import goalspec2BT, reset_env


def qlearningmod(env, startloc, epoch):
    # qtable = np.zeros((len(self.states), 4))
    qtable = dict()
    for state in env.states:
        qtable[state] = dict(zip(orientations, [0.25, 0.25, 0.25, 0.25]))
    alpha = 0.9
    gamma = 0.6
    epsilon = 0.1
    # slookup = dict(zip(self.states, range(len(self.states))))
    for e in range(epoch):
        reward = 0
        state = (2, 3)
        while True:
            # if env.nprandom.uniform(0, 1) < epsilon:
            #    action = env.nprandom.choice([0, 1, 2, 3])
            #    action = orientations[action]
            #else:
            action = dictmax(qtable[state], s='key')
            p, s1 = zip(*env.T(state, action))
            p, s1 = list(p), list(s1)
            s1 = s1[np.argmax(p)]
            next_state = s1
            reward = env.R(next_state)
            if action is None:
                break
            old_value = qtable[state][action]
            next_max = dictmax(qtable[next_state], s='val')
            print(e, 'state', state, 'action', action)
            new_value = (1-alpha) * old_value + alpha * (
                reward + gamma * next_max)
            print(e, 'old value', old_value, 'new value', new_value)
            print(e, 'old qtable', qtable[state])
            # qtable[state][action] = new_value
            # state = next_state

            qtable[state][action] = new_value
            probs = np.array(list(qtable[state].values()))
            probs = probs / probs.sum()

            qtable[state] = dict(zip(qtable[state].keys(), probs))
            print(e, 'new qtable',qtable[state])
            state = next_state

            if state in env.terminals:
                break
    return qtable


def qlearning(env, startloc, epoch):
    # qtable = np.zeros((len(self.states), 4))
    qtable = dict()
    for state in env.states:
        qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    # slookup = dict(zip(self.states, range(len(self.states))))
    for e in range(epoch):
        reward = 0
        state = startloc
        while True:
            if env.nprandom.uniform(0, 1) < epsilon:
                action = env.nprandom.choice([0, 1, 2, 3])
                action = orientations[action]
            else:
                action = dictmax(qtable[state], s='key')
            p, s1 = zip(*env.T(state, action))
            p, s1 = list(p), list(s1)
            s1 = s1[np.argmax(p)]
            next_state = s1
            reward = env.R(next_state)
            if action is None:
                break
            old_value = qtable[state][action]
            next_max = dictmax(qtable[next_state], s='val')
            new_value = (1-alpha) * old_value + alpha * (
                reward + gamma * next_max)
            qtable[state][action] = new_value
            state = next_state
            if state in env.terminals:
                break
    return qtable


def init_mdp(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trap
    grid[0][3] = None
    grid[1][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[1][3] = -2

    mdp = GridMDP(
        grid, terminals=[(3, 3), (3, 2)], startloc=sloc)

    return mdp


def find_cheese(seed):
    # Define the environment for the experiment
    startpoc = (3, 1)
    env = init_mdp(startpoc)
    # policy = qlearning(env, startpoc, 100)
    policy = qlearningmod(env, startpoc, 3)
    # print(policy)
    policy = create_policy(policy)
    visual = env.to_arrows(policy)
    for row in visual:
        print(row)
    # print(visual)
    # actions = [0, 1, 2, 3]


def main():
    find_cheese(12)

main()
