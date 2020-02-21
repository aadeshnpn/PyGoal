"""Learn policy for MDP world using GenRecProp algorithm."""

import numpy as np
from py_trees.trees import BehaviourTree


from pygoal.lib.mdp import GridMDP, create_policy
from pygoal.lib.genrecprop import GenRecPropMDP, GenRecPropMDPNear
from pygoal.utils.bt import goalspec2BT


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
    goalspec = 'F P_[L][33,none,==]'
    startpoc = (3, 0)
    # env1 = init_mdp(startpoc)
    env2 = init_mdp(startpoc)
    keys = ['L', 'IC', 'IT']
    actions = [0, 1, 2, 3]
    gmdp = GenRecPropMDP(env2, keys, goalspec, dict(), 30, actions, False)
    gmdp.gen_rec_prop(100)

    policy = create_policy(gmdp.gtable)

    print(gmdp.run_policy(policy))


def get_near_trap(seed):
    # Define the environment for the experiment
    goalspec = 'F P_[NT][True,none,==]'
    startpoc = (3, 0)
    # env1 = init_mdp(startpoc)
    env2 = init_mdp(startpoc)
    keys = ['L', 'IC', 'IT', 'NC', 'NT']
    actions = [0, 1, 2, 3]
    gmdp = GenRecPropMDPNear(env2, keys, goalspec, dict(), 30, actions, False)
    gmdp.gen_rec_prop(100)

    policy = create_policy(gmdp.gtable)

    print(gmdp.run_policy(policy))


def find_cheese_return(seed):
    # Define the environment for the experiment
    goalspec = 'F P_[IC][True,none,==] U F P_[L][13,none,==]'
    startpoc = (1, 3)
    # env1 = init_mdp(startpoc)
    env2 = init_mdp(startpoc)
    keys = ['L', 'IC', 'IT', 'NC', 'NT']
    actions = [0, 1, 2, 3]
    gmdp = GenRecPropMDPNear(env2, keys, goalspec, dict(), 30, actions, False)
    gmdp.gen_rec_prop(100)

    policy = create_policy(gmdp.gtable)

    print(gmdp.run_policy(policy))


def find_cheese_bt(seed):
    # Define the environment for the experiment
    # goalspec = 'F P_[IC][True,none,==] U F P_[L][13,none,==]'
    goalspec = 'F P_[IC][True,none,==]'
    startpoc = (1, 3)
    env = init_mdp(startpoc)
    keys = ['L', 'IC']
    actions = [0, 1, 2, 3]
    planner = GenRecPropMDP(
        env, keys, goalspec, dict(), actions=actions, max_trace=10)
    root = goalspec2BT(goalspec, planner=planner)
    behaviour_tree = BehaviourTree(root)
    # print(behaviour_tree)
    for i in range(1):
        behaviour_tree.tick()
        print(behaviour_tree.root.status)
    # print(type(behaviour_tree.root), dir(behaviour_tree.root))
    # print(behaviour_tree.root.train)

    behaviour_tree.root.train = False
    # # print(behaviour_tree.root.train)
    behaviour_tree.root.planner.env.restart()
    for i in range(1):
        behaviour_tree.tick()
        print(behaviour_tree.root.status)


def main():
    # find_cheese(1234)
    # get_near_trap(123)
    find_cheese_bt(12)


main()
