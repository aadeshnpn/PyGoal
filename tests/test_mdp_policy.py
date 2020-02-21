from unittest import TestCase
from py_trees.trees import BehaviourTree
from py_trees import Status
import numpy as np

from pygoal.lib.mdp import GridMDP
from pygoal.lib.genrecprop import GenRecPropMDP
from pygoal.utils.bt import goalspec2BT, reset_env


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


class TestMDPTraining(TestCase):

    def setUp(self):
        goalspec = 'F P_[IC][True,none,==]'
        startpoc = (1, 3)
        env = init_mdp(startpoc)
        keys = ['L', 'IC']
        actions = [0, 1, 2, 3]
        planner = GenRecPropMDP(
            env, keys, goalspec, dict(), actions=actions, max_trace=10)
        root = goalspec2BT(goalspec, planner=planner)
        self.behaviour_tree = BehaviourTree(root)

        for child in self.behaviour_tree.root.children:
            print(child, child.name)
            child.setup(0, planner, True, 10)
            # child.planner.env = env
            print(child.goalspec, child.planner.goalspec, child.planner.env)

        for i in range(10):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )

    def test_training(self):
        self.assertEqual(self.behaviour_tree.root.status, Status.SUCCESS)


class TestMDPInference(TestCase):

    def setUp(self):
        goalspec = 'F P_[IC][True,none,==]'
        startpoc = (1, 3)
        env = init_mdp(startpoc)
        keys = ['L', 'IC']
        actions = [0, 1, 2, 3]
        planner = GenRecPropMDP(
            env, keys, goalspec, dict(), actions=actions, max_trace=10)
        root = goalspec2BT(goalspec, planner=planner)
        self.behaviour_tree = BehaviourTree(root)

        for child in self.behaviour_tree.root.children:
            print(child, child.name)
            child.setup(0, planner, True, 10)
            # child.planner.env = env
            print(child.goalspec, child.planner.goalspec)

        for i in range(10):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )

        self.behaviour_tree.root.train = False
        self.behaviour_tree.root.planner.env.restart()
        for i in range(1):
            self.behaviour_tree.tick()

    def test_inference(self):
        self.assertEqual(self.behaviour_tree.root.status, Status.SUCCESS)


class TestMDPSequenceGoal(TestCase):

    def setUp(self):
        goalspec = 'F P_[IC][True,none,==] U F P_[L][13,none,==]'
        startpoc = (1, 3)
        self.env = init_mdp(startpoc)
        keys = ['L', 'IC']
        actions = [0, 1, 2, 3]
        planner = GenRecPropMDP(
            self.env, keys, goalspec, dict(), actions=actions, max_trace=10)
        root = goalspec2BT(goalspec, planner=planner)
        self.behaviour_tree = BehaviourTree(root)

        for child in self.behaviour_tree.root.children:
            # print(child, child.name)
            child.setup(0, planner, True, 20)
            # child.planner.env = env
            # print(child.goalspec, child.planner.goalspec)

        for i in range(20):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(self.env)
            )

        for child in self.behaviour_tree.root.children:
            child.setup(0, planner, True, 10)
            child.train = False
            # print(child, child.name, child.train)

        for i in range(2):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(self.env)
            )
        # print('inference', behaviour_tree.root.status)

    def test_inference(self):
        self.assertEqual(self.behaviour_tree.root.status, Status.SUCCESS)

    def test_final_loc(self):
        self.assertEqual(self.env.curr_loc, (1, 3))
