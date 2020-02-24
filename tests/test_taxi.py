from unittest import TestCase
import gym

from py_trees.trees import BehaviourTree
from py_trees import Status

from pygoal.lib.genrecprop import GenRecPropTaxi
from pygoal.utils.bt import goalspec2BT


def env_setup(env, seed=1234):
    env.seed(seed)
    env.reset()
    return env


def reset_env(env, seed=1234):
    env.seed(seed)
    env.reset()


def init_taxi(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup(env, seed)
    return env


def init_taxi_s(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup_s(env, seed)
    return env


def env_setup_s(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 3
    return env


def reset_env_s(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 3


def give_loc(idx):
    # # (0,0) -> R -> 0
    # # (4,0) -> Y -> 2
    # # (0,4) -> G -> 1
    # # (4,3) -> B -> 3
    # # In taxi -> 4
    locdict = {0: '00', 1: '04', 2: '40', 3: '43'}
    return locdict[idx]


class TestTaxiTrainingSimpleGoal(TestCase):

    def setUp(self):
        env = init_taxi(seed=1234)
        target = list(env.decode(env.s))
        goalspec = 'F(P_[L]['+give_loc(target[2])+',none,==])'
        keys = ['L', 'TI', 'PI']
        actions = [0, 1, 2, 3]
        root = goalspec2BT(goalspec, planner=None)
        self.behaviour_tree = BehaviourTree(root)
        child = self.behaviour_tree.root
        planner = GenRecPropTaxi(
            env, keys, None, dict(), actions=actions,
            max_trace=40, seed=1234)
        child.setup(0, planner, True, 50)
        print(child.goalspec, child.planner.goalspec, child.planner.env)
        for i in range(50):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )

    def test_training(self):
        self.assertEqual(self.behaviour_tree.root.status, Status.SUCCESS)


class TestTaxiInferenceSimpleGoal(TestCase):

    def setUp(self):
        env = init_taxi(seed=1234)
        target = list(env.decode(env.s))
        goalspec = 'F(P_[L]['+give_loc(target[2])+',none,==])'
        keys = ['L', 'TI', 'PI']
        actions = [0, 1, 2, 3]
        root = goalspec2BT(goalspec, planner=None)
        self.behaviour_tree = BehaviourTree(root)
        child = self.behaviour_tree.root
        planner = GenRecPropTaxi(
            env, keys, None, dict(), actions=actions,
            max_trace=40, seed=1234)
        child.setup(0, planner, True, 50)
        print(child.goalspec, child.planner.goalspec, child.planner.env)
        for i in range(50):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )

        # child.setup(0, planner, True, 40)
        child.train = False
        print(child, child.name, child.train)

        for i in range(1):
            self.behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )
        print('inference', self.behaviour_tree.root.status)
        # print(env.curr_loc)

    def test_inference(self):
        self.assertEqual(self.behaviour_tree.root.status, Status.SUCCESS)
