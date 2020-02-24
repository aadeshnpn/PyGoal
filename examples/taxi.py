"""Learn policy in Taxi world using GenRecProp."""

import gym
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecPropTaxi
from pygoal.utils.bt import goalspec2BT, display_bt

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


def env_setup(env, seed=1234):
    env.seed(seed)
    env.reset()
    # env.env.s = 3
    return env


def reset_env(env, seed=1234):
    env.seed(seed)
    env.reset()
    # env.env.s = 3


def init_taxi(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup(env, seed)
    return env


def init_taxi_d(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup_d(env, seed)
    return env


def env_setup_d(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 479
    return env


def reset_env_d(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 479


def give_loc(idx):
    # # (0,0) -> R -> 0
    # # (4,0) -> Y -> 2
    # # (0,4) -> G -> 1
    # # (4,3) -> B -> 3
    # # In taxi -> 4
    locdict = {0: '00', 1: '04', 2: '40', 3: '43'}
    return locdict[idx]


def taxi():
    env = init_taxi(seed=1234)
    target = list(env.decode(env.s))
    print(target)
    goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'    # noqa: E501
    # goalspec = 'F P_[L]['+give_loc(target[2])+',none,==] U F P_[PI]['+str(4)+',none,==]'  # noqa: E501
    keys = ['L', 'PI', 'DI']
    actions = [[0, 1, 2, 3], [4], [0, 1, 2, 3], [5]]
    # actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None)
    print('root', root)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)

    epoch = [50, 10, 50, 10]
    j = 0
    for child in behaviour_tree.root.children:
        print('children', child, child.name, child.id)
        planner = GenRecPropTaxi(
             env, keys, child.name, dict(), actions=actions[j],
             max_trace=40)
        child.setup(0, planner, True, epoch[j])
        j += 1
        # planner.env = env
        print(child.goalspec, child.planner.goalspec, child.planner.env)
    print('rootname', behaviour_tree.root.name)
    # behaviour_tree.root.remove_child_by_id(id)
    # display_bt(behaviour_tree)
    for i in range(100):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print('Training', behaviour_tree.root.status)

    for child in behaviour_tree.root.children:
        # child.setup(0, planner, True, 20)
        child.train = False
        # print(child, child.name, child.train)

    for i in range(2):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print('inference', behaviour_tree.root.status)


def taxid():
    env = init_taxi_d(seed=1234)
    target = list(env.decode(env.s))
    print(target)
    goalspec = 'F P_[PI]['+str(3)+',none,==]'
    keys = ['L', 'PI', 'DI']
    actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    child = behaviour_tree.root
    planner = GenRecPropTaxi(
        env, keys, child.name, dict(), actions=actions,
        max_trace=5, seed=123)
    child.setup(0, planner, True, 5)
    # 4,3,4,3
    # print(child.goalspec, child.planner.goalspec, child.planner.env)
    for i in range(5):
        behaviour_tree.tick(
            pre_tick_handler=reset_env_d(env)
        )
        print(i, behaviour_tree.root.status)

    child.train = False

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env_d(env)
        )
    print(i, behaviour_tree.root.status)


def main():
    # taxid()
    taxi()


if __name__ == '__main__':
    main()
