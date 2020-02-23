"""Learn policy in Taxi world using GenRecProp."""

import gym
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecPropTaxi
from pygoal.utils.bt import goalspec2BT


def env_setup(env, seed=123):
    env.seed(seed)
    env.reset()
    return env


def reset_env(env, seed=123):
    env.seed(seed)
    env.reset()


def init_taxi(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup(env, seed)
    return env


def give_loc(idx):
    # # (0,0) -> R -> 0
    # # (4,0) -> Y -> 2
    # # (0,4) -> G -> 1
    # # (4,3) -> B -> 3
    # # In taxi -> 4
    locdict = {0: '00', 1: '04', 2: '40', 3: '43'}
    return locdict[idx]


def taxi():
    env = init_taxi(seed=123)
    target = list(env.decode(env.s))
    # goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'
    goalspec = 'F(P_[L]['+give_loc(target[2])+',none,==])'
    keys = ['L', 'TI', 'PI']
    actions = [0, 1, 2, 3, 4, 5]
    planner = GenRecPropTaxi(
         env, keys, None, dict(), actions=actions, max_trace=40)
    root = goalspec2BT(goalspec, planner=planner)
    print('root', root)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    child = behaviour_tree.root
    child.setup(0, planner, True, 150)
    print(child.goalspec, child.planner.goalspec, child.planner.env)
    # for child in behaviour_tree.root.children:
    #     print('children', child, child.name)
    #     child.setup(0, planner, True, 150)
    #     # planner.env = env
    #     print(child.goalspec, child.planner.goalspec, child.planner.env)

    for i in range(100):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    # print(behaviour_tree.root.status)
    # for child in behaviour_tree.root.children:
    child.setup(0, planner, True, 40)
    child.train = False
    print(child, child.name, child.train)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print('inference', behaviour_tree.root.status)
    # print(env.curr_loc)


def main():
    taxi()


if __name__ == '__main__':
    main()
