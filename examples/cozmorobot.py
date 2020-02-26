'''Make Cozmo to accomplish goal specified by our GoalFramework.

The script shows an example of accomplishing complex goal using
GoalFrame work.
'''

import asyncio
import cozmo
from cozmo.util import degrees, distance_mm

from py_trees.trees import BehaviourTree
from py_trees import Blackboard
# from pygoal.lib.genrecprop import GenRecPropTaxi
from pygoal.utils.bt import goalspec2BT

from robot import ComplexGoal

class CozmoPlanner:
    def __init__(
            self, env, keys, goalspec, max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, policy=None):
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.max_trace_len = max_trace
        self.actions = actions
        self.seed = seed
        self.policy = policy
        self.blackboard = Blackboard()

    def train(self, epoch=10, verbose=False):
        pass

    def inference(self, render=False, verbose=False):
        # conn = cozmo.run.connect()
        print('inference', self.policy)
        cozmo.run_program(
            self.policy, use_viewer=False, force_viewer_on_top=False)
        if self.blackboard.shared_content['status']:
            return True
        else:
            return False
        # cozmo.run.connect(self.policy)

# Task description for cozmo robot
# Random Walk.
# Finding cube
# Go to Cube
# Carry cube
# Find charging station
# Drop the cube near charging station


def cozmo_reset():
    pass


def reset_env(robot):
    blackboard = Blackboard()
    blackboard.shared_content['robot'] = robot


def cozmomain():
    goal = 'F(P_[P][2,none,==])'
    goalspec = '((((('+goal+' U '+goal+') U '+goal+') U '+goal+') U '+goal+') U '+goal+')'  # noqa:
    # goalspec = goal+' U '+goal
    print(goalspec)
    keys = ['P', 'DC', 'FC', 'CC', 'DD', 'FD', 'D']

    # actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    policies = [
        detect_cube, find_cube, carry_cube, find_charger,
        move_to_charger, drop_cube]
    # policies = [detect_cube, find_cube]
    j = 0
    for child in behaviour_tree.root.children:
        # planner = planners[j]
        planner = CozmoPlanner(crobot, keys, child.name, policy=policies[j])
        j += 1
        child.setup(0, planner, False, 5)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(crobot)
        )
        print(i, behaviour_tree.root.status)

    # child.train = False

    # for i in range(1):
    #     behaviour_tree.tick(
    #         pre_tick_handler=reset_env_d(env)
    #     )
    # print(i, behaviour_tree.root.status)


def cozmomain1():
    goalspec = 'F(P_[P][2,none,==])'
    # goalspec = '((((('+goal+' U '+goal+') U '+goal+') U '+goal+') U '+goal+') U '+goal+')'
    # goalspec = goal+' U '+goal
    print(goalspec)
    keys = ['P', 'DC', 'FC', 'CC', 'DD', 'FD', 'D']

    # actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    # policies = [detect_cube, find_cube, carry_cube, find_charger, move_to_charger, drop_cube]
    policies = [detect_cube, find_cube]
    j = 0
    child = behaviour_tree.root
    #for child in behaviour_tree.root.children:
        # planner = planners[j]
    planner = CozmoPlanner(crobot, keys, child.name, policy=policies[j])
    #    j += 1
    child.setup(0, planner, False, 5)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(crobot)
        )
        print(i, behaviour_tree.root.status)


def main():
    # cozmomain()
    # robot = ComplexGoal(normal=True, goal='detect_cube')
    robot = ComplexGoal(normal=False, goal='detect_cube')


if __name__ == '__main__':
    main()
