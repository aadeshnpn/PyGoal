'''Make Cozmo to accomplish goal specified by our GoalFramework.

The script shows an example of accomplishing complex goal using
GoalFrame work.
'''

import asyncio
import cozmo
from cozmo.util import degrees, distance_mm
import copy

from py_trees.trees import BehaviourTree
from py_trees import Blackboard
from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT

from flloat.parser.ltlfg import LTLfGParser


from robot import ComplexGoal


class CozmoPlanner(GenRecProp):
    def __init__(
            self, env, keys, goalspec, max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, policy=None):
        super().__init__(
            env, keys, goalspec, dict(), max_trace, actions, epoch, seed)
        self.tcount = 0
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.max_trace_len = max_trace
        self.actions = actions
        self.seed = seed
        self.policy = policy
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        temp = list(env.decode(env.s))
        return (str(temp[0])+str(temp[1]), temp[2], temp[3])

    def set_state(self, env, trace, i):
        state = []
        for k in self.keys:
            if k == 'L':
                temp = trace[k][i][-1]
                state.append(int(temp[0]))
                state.append(int(temp[1]))
            else:
                temp = trace[k][i][-1]
                state.append(int(temp))
        state = env.encode(*tuple(state))
        env.env.s = state

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = []
        # Hack to fix the bug of terminal state. Adding big action number
        # This makes the trace length same accross the keys
        trace['A'].append(9)
        return trace

    def get_action_policy(self, policy, state):
        action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        ss = state
        return tuple(ss)

    def get_policy(self):
        policy = dict()
        for s, v in self.gtable.items():
            elem = sorted(v.items(),  key=lambda x: x[1], reverse=True)
            try:
                policy[s] = elem[0][0]
                pass
            except IndexError:
                pass

        return policy

    def evaluate_trace_simple(self, goalspec, trace):
        # Test the prop algorithm
        parser = LTLfGParser()
        parsed_formula = parser(goalspec)
        # Quick fix. create_trace_float requires action to be list of list
        # temp = trace['A']
        # trace['A'] = [temp]
        # Create a trace compatiable with Flloat library
        trace = self.list_to_trace(trace)
        print(trace['CC'])
        t = self.create_trace_flloat(trace, 0)
        result = parsed_formula.truth(t)
        return result

    def train(self, epoch=10, verbose=False):
        pass

    def inference(self, evaluate=True, verbose=False):
        # conn = cozmo.run.connect()
        print('inference', self.policy)
        # cozmo.run_program(
        #    self.policy, use_viewer=False, force_viewer_on_top=False)
        self.env(normal=False, goal=self.policy, tkeys=self.keys)
        # print(self.blackboard.shared_content['states'])
        # self.
        trace = self.blackboard.shared_content['states']
        trace = copy.copy(trace)
        if evaluate:
            if self.evaluate_trace_simple(self.goalspec, trace):
                return True
            else:
                return False
        else:
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
    goal1 = 'F(P_[DC][True,none,==])'
    goal2 = 'F(P_[FC][True,none,==])'
    goal3 = 'F(P_[CC][True,none,==])'
    goal4 = 'G(P_[CC][True,none,==]) & F(P_[DD][True,none,==])'
    goal5 = 'F(P_[FD][True,none,==])'
    goal6 = 'F(P_[CC][False,none,==])'
    # goal1 = 'F(P_[P][2,none,==])'
    # goal1 = 'F(P_[P][2,none,==])'

    goalspec = '((((('+goal1+' U '+goal2+') U '+goal3+') U '+goal4+') U '+goal5+') U '+goal6+')'  # noqa:
    # goalspec = goal+' U '+goal
    print(goalspec)
    keys = ['P', 'DC', 'FC', 'CC', 'DD', 'FD', 'D', 'A']
    # Pose, Detected Cube, Found Cube, Carried Cube, Detected Desk, Found Desk, Drop Desk
    # actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    policies = [
        'detect_cube', 'find_cube', 'carry_cube', 'find_charger',
        'move_to_charger', 'drop_cube']
    # policies = [detect_cube, find_cube]
    j = 0
    for child in behaviour_tree.root.children:
        # planner = planners[j]
        planner = CozmoPlanner(ComplexGoal, keys, child.name, policy=policies[j])
        j += 1
        child.setup(0, planner, False, 5)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(ComplexGoal)
        )
        print(i, behaviour_tree.root.status)

    # child.train = False

    # for i in range(1):
    #     behaviour_tree.tick(
    #         pre_tick_handler=reset_env_d(env)
    #     )
    # print(i, behaviour_tree.root.status)


def cozmomain1():
    goalspec = 'F(P_[DC][2,none,==])'
    # goalspec = '((((('+goal+' U '+goal+') U '+goal+') U '+goal+') U '+goal+') U '+goal+')'
    # goalspec = goal+' U '+goal
    print(goalspec)
    keys = ['P', 'DC', 'FC', 'CC', 'DD', 'FD', 'D']

    # actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    # policies = [detect_cube, find_cube, carry_cube, find_charger, move_to_charger, drop_cube]
    policies = ['detect_cube', 'find_cube']
    j = 0
    child = behaviour_tree.root
    #for child in behaviour_tree.root.children:
        # planner = planners[j]
    planner = CozmoPlanner(ComplexGoal, keys, child.name, policy=policies[j])
    #    j += 1
    child.setup(0, planner, False, 5)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(ComplexGoal)
        )
        print(i, behaviour_tree.root.status)


def cozmomain2():
    goal1 = 'F(P_[DC][True,none,==])'
    goal2 = 'F(P_[FC][True,none,==])'
    # goalspec = '((((('+goal+' U '+goal+') U '+goal+') U '+goal+') U '+goal+') U '+goal+')'
    goalspec = goal1+' U '+goal2
    print(goalspec)
    keys = ['P', 'DC', 'FC', 'CC', 'DD', 'FD', 'D', 'A']

    # actions = [0, 1, 2, 3, 5]
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    # policies = [detect_cube, find_cube, carry_cube, find_charger, move_to_charger, drop_cube]
    policies = ['detect_cube', 'find_cube']
    j = 0
    child = behaviour_tree.root
    for child in behaviour_tree.root.children:
        # planner = planners[j]
        planner = CozmoPlanner(ComplexGoal, keys, child.name, policy=policies[j])
        j += 1
        child.setup(0, planner, False, 5)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(ComplexGoal)
        )
        print(i, behaviour_tree.root.status)


def main():
    cozmomain()
    # robot = ComplexGoal(normal=True, goal='detect_cube')
    # robot = ComplexGoal(normal=False, goal='detect_cube')


if __name__ == '__main__':
    main()
