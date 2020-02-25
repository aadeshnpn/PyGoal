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


def detect_cube(robot):
    # def detect_cube():
    blackboard = Blackboard()
    # robot = blackboard.shared_content['robot']
    look_around = robot.start_behavior(
        cozmo.behavior.BehaviorTypes.LookAroundInPlace)

    if 'cube' in blackboard.shared_content.keys():
        cube = blackboard.shared_content['cube']
    else:
        # try to find a block
        cube = None

    try:
        cube = robot.world.wait_for_observed_light_cube(timeout=30)
        print("Found cube", cube)
        blackboard.shared_content['cube'] = cube
        blackboard.shared_content['status'] = True
    except asyncio.TimeoutError:
        print("Didn't find a cube :-(")
        blackboard.shared_content['status'] = False
    finally:
        # whether we find it or not, we want to stop the behavior
        look_around.stop()


def find_cube(robot):
    # def find_cube():
    blackboard = Blackboard()
    cube = blackboard.shared_content['cube']
    print('find cube', cube)
    try:
        action = robot.go_to_object(cube, distance_mm(75.0))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")
        blackboard.shared_content['status'] = True
        # return True
    except:     # noqa: E722
        blackboard.shared_content['status'] = False
        # return False


def carry_cube(robot):
    # Carry cube
    blackboard = Blackboard()
    cube = blackboard.shared_content['cube']
    try:
        action = robot.pickup_object(cube)
        print("got action", action)
        result = action.wait_for_completed(timeout=30)
        print("got action result", result)
        blackboard.shared_content['status'] = True
    except:     # noqa: E722
        blackboard.shared_content['status'] = False


def find_charger(robot):
    # see if Cozmo already knows where the charger is
    blackboard = Blackboard()
    charger = None
    if robot.world.charger:
        if robot.world.charger.pose.is_comparable(robot.pose):
            print("Cozmo already knows where the charger is!")
            charger = robot.world.charger
            blackboard.shared_content['charger'] = charger
            blackboard.shared_content['status'] = True
        else:
            # Cozmo knows about the charger, but the pose is not based on the
            # same origin as the robot (e.g. the robot was moved since seeing
            # the charger) so try to look for the charger first
            blackboard.shared_content['status'] = False
            pass

    if not charger:
        # Tell Cozmo to look around for the charger
        look_around = robot.start_behavior(
            cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        try:
            charger = robot.world.wait_for_observed_charger(timeout=30)
            blackboard.shared_content['charger'] = charger
            print("Found charger: %s" % charger)
            blackboard.shared_content['status'] = True
        except asyncio.TimeoutError:
            print("Didn't see the charger")
            blackboard.shared_content['status'] = False
        finally:
            # whether we find it or not, we want to stop the behavior
            look_around.stop()


def move_to_charger(robot):
    blackboard = Blackboard()
    charger = blackboard.shared_content['charger']
    try:
        action = robot.go_to_object(charger, distance_mm(65.0))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")
        blackboard.shared_content['status'] = True
    except:     # noqa: E722
        blackboard.shared_content['status'] = False


def drop_cube(robot):
    blackboard = Blackboard()
    cube = blackboard.shared_content['cube']
    try:
        action = robot.place_object_on_ground_here(cube)
        print("got action", action)
        result = action.wait_for_completed(timeout=30)
        print("got action result", result)
        robot.turn_in_place(degrees(90)).wait_for_completed()
        blackboard.shared_content['status'] = True
    except:     # noqa: E722
        blackboard.shared_content['status'] = False


def drive_to_charger(robot):
    '''The core of the drive_to_charger program'''
    # Need to reset robot every time its called
    print('Detect cube')
    # methods = dir(robot)
    # for method in methods:
    #     print(method)
    cube = detect_cube(robot)
    print('cube', cube)
    if cube is not None:
        find_cube(robot, cube)
        carry_cube(robot, cube)
        charger = find_charger(robot)
        move_to_charger(robot, charger)
        drop_cube(robot, cube)

    # {
    # print(robot, dir(robot))
    # methods = dir(robot)
    # for method in methods:
    #     print(method)
    # print('accleromete',robot.acclerometer)
    # print('is_carrying_block',robot.is_carrying_block)
    # print('pose', robot.pose)
    # robot.pose_angle, robot.pose_pitch)

    # if charger:
    #     # Attempt to drive near to the charger, and then stop.
    #     action = robot.go_to_object(charger, distance_mm(65.0))
    #     action.wait_for_completed()
    #     print("Completed action: result = %s" % action)
    #     print("Done.")
    #     #}

# cozmo.robot.Robot.drive_off_charger_on_connect = False
# # Cozmo can stay on charger for now


# cozmo.run_program(
# drive_to_charger, use_viewer=True, force_viewer_on_top=True)

def cozmo_reset():
    pass


def reset_env(robot):
    blackboard = Blackboard()
    blackboard.shared_content['robot'] = robot


def cozmomain():
    crobot = cozmo.robot.Robot
    print(crobot.pose)
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


def main():
    cozmomain()
    # conn = cozmo.conn.CozmoConnection()
    # print(conn)

# cozmo.connect(run)


# async def run(self, coz_conn):
#     asyncio.set_event_loop(coz_conn._loop)
#     coz = await coz_conn.wait_for_robot()

#     asyncio.ensure_future(self.update())
#     while not self.exit_flag:
#         await asyncio.sleep(0)
#     coz.abort_all_actions()


if __name__ == '__main__':
    main()
