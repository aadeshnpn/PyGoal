'''Make Cozmo to accomplish goal specified by our GoalFramework.

The script shows an example of accomplishing complex goal using
GoalFrame work.
'''

import asyncio
import time

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

import gym
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecPropTaxi
from pygoal.utils.bt import goalspec2BT


# Task description for cozmo robot
# Random Walk.
# Finding cube
# Go to Cube
# Carry cube
# Find charging station
# Drop the cube near charging station
#

def detect_cube(robot: cozmo.robot.Robot):
    look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)

    # try to find a block
    cube = None

    try:
        cube = robot.world.wait_for_observed_light_cube(timeout=30)
        print("Found cube", cube)
    except asyncio.TimeoutError:
        print("Didn't find a cube :-(")
    finally:
        # whether we find it or not, we want to stop the behavior
        look_around.stop()
        return cube


def find_cube(robot, cube):
    try:
        action = robot.go_to_object(cube, distance_mm(75.0))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")
        return True
    except:
        return False



def carry_cube(robot, cube):
    # Carry cube
    action = robot.pickup_object(cube)
    print("got action", action)
    result = action.wait_for_completed(timeout=30)
    print("got action result", result)


def find_charger(robot):
    # see if Cozmo already knows where the charger is
    charger = None
    if robot.world.charger:
        if robot.world.charger.pose.is_comparable(robot.pose):
            print("Cozmo already knows where the charger is!")
            charger = robot.world.charger
        else:
            # Cozmo knows about the charger, but the pose is not based on the
            # same origin as the robot (e.g. the robot was moved since seeing
            # the charger) so try to look for the charger first
            pass

    if not charger:
        # Tell Cozmo to look around for the charger
        look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        try:
            charger = robot.world.wait_for_observed_charger(timeout=30)
            print("Found charger: %s" % charger)
        except asyncio.TimeoutError:
            print("Didn't see the charger")
        finally:
            # whether we find it or not, we want to stop the behavior
            look_around.stop()
            return charger


def move_to_charger(robot, charger):
    try:
        action = robot.go_to_object(charger, distance_mm(65.0))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")
        return True
    except:
        return False


def drop_cube(robot, cube):
    action = robot.place_object_on_ground_here(cube)
    print("got action", action)
    result = action.wait_for_completed(timeout=30)
    print("got action result", result)
    robot.turn_in_place(degrees(90)).wait_for_completed()


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

    #{
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

# cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on charger for now
cozmo.run_program(drive_to_charger, use_viewer=True, force_viewer_on_top=True)
