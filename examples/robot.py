import sys
import asyncio
import time

import cozmo
from cozmo.util import degrees, Pose, distance_mm, speed_mmps

from py_trees.trees import BehaviourTree
from py_trees import Blackboard
from scipy.spatial.distance import cdist

class ComplexGoal:
    def __init__(self, *a, **kw):
        self.robot = None
        # self.dictDraw = {'a':self.drawA}
        self.dict_planner = {
        'detect_cube': self.detect_cube,
        'find_cube': self.find_cube,
        'carry_cube': self.carry_cube,
        'find_charger': self.find_charger,
        'move_to_charger': self.move_to_charger,
        'drop_cube': self.drop_cube
        }
        self.blackboard = Blackboard()
        # self.blackboard.shared_content = dict()
        self.goal = kw['goal']
        self.tkeys = kw['tkeys']
        normal = kw['normal']
        if normal:
            cozmo.connect(self.runall)
        else:
            cozmo.connect(self.run)
        # self.crun = cozmo.connect()

    def run(self, sdk_conn):
        '''The run method runs once Cozmo is connected.'''
        self.robot =  sdk_conn.wait_for_robot()
        initPos = self.robot.pose
        updatePos = initPos.position
        self.dict_planner[self.goal]()


    def runall(self, sdk_conn):
        '''The run method runs once Cozmo is connected.'''
        self.robot =  sdk_conn.wait_for_robot()
        initPos = self.robot.pose
        updatePos = initPos.position
        j = 0
        result = True
        for key in self.dict_planner.keys():
            if result is True:
                print('calling ', key)
                self.dict_planner[key]()
                result = self.blackboard.shared_content['status']
                print('result', result)

    async def collect_state(self, evt, **kwargs):
        # self.states.append(self.robot.pose)
        # print(self.robot.pose)
        self.states['P'].append(self.robot.pose.position)
        dist = 1000
        if 'cube' in self.blackboard.shared_content.keys():
            self.states['DC'].append(True)
            self.robot.pose.position
            cube = self.blackboard.shared_content['cube']
            dist = cdist(self.robot.pose.position, cube.pose.position, 'euclidean')
        else:
            self.states['DC'].append(False)
        if dist < 67.0:
            self.states['DC'].append(True)
        else:
            self.states['DC'].append(False)


    def detect_cube(self):
        self.states = dict(zip(self.tkeys, [[] for i in range(len(self.tkeys))]))
        self.robot.world.add_event_handler(cozmo.robot.EvtRobotStateUpdated, self.collect_state)
        look_around = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        if 'cube' in self.blackboard.shared_content.keys():
            cube = self.blackboard.shared_content['cube']
        else:
            # try to find a block
            cube = None
        try:
            cube = self.robot.world.wait_for_observed_light_cube(timeout=30)
            print("Found cube", cube)
            self.blackboard.shared_content['cube'] = cube
            self.blackboard.shared_content['status'] = True
        except asyncio.TimeoutError:
            print("Didn't find a cube :-(")
            self.blackboard.shared_content['status'] = False
        finally:
            # whether we find it or not, we want to stop the behavior
            look_around.stop()
            # print(self.states)
            self.robot.world.remove_event_handler(cozmo.robot.EvtRobotStateUpdated, self.collect_state)
            print('from detect cube',self.states)
            self.blackboard.shared_content['states'] = self.states

    def find_cube(self):
        cube = self.blackboard.shared_content['cube']
        # print('find cube', cube)

        action = self.robot.go_to_object(cube, distance_mm(75.0))
        # print(dir(action))
        action.wait_for_completed()
        # print("Completed action: result = %s" % action)
        # print("Done.")
        # print(action.has_succeeded, action.is_completed)
        # print('result',action.result, dir(action.result))
        # if action.is_completed:
        if not action.has_failed:
            self.blackboard.shared_content['status'] = True
        else:
            self.blackboard.shared_content['status'] = False
        # return False


    def carry_cube(self):
        # Carry cube
        cube = self.blackboard.shared_content['cube']
        action = self.robot.pickup_object(cube)
        # print("got action", action)
        result = action.wait_for_completed(timeout=30)
        # print("got action result", result)

        if not action.has_failed:
            self.blackboard.shared_content['status'] = True
        else:
            self.blackboard.shared_content['status'] = False


    def find_charger(self):
        # see if Cozmo already knows where the charger is
        charger = None
        if self.robot.world.charger:
            if self.robot.world.charger.pose.is_comparable(self.robot.pose):
                print("Cozmo already knows where the charger is!")
                charger = self.robot.world.charger
                self.blackboard.shared_content['charger'] = charger
                self.blackboard.shared_content['status'] = True
            else:
                # Cozmo knows about the charger, but the pose is not based on the
                # same origin as the robot (e.g. the robot was moved since seeing
                # the charger) so try to look for the charger first
                self.blackboard.shared_content['status'] = False
                # pass

        if not charger:
            # Tell Cozmo to look around for the charger
            look_around = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            try:
                charger = self.robot.world.wait_for_observed_charger(timeout=30)
                self.blackboard.shared_content['charger'] = charger
                print("Found charger: %s" % charger)
                self.blackboard.shared_content['status'] = True
            except asyncio.TimeoutError:
                print("Didn't see the charger")
                self.blackboard.shared_content['status'] = False
            finally:
                # whether we find it or not, we want to stop the behavior
                look_around.stop()


    def move_to_charger(self):
        charger = self.blackboard.shared_content['charger']
        action = self.robot.go_to_object(charger, distance_mm(65.0))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")
        if not action.has_failed:
            self.blackboard.shared_content['status'] = True
        else:
            self.blackboard.shared_content['status'] = False


    def drop_cube(self):
        cube = self.blackboard.shared_content['cube']
        action = self.robot.place_object_on_ground_here(cube)
        print("got action", action)
        result = action.wait_for_completed(timeout=30)
        if not result.has_failed:
            print("got action result", result)
            self.robot.turn_in_place(degrees(90)).wait_for_completed()
            self.blackboard.shared_content['status'] = True
        else:
            self.blackboard.shared_content['status'] = False


# if __name__ == '__main__':
#     cozmo.setup_basic_logging()
#     ComplexGoal(normal=True, goal='detect_cube')