"""Compute Competency in a GridWorld"""
import time
import numpy as np
import gym
import gym_minigrid     # noqa: F401
from gym_minigrid.minigrid import (     # noqa: F401
    Grid, OBJECT_TO_IDX, Key, Door, Goal, Ball, Box, Lava,
    COLOR_TO_IDX)

# from utils import (
#     KeyDoorEnvironmentFactory,
#     KeyDoorPolicyNetwork, TransformerModel, Attention, Regression,
#     ValueNetwork, ppo, RegressionLoss, multinomial_likelihood
#     )

from py_trees.trees import BehaviourTree
from py_trees import Blackboard
from pygoal.utils.bt import goalspec2BT
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper
from pygoal.lib.genrecprop import GenRecProp
from pygoal.lib.bt import CompetentNode
from pygoal.utils.distribution import (
    logistfunc, compare_curve)


class GenRecPropKeyDoor(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, allkeys=None,
            actionu=0.90):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0
        self.door_history = 0
        if allkeys is None:
            self.allkeys = keys
        else:
            self.allkeys = allkeys
        # Initialize blackboard to store data
        self.blackboard = Blackboard()
        # Action uncertainty
        self.prob = actionu

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        # Things that I need to make the trace
        # # Agent location
        # # Agent direction
        # # Carrying key or not
        # # Carrying ball or not
        # # Carring box or not
        # # Room 1 or Room 2
        # # Near Lava or not
        # # Near door or not
        # # Door open or closed
        # # Goal or not
        pos = env.agent_pos
        ori = env.agent_dir

        fwd_pos = env.front_pos
        # Open door variable
        door_open = 0

        # # Key
        carrykey = 1 if isinstance(env.carrying, Key) else 0

        # # Ball
        carryball1 = 1 if (isinstance(env.carrying, Ball) and env.carrying.color == 'blue') else 0  # noqa: E501
        carryball2 = 1 if (isinstance(env.carrying, Ball) and env.carrying.color == 'red') else 0   # noqa: E501
        # # Box
        carrybox1 = 1 if (isinstance(env.carrying, Box) and env.carrying.color == 'blue') else 0    # noqa: E501
        carrybox2 = 1 if (isinstance(env.carrying, Box) and env.carrying.color == 'red') else 0     # noqa: E501

        # Room 1 or 2
        room1 = 1

        # Front item
        item = env.grid.get(fwd_pos[0], fwd_pos[1])

        # Key near
        key = 1 if isinstance(item, Key) else 0
        # Goal near
        goal = 1 if isinstance(item, Goal) else 0
        # Box near
        box1 = 1 if (isinstance(item, Box) and item.color == 'blue') else 0
        box2 = 1 if (isinstance(item, Box) and item.color == 'red') else 0
        # Ball near
        ball1 = 1 if (isinstance(item, Ball) and item.color == 'blue') else 0
        ball2 = 1 if (isinstance(item, Ball) and item.color == 'red') else 0
        # Lava near
        lava = 1 if isinstance(item, Lava) else 0

        # Door near
        if isinstance(item, Door):
            door = 1
            # print(item, item.is_open, env.carrying)
            if item.is_open:
                door_open = 1
            else:
                door_open = 0
            if self.door_history != door_open:
                room1 ^= 1  # biwise XOR to filp the integers
            self.door_history = door_open
        else:
            door = 0

        # return (
        #     str(pos[0]) + str(pos[1]) + str(ori),
        #     str(fwd_pos[0]) + str(fwd_pos[1]),
        #     str(key), str(door),
        #     str(box1), str(box2), str(ball1), str(ball2),
        #     str(lava), str(goal), str(carrykey),
        #     str(carrybox1), str(carrybox2), str(carryball1), str(carryball2),
        #     str(door_open), str(room1)
        #     )
        generalizess = {
            self.allkeys[0]: str(pos[0]) + str(pos[1]) + str(ori),
            self.allkeys[1]: str(fwd_pos[0]) + str(fwd_pos[1]),
            self.allkeys[2]: str(key), self.allkeys[3]: str(door),
            self.allkeys[4]: str(box1), self.allkeys[5]: str(box2),
            self.allkeys[6]: str(ball1), self.allkeys[7]: str(ball2),
            self.allkeys[8]: str(lava), self.allkeys[9]: str(goal),
            self.allkeys[10]: str(carrykey), self.allkeys[11]: str(carrybox1),
            self.allkeys[12]: str(carrybox2), self.allkeys[13]: str(carryball1),
            self.allkeys[14]: str(carryball2), self.allkeys[15]: str(door_open),
            self.allkeys[16]: str(room1)
        }
        return [generalizess[k] for k in self.keys]

    # Need to work on this
    def set_state(self, env, trace, i):
        envobs = env.unwrapped
        full_grid = envobs.grid.encode()
        full_grid[envobs.agent_pos[0]][envobs.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            envobs.agent_dir
        ])

        def set_items(val, obj, itemcode, env, full_grid, color=2):
            if val == '0':
                pass
            elif val == '1':
                k = 0
                if env.carrying is None:
                    full_grid[:, :, 0] == itemcode
                    i, j = np.where(full_grid[:, :, 0] == itemcode)
                    if len(i) == 1:
                        item = env.grid.get(i[0], j[0])
                    else:
                        for k in range(len(i)):
                            if full_grid[i[k], j[k]][1] == color:
                                item = env.grid.get(i[k], j[k])
                                break
                    if isinstance(item, obj):
                        env.carrying = item
                        env.carrying.cur_pos = np.array([-1, -1])
                        env.grid.set(i[k], j[k], None)

        for k in self.keys:
            # print('set trace', k, trace[k][i])
            if k == 'LO':
                x, y, d = trace[k][i][-1]
                env.agent_pos = (int(x), int(y))
                env.agent_dir = int(d)
            # elif k == 'F':
            #    fx, fy = trace[k][i][-1]
            elif k == 'DO':
                o = trace[k][i][-1]
                if o == '0':
                    status = False
                else:
                    status = True
                # full_grid[:, :, 0] == 4
                i, j = np.where(full_grid[:, :, 0] == 4)
                item = env.grid.get(i[0], j[0])
                if isinstance(item, Door):
                    item.is_open = status

            # Need to work on this function
            elif k == 'CK':
                c = trace[k][i][-1]
                set_items(c, Key, 5, env, full_grid)
            elif k == 'CBB':
                c = trace[k][i][-1]
                set_items(c, Box, 7, env, full_grid)
            elif k == 'CBR':
                c = trace[k][i][-1]
                set_items(c, Box, 7, env, full_grid, color=0)
            elif k == 'CAB':
                c = trace[k][i][-1]
                set_items(c, Ball, 6, env, full_grid)
            elif k == 'CAR':
                c = trace[k][i][-1]
                set_items(c, Ball, 6, env, full_grid, color=0)

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = [list()]
        # Hack to fix the bug of terminal state. Adding big action number
        # This makes the trace length same accross the keys
        trace['A'][0].append(9)
        return trace

    def get_action_policy(self, policy, state):
        action = policy[tuple(state)]
        action = self.action_uncertainty(action)
        return action

    def action_uncertainty(self, action):
        action_choices = {
            2: [(1-self.prob)/2, (1-self.prob)/2, self.prob],
            0: [self.prob, (1-self.prob)/2, (1-self.prob)/2],
            1: [(1-self.prob)/2, self.prob, (1-self.prob)/2]}
        return self.nprandom.choice([0, 1, 2], p=action_choices[action])

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

    def train(self, epoch, verbose=False):
        # Run the generator, recognizer loop for some epocs
        # for _ in range(epoch):

        # Generator
        trace = self.generator()
        # Recognizer
        result, trace = self.recognizer(trace)
        # No need to propagate results after exciding the train epoch
        gkey = self.extract_key()
        if self.tcount <= epoch:
            # Update the data to compute competency
            data = self.aggrigate_data(len(trace[gkey]), result)
            self.blackboard.shared_content[
                'ctdata'][self.goalspec].append(data)
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1
            # print(self.tcount, trace[gkey], result)
            # print(self.tcount, data)
        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        result, trace = self.run_policy(
            policy, self.max_trace_len, verbose=False)
        gkey = self.extract_key()
        # print('from inference', self.tcount, self.epoch)
        # print(result, trace)
        if self.tcount <= self.epoch:
            data = self.aggrigate_data(len(trace[gkey]), result)
            self.blackboard.shared_content[
                'cidata'][self.goalspec].append(data)
            self.tcount += 1
        return result

    def aggrigate_data(self, size, result):
        data = np.zeros((self.max_trace_len+3))
        if result:
            data[:size] = np.array(
                data[size], dtype=np.float)
            data[size:] = 1.0
        return data

    def extract_key(self):
        import re
        match = re.search('\[[A-Z0-9]+\]', self.goalspec)
        gkey = match.group(0)[1:-1]
        return gkey

    def compute_competency(self, train=True):
        from scipy.optimize import curve_fit
        if train:
            data = np.mean(
                self.blackboard.shared_content[
                    'ctdata'][self.goalspec], axis=0)
            # print(data)
            popt, pcov = curve_fit(
                logistfunc, range(data.shape[0]), data,
                maxfev=800)
        else:
            data = np.mean(
                self.blackboard.shared_content[
                    'cidata'][self.goalspec], axis=0)
            popt, pcov = curve_fit(
                logistfunc, range(data.shape[0]), data,
                maxfev=800)
        self.blackboard.shared_content['curve'] = popt
        return popt


def reset_env(env, seed=12345):
    # env.seed(12345)
    env.reset()


def find_key():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    state, reward, done, _ = env.step(2)
    # print(state['image'].shape, reward, done, _)
    # Find the key
    goalspec = 'F P_[KE][1,none,==]'
    # keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    allkeys = [
        'LO', 'FW', 'KE', 'DR',
        'BOB', 'BOR', 'BAB', 'BAR',
        'LV', 'GO', 'CK',
        'CBB', 'CBR', 'CAB', 'CAR',
        'DO', 'RM']

    keys = [
        'LO', 'FW', 'KE']

    actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None, node=CompetentNode)
    behaviour_tree = BehaviourTree(root)
    child = behaviour_tree.root

    planner = GenRecPropKeyDoor(
        env, keys, child.name, dict(), actions=actions,
        max_trace=40, seed=None, allkeys=allkeys)

    def run(pepoch=50, iepoch=10):
        # pepoch = 50
        child.setup(0, planner, True, pepoch)
        # Train
        for i in range(pepoch):
            behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )
        # Inference
        child.train = False
        child.planner.epoch = iepoch
        child.planner.tcount = 0
        for i in range(iepoch):
            behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )
    competency = []
    epochs = [(50, 10)] * 2
    datas = []
    for i in range(2):
        run(epochs[i][0], epochs[i][1])
        datas.append(
            np.mean(
                planner.blackboard.shared_content[
                    'ctdata'][planner.goalspec], axis=0))
        competency.append(planner.compute_competency())
    print(competency)
    compare_curve(competency, datas)
    # print(
    #     planner.compute_competency(),
    #     planner.blackboard.shared_content['curve'])


def carry_key():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    state, reward, done, _ = env.step(2)
    # print(state['image'].shape, reward, done, _)
    # Find the key
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    # keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    allkeys = [
        'LO', 'FW', 'KE', 'DR',
        'BOB', 'BOR', 'BAB', 'BAR',
        'LV', 'GO', 'CK',
        'CBB', 'CBR', 'CAB', 'CAR',
        'DO', 'RM']

    keys = [
        'LO', 'FW', 'KE', 'CK']

    actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None, node=CompetentNode)
    behaviour_tree = BehaviourTree(root)
    epoch = [50, 20]
    j = 0
    for child in behaviour_tree.root.children:
        planner = GenRecPropKeyDoor(
            env, keys, child.name, dict(), actions=actions,
            max_trace=40, seed=None, allkeys=allkeys)

        child.setup(0, planner, True, epoch[j])
        j += 1
        # print(child.goalspec, child.planner.goalspec, type(child.planner.env))
        # print(
        #     type(child), child.name, child.planner.goalspec,
        #     child.planner.blackboard.shared_content)
    # Train
    for i in range(100):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Training', behaviour_tree.root.status)

    for child in behaviour_tree.root.children:
        child.train = False
        child.planner.epoch = 5
        child.planner.tcount = 0
    # Inference

    for i in range(5):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', behaviour_tree.root.status)
    for child in behaviour_tree.root.children:
        print(
            child.name, child.planner.compute_competency(),
            )


if __name__ == "__main__":
    find_key()
    # carry_key()
