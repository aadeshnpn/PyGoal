import numpy as np
import os
import copy
import gym
import gym_minigrid     # noqa: F401
from gym_minigrid.minigrid import (     # noqa: F401
    Grid, OBJECT_TO_IDX, Key, Door, Goal, Ball, Box, Lava,
    COLOR_TO_IDX)
# import py_trees
from py_trees.trees import BehaviourTree
from py_trees import Blackboard
from py_trees import Status
from py_trees.composites import Parallel
from pygoal.utils.bt import goalspec2BT
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper
from pygoal.lib.genrecprop import GenRecProp, GenRecPropUpdated
from pygoal.lib.bt import CompetentNode
from pygoal.utils.distribution import (
    recursive_com, logistfunc, recursive_setup,
    compare_curve)

import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402


class MultiGoalGridExp():
    def __init__(
            self, expname='key', goalspecs='F P_[KE][1,none,==]',
            keys=['LO', 'FW', 'KE'], actions=list(range(5)),
            seed=None, maxtracelen=40, trainc=False, epoch=80):
        env_name = 'MiniGrid-Goals-v0'
        env = gym.make(env_name)
        if seed is None:
            pass
        else:
            env = ReseedWrapper(env, seeds=[seed])
        env = FullyObsWrapper(env)
        self.env = env
        self.env.max_steps = min(env.max_steps, 200)
        # self.env.agent_view_size = 1
        self.env.reset()
        self.expname = expname
        self.goalspecs = goalspecs
        self.epoch = epoch
        self.maxtracelen = maxtracelen
        self.trainc = trainc
        self.allkeys = [
            'LO', 'FW', 'KE', 'DR',
            'BOB', 'BOR', 'BAB', 'BAR',
            'LV', 'GO', 'CK',
            'CBB', 'CBR', 'CAB', 'CAR',
            'DO', 'RM']
        self.keys = keys
        self.actions = actions
        root = goalspec2BT(goalspecs, planner=None, node=CompetentNode)
        self.behaviour_tree = BehaviourTree(root)
        self.blackboard = Blackboard()

    def run(self):
        def fn_c(child):
            pass

        def fn_eset(child):
            planner = GenRecPropMultiGoal(
                self.env, self.keys, child.name, dict(), actions=self.actions,
                max_trace=self.maxtracelen, seed=None, allkeys=self.allkeys)

            child.setup(0, planner, True, self.epoch)

        def fn_einf(child):
            child.train = False
            child.planner.epoch = 5
            child.planner.tcount = 0

        def fn_ecomp(child):
            child.planner.compute_competency(self.trainc)

        # Save the environment to visualize
        # self.save_data(env=True)

        # Setup planners
        recursive_setup(self.behaviour_tree.root, fn_eset, fn_c)
        # import py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.behaviour_tree.root)
        # exit()
        # print(dir(self.behaviour_tree.root.children[0]))
        # print(self.behaviour_tree.root.children[0].parent.children)
        # Train
        for i in range(150):
            self.behaviour_tree.tick(
                pre_tick_handler=self.reset_env(self.env)
            )
        # print(i, 'Training', self.behaviour_tree.root.status)

        # Inference
        recursive_setup(self.behaviour_tree.root, fn_einf, fn_c)
        for i in range(10):
            self.behaviour_tree.tick(
                pre_tick_handler=self.reset_env(self.env)
            )
        # print(i, 'Inference', self.behaviour_tree.root.status)
        # Recursive compute competency for execution nodes
        recursive_setup(self.behaviour_tree.root, fn_ecomp, fn_c)

        # Recursive compute competency for control nodes
        recursive_com(self.behaviour_tree.root, self.blackboard)

    def reset_env(self, seed=12345):
        self.env.reset()

    def save_data(self, env=False):
        # Create folder if not exists
        import pathlib
        import os
        dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        if env:
            fname = os.path.join(dname, self.expname + '_env.png')
            img = self.env.render(mode='exp')
            plt.imsave(fname, img)
        else:
            fname = os.path.join(dname, self.expname + '.pkl')
            import pickle
            pickle.dump(self.blackboard, open(fname, "wb"))

    def draw_plot(self, nodenames, root=False, train=True):
        curves = []
        datas = []
        for nname in nodenames:
            if train:
                datas.append(np.mean(
                    self.blackboard.shared_content[
                        'ctdata'][nname], axis=0))
            else:
                datas.append(np.mean(
                    self.blackboard.shared_content[
                        'cidata'][nname], axis=0))
            curves.append(self.blackboard.shared_content['curve'][nname])
        compare_curve(curves, datas, name=self.expname, root=root)


class MultiGoalGridUExp():
    def __init__(
            self, expname='key', goalspecs='F P_[KE][1,none,==]',
            keys=['LO', 'FW', 'KE'], actions=list(range(5)),
            seed=None, maxtracelen=40, trainc=False, epoch=80):
        env_name = 'MiniGrid-Goals-v0'
        env = gym.make(env_name)
        if seed is None:
            pass
        else:
            env = ReseedWrapper(env, seeds=[seed])
        env = FullyObsWrapper(env)
        self.env = env
        self.env.max_steps = min(env.max_steps, maxtracelen-3)
        # self.env.agent_view_size = 1
        self.env.reset()
        #   self.env.render()
        # import time
        # time.sleep(10)
        self.expname = expname
        self.goalspecs = goalspecs
        self.epoch = epoch
        self.maxtracelen = maxtracelen
        self.trainc = trainc
        self.allkeys = [
            'LO', 'FW', 'KE', 'DR',
            'BOB', 'BOR', 'BAB', 'BAR',
            'LV', 'GO', 'CK',
            'CBB', 'CBR', 'CAB', 'CAR',
            'DO', 'RM']
        self.keys = keys
        self.actions = actions
        root = goalspec2BT(goalspecs, planner=None, node=CompetentNode)
        self.behaviour_tree = BehaviourTree(root)
        self.blackboard = Blackboard()

    def run(self):
        def fn_c(child):
            pass

        def fn_eset(child):
            planner = GenRecPropMultiGoalU(
                self.env, self.keys, child.name, dict(), actions=self.actions,
                max_trace=self.maxtracelen, epoch=self.epoch,
                seed=None, allkeys=self.allkeys, id=child.nodename)

            child.setup(0, planner, True, self.epoch)

        def fn_einf(child):
            child.train = False
            child.planner.epoch = 5
            child.planner.tcount = 0
            # child.planner.verbose = True

        def fn_ecomp(child):
            child.planner.compute_competency(self.trainc)

        def parallel_hack(node):
            if (isinstance(node, Parallel)):
                return node
            elif node.children:
                for c in node.children:
                    if (isinstance(c, Parallel)):
                        # Control nodes
                        return c
                    parallel_hack(c)

        # Save the environment to visualize
        # self.save_data(env=True)

        # Setup planners
        recursive_setup(self.behaviour_tree.root, fn_eset, fn_c)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.behaviour_tree.root)

        # Parallel node hack
        def for_parallel_node(root):
            node = parallel_hack(root)
            if node:
                combgoal = [node.name for node in node.children]
                othernodes = []
                for i in range(len(node.children)):
                    if i == 0:
                        node.children[i].planner.list_goalspec = combgoal
                        node.children[i].planner.parallel_node = True
                    else:
                        othernodes.append(node.children[i])
                        node.remove_child(node.children[i])
                return node, othernodes
            else:
                return None, None
        pnode, othernodes = for_parallel_node(self.behaviour_tree.root)

        # combgoal = ['F(P_[KE][1,none,==])', 'G(P_[LV][0,none,==])']
        # self.behaviour_tree.root.planner.list_goalspec = combgoal
        # self.behaviour_tree.root.planner.parallel_node = True

        self.blackboard.shared_content['current'] = dict()
        for i in range(self.epoch):
            self.env.reset()
            self.blackboard.shared_content['current']['epoch'] = i
            for j in range(self.maxtracelen):
                self.behaviour_tree.tick(
                    # pre_tick_handler=self.reset_env()
                )
                # print(j, self.behaviour_tree.root.planner.gtable.keys())
                if self.behaviour_tree.root.status == Status.SUCCESS:
                    break
                if self.blackboard.shared_content['current']['done']:
                    break
            print(i, 'Training', self.behaviour_tree.root.status)

        # Inference
        recursive_setup(self.behaviour_tree.root, fn_einf, fn_c)
        for i in range(self.epoch):
            self.env.reset()
            self.blackboard.shared_content['current']['epoch'] = i
            for j in range(self.maxtracelen):
                self.behaviour_tree.tick(
                    # pre_tick_handler=self.reset_env(self.env)
                )
                if self.behaviour_tree.root.status == Status.SUCCESS:
                    break
                if self.blackboard.shared_content['current']['done']:
                    break
            print(i, 'Inference', self.behaviour_tree.root.status)
        # Recursive compute competency for execution nodes
        # print(self.behaviour_tree.root.children[0].planner.idata[0])
        # print(self.behaviour_tree.root.children[0].planner.tdata[0])
        recursive_setup(self.behaviour_tree.root, fn_ecomp, fn_c)
        self.trainc = not self.trainc
        # Recursive compute competency for execution nodes
        recursive_setup(self.behaviour_tree.root, fn_ecomp, fn_c)

        print(self.blackboard.shared_content['curve'])
        # Recursive compute competency for control nodes
        recursive_com(self.behaviour_tree.root, self.blackboard)

    def reset_env(self):
        self.env.reset()

    def check_env_done(self):

        self.reset_env()

    def save_data(self, env=False):
        # Create folder if not exists
        import pathlib
        import os
        dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        if env:
            fname = os.path.join(dname, self.expname + '_env.png')
            img = self.env.render(mode='exp')
            plt.imsave(fname, img)
        else:
            fname = os.path.join(dname, self.expname + '.pkl')
            import pickle
            pickle.dump(self.blackboard, open(fname, "wb"))

    def draw_plot(self, nodenames, root=False, train=True):
        curves = []
        datas = []
        for nname in nodenames:
            if train:
                datas.append(np.mean(
                    self.blackboard.shared_content[
                        'ctdata'][nname], axis=0))
            else:
                datas.append(np.mean(
                    self.blackboard.shared_content[
                        'cidata'][nname], axis=0))
            curves.append(
                self.blackboard.shared_content['curve'][nname][str(train)])
        compare_curve(curves, datas, name=self.expname+str(train), root=root)


class GenRecPropMultiGoal(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, allkeys=None,
            actionu=0.90, id=None):
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
        # Id to reference the blackboard
        self.id = self.goalspec if id is None else id

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
        pos = copy.copy(env.agent_pos)
        ori = copy.copy(env.agent_dir)
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
            self.allkeys[12]: str(carrybox2),
            self.allkeys[13]: str(carryball1),
            self.allkeys[14]: str(carryball2),
            self.allkeys[15]: str(door_open),
            self.allkeys[16]: str(room1)
        }
        return [generalizess[k] for k in self.keys]

    # Need to work on this
    def set_state(self, env, trace, i):
        # envobs = copy.copy(env)
        envobs = env.unwrapped
        full_grid = envobs.grid.encode()
        full_grid[envobs.agent_pos[0]][envobs.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            envobs.agent_dir
        ])

        def set_items(val, obj, itemcode, envobs, full_grid, color=2):
            if val == '0':
                pass
            elif val == '1':
                k = 0
                if env.carrying is None:
                    full_grid[:, :, 0] == itemcode
                    i, j = np.where(full_grid[:, :, 0] == itemcode)
                    if len(i) == 1:
                        item = envobs.grid.get(i[0], j[0])
                    else:
                        for k in range(len(i)):
                            if full_grid[i[k], j[k]][1] == color:
                                item = envobs.grid.get(i[k], j[k])
                                break
                    if isinstance(item, obj):
                        envobs.carrying = item
                        envobs.carrying.cur_pos = np.array([-1, -1])
                        envobs.grid.set(i[k], j[k], None)

        for k in self.keys:
            # print('set trace', k, trace[k][i])
            if k == 'LO':
                x, y, d = trace[k][i][-1]
                envobs.agent_pos = (int(x), int(y))
                envobs.agent_dir = int(d)
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
                item = envobs.grid.get(i[0], j[0])
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
        # action = self.action_uncertainty(action)
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
        # print(self.tcount, self.goalspec, result)
        # No need to propagate results after exciding the train epoch
        gkey = self.extract_key()
        if self.tcount <= epoch:
            # Update the data to compute competency
            data = self.aggrigate_data(len(trace[gkey]), result)
            self.blackboard.shared_content[
                'ctdata'][self.id].append(data)
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
                'cidata'][self.id].append(data)
            self.tcount += 1
        return result

    def aggrigate_data(self, size, result):
        data = np.zeros((self.max_trace_len+4))
        if result:
            data[:size] = np.array(
                data[size], dtype=np.float)
            data[size:] = 1.0
        return data

    def extract_key(self):
        import re
        match = re.search('\[[A-Z0-9]+\]', self.goalspec)   # noqa: W605
        gkey = match.group(0)[1:-1]
        return gkey

    def compute_competency(self, train=True):
        from scipy.optimize import curve_fit
        if train:
            data = np.mean(
                self.blackboard.shared_content[
                    'ctdata'][self.id], axis=0)
            # print(data)
            try:
                popt, pcov = curve_fit(
                    logistfunc, range(data.shape[0]), data,
                    maxfev=800)
            except RuntimeError:
                popt = np.array([0.99, 1., 1.])
        else:
            data = np.mean(
                self.blackboard.shared_content[
                    'cidata'][self.id], axis=0)
            try:
                popt, pcov = curve_fit(
                    logistfunc, range(data.shape[0]), data,
                    maxfev=800)
            except (RuntimeError, IndexError):
                popt = np.array([0.99, 1., 1.])
        self.blackboard.shared_content['curve'][self.id] = popt
        return popt


class GenRecPropMultiGoalU(GenRecPropUpdated):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, allkeys=None,
            actionu=0.90, id=None):
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
        # Id to reference the blackboard
        self.id = self.goalspec if id is None else id
        # Numpy array to hold competency data
        # print(self.epoch)
        self.tdata = np.zeros((self.epoch, max_trace+4))
        self.idata = np.zeros((self.epoch, max_trace+4))
        self.blackboard.shared_content[
            'ctdata'][self.id] = self.tdata
        self.blackboard.shared_content[
            'cidata'][self.id] = self.idata

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
        pos = copy.copy(env.agent_pos)
        ori = copy.copy(env.agent_dir)
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
            self.allkeys[12]: str(carrybox2),
            self.allkeys[13]: str(carryball1),
            self.allkeys[14]: str(carryball2),
            self.allkeys[15]: str(door_open),
            self.allkeys[16]: str(room1)
        }
        return [generalizess[k] for k in self.keys]

    # Need to work on this
    def set_state(self, env, trace, i):
        # envobs = copy.copy(env)
        envobs = env.unwrapped
        full_grid = envobs.grid.encode()
        full_grid[envobs.agent_pos[0]][envobs.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            envobs.agent_dir
        ])

        def set_items(val, obj, itemcode, envobs, full_grid, color=2):
            if val == '0':
                pass
            elif val == '1':
                k = 0
                if env.carrying is None:
                    full_grid[:, :, 0] == itemcode
                    i, j = np.where(full_grid[:, :, 0] == itemcode)
                    if len(i) == 1:
                        item = envobs.grid.get(i[0], j[0])
                    else:
                        for k in range(len(i)):
                            if full_grid[i[k], j[k]][1] == color:
                                item = envobs.grid.get(i[k], j[k])
                                break
                    if isinstance(item, obj):
                        envobs.carrying = item
                        envobs.carrying.cur_pos = np.array([-1, -1])
                        envobs.grid.set(i[k], j[k], None)

        for k in self.keys:
            # print('set trace', k, trace[k][i])
            if k == 'LO':
                x, y, d = trace[k][i][-1]
                envobs.agent_pos = (int(x), int(y))
                envobs.agent_dir = int(d)
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
                item = envobs.grid.get(i[0], j[0])
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

        # Generator
        trace = self.generator()
        # Recognizer
        # result, trace = self.recognizer(trace)
        results = self.recognizer(trace)
        result = np.all([r[0] for r in list(results.values())])
        trace = results[self.goalspec][1]
        # result = np.all(results)
        # print(result, self.goalspec, trace['KE'], trace['LV'], self.env_done)
        # No need to propagate results after exciding the train epoch
        gkey = self.extract_key()
        # Update the data to compute competency
        self.blackboard.shared_content['current'][self.id] = self.tcount
        self.blackboard.shared_content['current']['done'] = self.env_done
        self.update_data(result)
        # Progagrate the error generate from recognizer
        if (
                self.env_done or
                len(trace[gkey]) >= self.max_trace_len or
                result
                ):
            # for r in results:
            # print(r, )
            self.propagatep(results, trace)
            # print(result, self.id, trace['LV'])
            # print('trace len', len(trace[gkey]), self.tcount, self.env_done, end=' ')
            self.aggrigate_data(len(trace[gkey]), result)
            # print(self.tdata[self.blackboard.shared_content['current']['epoch']])
            self.trace = dict()
            self.blackboard.shared_content['current'][self.id] = 0
            self.tcount = 0

        # Increment the count
        self.tcount += 1
        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        result, trace = self.run_policy(
            policy, self.max_trace_len, verbose=False)
        gkey = self.extract_key()
        # print('from inference', self.tcount, self.epoch)
        self.blackboard.shared_content['current'][self.id] = self.tcount
        self.blackboard.shared_content['current']['done'] = self.env_done
        self.update_data(result, train=False)
        if (
                self.env_done or
                len(trace[gkey]) >= self.max_trace_len
                or result):
            # if self.tcount <= self.epoch:
            # print(len(trace[gkey]))
            # print(result, self.idata[self.blackboard.shared_content['current']['epoch']])
            self.aggrigate_data(len(trace[gkey]), result, train=False)
            # print(result, self.idata[self.blackboard.shared_content['current']['epoch']])
            self.itrace = dict()
            self.blackboard.shared_content['current'][self.id] = 0
            self.tcount = 0

        self.tcount += 1
        return result

    def aggrigate_data(self, indx, result, train=True):
        epoch = self.blackboard.shared_content['current']['epoch']

        def filldata(data, indx):
            if result:
                data[epoch][:indx] = np.array(
                    data[epoch][indx], dtype=np.float)
                data[epoch][indx:] = 1.0
            return data
        if train:
            self.tdata = filldata(self.tdata, indx)
        else:
            self.idata = filldata(self.idata, indx)
            # print(self.idata[epoch])

    def update_data(self, result, train=True):
        epoch = self.blackboard.shared_content['current']['epoch']
        trace = self.blackboard.shared_content['current'][self.id]
        # print('update data', epoch, trace, end=' ')
        if train:
            self.tdata[epoch][trace] = result * 1.0
        else:
            self.idata[epoch][trace] = result * 1.0

    def extract_key(self):
        import re
        match = re.search('\[[A-Z0-9]+\]', self.goalspec)   # noqa: W605
        gkey = match.group(0)[1:-1]
        return gkey

    def compute_competency(self, train=True):
        from scipy.optimize import curve_fit
        if train:
            data = np.mean(
                self.blackboard.shared_content[
                    'ctdata'][self.id], axis=0)
            # print(data)
            try:
                popt, pcov = curve_fit(
                    logistfunc, range(data.shape[0]), data,
                    maxfev=800)
            except RuntimeError:
                popt = np.array([0.99, 1., 1.])
        else:
            data = np.mean(
                self.blackboard.shared_content[
                    'cidata'][self.id], axis=0)
            try:
                popt, pcov = curve_fit(
                    logistfunc, range(data.shape[0]), data,
                    maxfev=800)
            except (RuntimeError, IndexError):
                popt = np.array([0.99, 1., 1.])
        self.blackboard.shared_content['curve'][self.id][str(train)] = popt
        return popt
