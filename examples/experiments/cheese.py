"""Experiment fom cheese problem.

Expreriments and reults for cheese problem using
GenRecProp and Q-learning algorithm."""


import numpy as np
import pickle

from joblib import Parallel, delayed

from py_trees.trees import BehaviourTree
from py_trees import Status

from pygoal.lib.mdp import GridMDP, orientations
from pygoal.lib.genrecprop import GenRecProp, GenRecPropMDP, GenRecPropMDPNear
from pygoal.utils.bt import goalspec2BT, reset_env


# Qlearning planner
class QLearning(GenRecProp):
    def __init__(
        self, env, keys, goalspec, qtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, qtable, max_trace, actions, epoch, seed)
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.qtable = qtable
        self.max_trace_len = max_trace
        self.actionsidx = actions
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)
        self.tcount = 0

        # Initlized q talbe
        for state in self.env.states:
            self.qtable[state] = dict(zip(orientations, [0, 0, 0, 0]))

    def env_action_dict(self, action):
        # action_dict = {
        #     0: (1, 0),
        #     1: (0, 1),
        #     2: (-1, 0),
        #     3: (0, -1)
        # }
        # return action_dict[action]
        return action

    def set_state(self, env, trace, i):
        state = []
        for k in self.keys:
            temp = trace[k][i][-1]
            state.append(temp)
        env.curr_loc = env.state_dict[state[0]]

    def get_curr_state(self, env):
        # env.format_state(env.curr_loc)
        curr_loc = env.curr_loc
        is_cheese = curr_loc == env.cheese
        # is_trap = curr_loc == env.trap
        # reward = env.curr_reward        # F841
        # return (env.format_state(curr_loc), is_cheese, is_trap, reward)
        # return (env.format_state(curr_loc), is_cheese, is_trap)
        return (env.format_state(curr_loc), is_cheese)

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = [list()]
        return trace

    def get_action_policy(self, policy, state):
        state = tuple(state[0])
        state = (int(state[0]), int(state[1]))
        action = policy[state]
        # print(state)
        # action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        # ss = state[0]
        ss = state
        return tuple(ss)

    # def get_policy(self):
    #     policy = dict()
    #     for s, v in self.qtable.items():
    #         elem = sorted(v.items(),  key=lambda x: x[1], reverse=True)
    #         try:
    #             policy[s] = elem[0][0]
    #             pass
    #         except IndexError:
    #             pass

    #     return policy

    def dictmax(self, d, s='key'):
        maxval = -999999999
        maxkey = None
        for key, value in d.items():
            if value > maxval:
                maxval = value
                maxkey = key
        if s == 'key':
            return maxkey
        elif s == 'val':
            return maxval

    def get_policy(self, qtable):
        for s in qtable.keys():
            qtable[s] = self.dictmax(qtable[s], s='key')

        return qtable

    def run_policy_qlearning(self, policy, max_trace_len=20):
        state = self.get_curr_state(self.env)
        try:
            action = self.get_action_policy(policy, state)
        except KeyError:
            return False
        trace = dict(zip(self.keys, [list() for k in range(len(self.keys))]))
        trace['A'] = [action]

        def updateTrace(trace, state):
            j = 0
            for k in self.keys:
                trace[k].append(state[j])
                j += 1
            return trace

        trace = updateTrace(trace, state)
        j = 0
        while True:
            # next_state, reward, done, info = self.env.step(
            #    self.env.env_action_dict[action])
            next_state, reward, done, info = self.env.step(
                self.env_action_dict(action))

            next_state = self.get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            try:
                action = self.get_action_policy(policy, state)
                trace['A'].append(action)
            # Handeling terminal state
            except KeyError:
                trace['A'].append(9)
            # Run the policy as long as the goal is not achieved or less than j
            traceset = trace.copy()

            if self.evaluate_trace(self.goalspec, traceset):
                return True
            if j > max_trace_len:
                return False
            j += 1
        return False

    def qlearning(self, epoch):
        # qtable = np.zeros((len(self.states), 4))
        # qtable = dict()
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1
        # slookup = dict(zip(self.states, range(len(self.states))))
        for e in range(epoch):
            reward = 0
            # state = (3, 1)
            state = self.env.startloc
            while True:
                if self.nprandom.uniform(0, 1) < epsilon:
                    action = self.nprandom.choice([0, 1, 2, 3])
                    action = orientations[action]
                else:
                    action = self.dictmax(self.qtable[state], s='key')
                p, s1 = zip(*self.env.T(state, action))
                p, s1 = list(p), list(s1)
                s1 = s1[np.argmax(p)]
                next_state = s1
                reward = self.env.R(next_state)
                if action is None:
                    break
                old_value = self.qtable[state][action]
                next_max = self.dictmax(self.qtable[next_state], s='val')
                new_value = (1-alpha) * old_value + alpha * (
                    reward + gamma * next_max)
                self.qtable[state][action] = new_value
                state = next_state
                if state in self.env.terminals:
                    break
        # return qtable

    def train(self, epoch, verbose=False):
        # Run the generator, recognizer loop for some epocs
        # for _ in range(epoch):

        if self.tcount <= epoch:
            self.qlearning(1)
            self.tcount += 1

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy(self.qtable.copy())
        print(self.env.to_arrows(policy))
        return self.run_policy_qlearning(policy, self.max_trace_len)


def init_mdp(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trap
    grid[0][3] = None
    grid[1][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[1][3] = -2

    mdp = GridMDP(
        grid, terminals=[(3, 3), (3, 2)], startloc=sloc)

    return mdp


def find_bt(goalspec):
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree, True)
    # print(dir(behaviour_tree))
    # # Need to udpate the planner parameters
    # child = behaviour_tree.root
    return behaviour_tree


def run_planner(planner, behaviour_tree, env, epoch=10):
    child = behaviour_tree.root
    # Training setup
    child.setup(0, planner, True, epoch)

    # Training loop
    for i in range(epoch):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(behaviour_tree.root.status)

    # for child in behaviour_tree.root.children:
    # child.setup(0, planner, True, 10)
    # Inference setup
    child.train = False
    print(child, child.name, child.train)

    # Inference loop
    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print('inference', behaviour_tree.root.status)
    print(env.curr_loc)


def find_cheese(seed):
    # Define the environment for the experiment
    goalspec = 'F P_[IC][True,none,==]'
    startpoc = (1, 3)
    env = init_mdp(startpoc)
    keys = ['L', 'IC']
    actions = [0, 1, 2, 3]
    # Find BT representation based on goal specification
    behaviour_tree = find_bt(goalspec)
    # child = behaviour_tree.root

    # Initialize GenRecProp Planner
    # for child in behaviour_tree.root.children:
    # print(child, child.name, env.curr_loc)
    gplanner = GenRecPropMDP(
        env, keys, None, dict(), actions=actions, max_trace=10)
    # child.setup(0, gplanner, True, 10)

    # Initilized Qlearning planner
    qplanner = QLearning(
        env, keys, None, dict(), actions=actions, max_trace=10, seed=12)
    # child.setup(0, qplanner, True, 10)

    # Run GenRecProp Planner
    run_planner(gplanner, behaviour_tree, env, epoch=10)

    # Run Q-learning Planner
    run_planner(qplanner, behaviour_tree, env, epoch=10)


def run_planner_complex(
        Planner, behaviour_tree, env, keys, actions, epoch=10, seed=None):
    # child = behaviour_tree.root
    # Training setup
    j = 0
    for child in behaviour_tree.root.children:
        # print(actions[j], child.name)
        planner = Planner(
             env, keys, child.name, dict(), actions=actions[0],
             max_trace=30, seed=seed)
        child.setup(0, planner, True, epoch[j])
        j += 1

    # Training loop
    for i in range(sum(epoch)):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    # print(behaviour_tree.root.status)

    for child in behaviour_tree.root.children:
        # Inference setup
        child.train = False
        # print(child, child.name, child.train)

    # Inference loop
    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    # print('inference', behaviour_tree.root.status)

    if behaviour_tree.root.status is Status.SUCCESS:
        return True
    else:
        return False


def find_cheese_back(seed, epoch):
    # Define the environment for the experiment
    goalspec = 'F P_[NC][True,none,==] U F P_[L][02,none,==]'
    startpoc = (0, 2)
    env = init_mdp(startpoc)
    keys = ['L', 'NC']
    actions = [[0, 1, 2, 3], [0, 1, 2, 3]]
    # Find BT representation based on goal specification
    behaviour_tree = find_bt(goalspec)
    # child = behaviour_tree.root

    # Initialize GenRecProp Planner
    # for child in behaviour_tree.root.children:
    # print(child, child.name, env.curr_loc)
    # gplanner = GenRecPropMDP(
    #    env, keys, None, dict(), actions=actions, max_trace=40)
    # child.setup(0, gplanner, True, 60)

    # Initilized Qlearning planner
    # qplanner = QLearning(
    #     env, keys, None, dict(), actions=actions, max_trace=20, seed=12)
    # child.setup(0, qplanner, True, 10)

    # Run GenRecProp Planner
    epochs = [epoch, epoch]
    result = []
    for i in range(50):
        status = run_planner_complex(
            GenRecPropMDPNear, behaviour_tree, env,
            keys, actions, epoch=epochs)
        result += [status*1]

    return result
    # Run Q-learning Planner
    # run_planner(qplanner, behaviour_tree, env, epoch=10)


def main():
    # find_cheese(12)
    # find_cheese_back(12)
    epoch = list(range(5, 50, 5))

    results = [Parallel(n_jobs=8)(
        delayed(find_cheese_back)(None, i) for i in epoch)]

    with open('/tmp/resultscheese.pi', 'wb') as f:
        pickle.dump((results), f, pickle.HIGHEST_PROTOCOL)


main()

