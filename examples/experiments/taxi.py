"""Experiment fom taxi problem.

Expreriments and reults for taxi problem using
GenRecProp and MAX-Q algorithm."""

import gym
import copy
import numpy as np
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecPropTaxi, GenRecProp
from pygoal.utils.bt import goalspec2BT


def env_setup(env, seed=1234):
    env.seed(seed)
    env.reset()
    return env


def reset_env(env, seed=1234):
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


# class MaxQAgent:
#     def __init__(self, env, alpha, gamma):
#         self.env = env

#         not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (non primitive actions)  # noqa: E501
#         nA = env.action_space.n + not_pr_acts
#         nS = env.observation_space.n
#         self.V = np.zeros((nA, nS))
#         self.C = np.zeros((nA, nS, nA))
#         self.V_copy = self.V.copy()

#         s = self.south = 0
#         n = self.north = 1
#         e = self.east = 2
#         w = self.west = 3
#         pickup = self.pickup = 4
#         dropoff = self.dropoff = 5
#         gotoS = self.gotoS = 6
#         gotoD = self.gotoD = 7
#         get = self.get = 8
#         put = self.put = 9
#         root = self.root = 10   # noqa: F841

#         self.graph = [
#             set(),  # south
#             set(),  # north
#             set(),  # east
#             set(),  # west
#             set(),  # pickup
#             set(),  # dropoff
#             {s, n, e, w},  # gotoSource
#             {s, n, e, w},  # gotoDestination
#             {pickup, gotoS},  # get -> pickup, gotoSource
#             {dropoff, gotoD},  # put -> dropoff, gotoDestination
#             {put, get},  # root -> put, get
#         ]

#         self.alpha = alpha
#         self.gamma = gamma
#         self.r_sum = 0
#         self.new_s = copy.copy(self.env.s)
#         self.done = False
#         self.num_of_ac = 0

#     def is_primitive(self, act):
#         if act <= 5:
#             return True
#         else:
#             return False

#     def is_terminal(self, a, done):
#         RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
#         taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
#         taxiloc = (taxirow, taxicol)
#         if done:
#             return True
#         elif a == self.root:
#             return done
#         elif a == self.put:
#             return passidx < 4
#         elif a == self.get:
#             return passidx >= 4
#         elif a == self.gotoD:
#             return passidx >= 4 and taxiloc == RGBY[destidx]
#         elif a == self.gotoS:
#             return passidx < 4 and taxiloc == RGBY[passidx]
#         elif self.is_primitive(a):
#             # just else
#             return True

#     def evaluate(self, act, s):
#         if self.is_primitive(act):
#             return self.V_copy[act, s]
#         else:
#             for j in self.graph[act]:
#                 self.V_copy[j, s] = self.evaluate(j, s)
#             Q = np.arange(0)
#             for a2 in self.graph[act]:
#                 Q = np.concatenate((Q, [self.V_copy[a2, s]]))
#             max_arg = np.argmax(Q)
#             return self.V_copy[max_arg, s]

#     # e-Greedy Approach with eps=0.001
#     def greed_act(self, act, s):
#         e = 0.001
#         Q = np.arange(0)
#         possible_a = np.arange(0)
#         for act2 in self.graph[act]:
#             if self.is_primitive(act2) or (
#                     not self.is_terminal(act2, self.done)):
#                 Q = np.concatenate(
#                     (Q, [self.V[act2, s] + self.C[act, s, act2]]))
#                 possible_a = np.concatenate((possible_a, [act2]))
#         max_arg = np.argmax(Q)
#         if np.random.rand(1) < e:
#             return np.random.choice(possible_a)
#         else:
#             return possible_a[max_arg]

#     def MAXQ_0(self, i, s):  # i is action number
#         if self.done:
#             i = 11                  # to end recursion
#         self.done = False
#         if self.is_primitive(i):
#             self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
#             if self.env.trace is not None:
#                 # Addd action to the trace
#                 try:
#                     temp = self.env.trace['A'][-1].copy()
#                     temp.append(i)
#                     self.env.trace['A'].append(temp)
#                 except (IndexError, AttributeError):
#                     self.env.trace['A'].append(i)

#                 ss = get_curr_state(self.env)
#                 self.env.trace = updateTrace(self.env.trace, ss)
#             self.r_sum += r
#             self.num_of_ac += 1
#             self.V[i, s] += self.alpha * (r - self.V[i, s])
#             return 1
#         elif i <= self.root:
#             count = 0
#             while not self.is_terminal(i, self.done):   # a is new action num
#                 a = self.greed_act(i, s)
#                 N = self.MAXQ_0(a, s)
#                 self.V_copy = self.V.copy()
#                 evaluate_res = self.evaluate(i, self.new_s)
#                 self.C[i, s, a] += self.alpha * (
#                     self.gamma ** N * evaluate_res - self.C[i, s, a])
#                 count += N
#                 s = self.new_s
#             return count

#     def reset(self, seed):
#         self.env.seed(seed)
#         self.env.reset()
#         self.r_sum = 0
#         self.num_of_ac = 0
#         self.done = False
#         self.new_s = copy.copy(self.env.s)


class MaxQTaxi(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None, alpha=0.2, gamma=1):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

        # MaxQ related init
        self.env = env

        not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (non primitive actions)  # noqa: E501
        nA = env.action_space.n + not_pr_acts
        nS = env.observation_space.n
        self.V = np.zeros((nA, nS))
        self.C = np.zeros((nA, nS, nA))
        self.V_copy = self.V.copy()

        s = self.south = 0
        n = self.north = 1
        e = self.east = 2
        w = self.west = 3
        pickup = self.pickup = 4
        dropoff = self.dropoff = 5
        gotoS = self.gotoS = 6
        gotoD = self.gotoD = 7
        get = self.get = 8
        put = self.put = 9
        root = self.root = 10   # noqa: F841
        self.trace = dict()

        self.graph = [
            set(),  # south
            set(),  # north
            set(),  # east
            set(),  # west
            set(),  # pickup
            set(),  # dropoff
            {s, n, e, w},  # gotoSource
            {s, n, e, w},  # gotoDestination
            {pickup, gotoS},  # get -> pickup, gotoSource
            {dropoff, gotoD},  # put -> dropoff, gotoDestination
            {put, get},  # root -> put, get
        ]

        self.alpha = alpha
        self.gamma = gamma
        self.r_sum = 0
        self.new_s = copy.copy(self.env.s)
        self.done = False
        self.num_of_ac = 0

    def is_primitive(self, act):
        if act <= 5:
            return True
        else:
            return False

    def is_terminal(self, a, done):
        RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
        taxiloc = (taxirow, taxicol)
        if done:
            return True
        elif a == self.root:
            return done
        elif a == self.put:
            return passidx < 4
        elif a == self.get:
            return passidx >= 4
        elif a == self.gotoD:
            return passidx >= 4 and taxiloc == RGBY[destidx]
        elif a == self.gotoS:
            return passidx < 4 and taxiloc == RGBY[passidx]
        elif self.is_primitive(a):
            # just else
            return True

    def evaluate(self, act, s):
        if self.is_primitive(act):
            return self.V_copy[act, s]
        else:
            for j in self.graph[act]:
                self.V_copy[j, s] = self.evaluate(j, s)
            Q = np.arange(0)
            for a2 in self.graph[act]:
                Q = np.concatenate((Q, [self.V_copy[a2, s]]))
            max_arg = np.argmax(Q)
            return self.V_copy[max_arg, s]

    # e-Greedy Approach with eps=0.001
    def greed_act(self, act, s):
        e = 0.001
        Q = np.arange(0)
        possible_a = np.arange(0)
        for act2 in self.graph[act]:
            if self.is_primitive(act2) or (
                    not self.is_terminal(act2, self.done)):
                Q = np.concatenate(
                    (Q, [self.V[act2, s] + self.C[act, s, act2]]))
                possible_a = np.concatenate((possible_a, [act2]))
        max_arg = np.argmax(Q)
        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    def MAXQ_0(self, i, s):  # i is action number
        def updateTrace(trace, state):
            j = 0
            for k in self.keys:
                trace[k].append(state[j])
                j += 1
            return trace
        if self.done:
            i = 11                  # to end recursion
        self.done = False
        if self.is_primitive(i):
            self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
            # Addd action to the trace
            try:
                temp = self.trace['A'][-1].copy()
                temp.append(i)
                self.trace['A'].append(temp)
            except (IndexError, AttributeError):
                self.trace['A'].append(i)

            ss = self.get_curr_state(self.env)
            self.trace = updateTrace(self.trace, ss)
            self.r_sum += r
            self.num_of_ac += 1
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            return 1
        elif i <= self.root:
            count = 0
            while not self.is_terminal(i, self.done):   # a is new action num
                a = self.greed_act(i, s)
                N = self.MAXQ_0(a, s)
                self.V_copy = self.V.copy()
                evaluate_res = self.evaluate(i, self.new_s)
                self.C[i, s, a] += self.alpha * (
                    self.gamma ** N * evaluate_res - self.C[i, s, a])
                count += N
                s = self.new_s
            return count

    def reset(self, seed):
        self.env.seed(seed)
        self.env.reset()
        self.r_sum = 0
        self.num_of_ac = 0
        self.done = False
        self.new_s = copy.copy(self.env.s)
        self.trace = None

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
        trace['A'] = [list()]
        # Hack to fix the bug of terminal state. Adding big action number
        # This makes the trace length same accross the keys
        trace['A'][0].append(9)
        return trace

    def create_trace_skeleton_maxq(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k].append(state[j])
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

    def run_policy(self, policy, max_trace_len=20, verbose=False):
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

    def maxQ(self):
        self.MAXQ_0(10, self.env.s)

    def train(self, epoch, verbose=False):
        # Run the generator, recognizer loop for some epocs
        # for _ in range(epoch):
        # Create trace skeleton
        self.trace = self.create_trace_skeleton_maxq(
            self.get_curr_state(self.env))
        if self.tcount <= epoch:
            self.maxQ()
            # Increment the count
            self.tcount += 1

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        # policy = self.get_policy()
        # return self.run_policy(policy, self.max_trace_len)
        self.reset(1234)
        self.trace = self.create_trace_skeleton_maxq(
            self.get_curr_state(self.env))
        self.MAXQ_0(10, self.env.s)
        return self.evaluate_trace(self.goalspec, self.trace)


def find_bt(goalspec):
    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree, True)
    return behaviour_tree


def run_planner(
        Planner, behaviour_tree, env, keys, actions, epoch=10, seed=None):
    # child = behaviour_tree.root
    # Training setup
    j = 0
    for child in behaviour_tree.root.children:
        planner = Planner(
             env, keys, child.name, dict(), actions=actions[0],
             max_trace=40, seed=seed)
        child.setup(0, planner, True, epoch[j])
        j += 1

    # Training loop
    for i in range(sum(epoch)):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(behaviour_tree.root.status)

    for child in behaviour_tree.root.children:
        # Inference setup
        child.train = False
        print(child, child.name, child.train)

    # Inference loop
    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print('inference', behaviour_tree.root.status)


def taxi():
    env = init_taxi(seed=1234)
    target = list(env.decode(env.s))
    print(target)
    goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'    # noqa: E501
    # goalspec = 'F P_[L]['+give_loc(target[2])+',none,==] U F P_[PI]['+str(4)+',none,==]'  # noqa: E501
    keys = ['L', 'PI', 'DI']
    actions = [[0, 1, 2, 3, 4, 5]]

    behaviour_tree = find_bt(goalspec)

    # epoch = [80, 50, 80, 50]
    epoch = [10, 10, 10, 10]
    # Run GenRecProp Planner
    # run_planner(
    #     GenRecPropTaxi, behaviour_tree, env, keys,
    #     actions, epoch=epoch)

    # Run MAX-Q Planner
    run_planner(
      MaxQTaxi, behaviour_tree, env, keys,
      actions, epoch=epoch, seed=1234)


def main():
    taxi()


if __name__ == '__main__':
    main()

