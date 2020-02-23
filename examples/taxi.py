"""Learn policy in Taxi world using GenRecProp."""

import gym
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt


def env_setup(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 3
    return env


def reset_env(env, seed=1234):
    env.seed(seed)
    env.reset()
    env.env.s = 3


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


class GenRecPropTaxi(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        temp = list(env.decode(env.s))
        return (str(temp[0])+str(temp[1]), temp[2], temp[3])

    def set_state(self, env, trace, i):
        # self.get_curr_state(self.env), trace['L'][i][-1], trace['PI'][i][-1], trace['TI'][i][-1])
        state = []
        for k in self.keys:
            if k == 'L':
                temp = trace[k][i][-1]
                state.append(int(temp[0]))
                state.append(int(temp[1]))
            else:
                temp = trace[k][i][-1]
                state.append(int(temp))
        # print('set state', state)
        state = env.encode(*tuple(state))
        # print('state encoded', state)
        env.env.s = state

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
        # action = policy[state[0]]
        action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        # ss = state[0]
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
        print('env state', self.goalspec, self.get_curr_state(self.env))
        trace = self.generator()
        # print(trace['PI'])
        # Recognizer
        result, trace = self.recognizer(trace)
        # print(result, trace['L'], trace['PI'], trace['A'])
        # No need to propagate results after exciding the train epoch
        if self.tcount <= epoch:
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1

        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        return self.run_policy(policy, self.max_trace_len)


def taxi():
    env = init_taxi(seed=1234)
    target = list(env.decode(env.s))
    print(target)
    # goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'
    goalspec = 'F P_[L]['+give_loc(target[2])+',none,==] U F P_[PI]['+str(4)+',none,==]'
    keys = ['L', 'TI', 'PI']
    actions = [[0, 1, 2, 3], [4]]

    root = goalspec2BT(goalspec, planner=None)
    # print('root', root)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    epoch = [10, 2]
    j = 0
    for child in behaviour_tree.root.children:
        # print('children', child, child.name, child.id)
        planner = GenRecPropTaxi(
             env, keys, child.name, dict(), actions=actions[j],
             max_trace=40, seed=None)
        child.setup(0, planner, True, epoch[j])
        j += 1
        # planner.env = env
        print(child.goalspec, child.planner.goalspec, child.planner.env)
    print('rootname', behaviour_tree.root.name)
    # behaviour_tree.root.remove_child_by_id(id)
    # display_bt(behaviour_tree)
    for i in range(12):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(behaviour_tree.root.status)

    # # for child in behaviour_tree.root.children:
    # child.setup(0, planner, True, 40)
    # child.train = False
    # print(child, child.name, child.train)

    # for i in range(1):
    #     behaviour_tree.tick(
    #         pre_tick_handler=reset_env(env)
    #     )
    # print('inference', behaviour_tree.root.status)
    # print(env.curr_loc)


def taxi1():
    env = init_taxi(seed=1234)
    target = list(env.decode(env.s))
    print(target)
    # goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'
    goalspec = 'F P_[PI]['+str(4)+',none,==]'
    keys = ['L', 'PI', 'DI']
    actions = [4]

    root = goalspec2BT(goalspec, planner=None)
    # print('root', root)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree)
    # epoch = [10, #2]
    # j = 0
    child = behaviour_tree.root
    # for child in behaviour_tree.root.children:
    # print('children', child, child.name, child.id)
    planner = GenRecPropTaxi(
            env, keys, child.name, dict(), actions=actions,
            max_trace=3, seed=None)
    child.setup(0, planner, True, 2)
    #    j += 1
    # planner.env = env
    print(child.goalspec, child.planner.goalspec, child.planner.env)
    print('rootname', behaviour_tree.root.name)
    # behaviour_tree.root.remove_child_by_id(id)
    # display_bt(behaviour_tree)
    for i in range(2):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(behaviour_tree.root.status)


def main():
    taxi1()


if __name__ == '__main__':
    main()
