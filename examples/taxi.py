"""Learn policy in Taxi world using GenRecProp."""

import gym
from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt


def env_setup(env, seed=123):
    env.seed(seed)
    env.reset()
    return env


def reset_env(env, seed=123):
    env.seed(seed)
    env.reset()


def init_taxi(seed):
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    env = env_setup(env, seed)
    return env


class GenRecPropTaxi(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

    def get_curr_state(self, env):
        temp = list(env.decode(env.s))
        return (str(temp[0])+str(temp[1]), temp[2], temp[3])

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        print(state, self.keys)
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
        trace = self.generator()

        # Recognizer
        result, trace = self.recognizer(trace)

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


def give_loc(idx):
    # # (0,0) -> R -> 0
    # # (4,0) -> Y -> 2
    # # (0,4) -> G -> 1
    # # (4,3) -> B -> 3
    # # In taxi -> 4
    locdict = {0: '00', 1: '04', 2: '40', 3: '43'}
    return locdict[idx]


def taxi():
    env = init_taxi(seed=123)
    target = list(env.decode(env.s))
    # goalspec = '((((F(P_[L]['+give_loc(target[2])+',none,==])) U (F(P_[PI]['+str(4)+',none,==]))) U (F(P_[L]['+give_loc(target[3])+',none,==]))) U (F(P_[PI]['+str(target[3])+',none,==])))'
    goalspec = 'F(P_[L]['+give_loc(target[2])+',none,==])'
    keys = ['L', 'TI', 'PI']
    actions = [0, 1, 2, 3, 4, 5]
    planner = GenRecPropTaxi(
         env, keys, None, dict(), actions=actions, max_trace=40)
    root = goalspec2BT(goalspec, planner=planner)
    print(root)
    behaviour_tree = BehaviourTree(root)
    display_bt(behaviour_tree)
    for child in behaviour_tree.root.children:
        print(child, child.name)
        child.setup(0, planner, True, 150)
        # planner.env = env
        # print(child.goalspec, child.planner.goalspec, child.planner.env)

    for i in range(100):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
        print(behaviour_tree.root.status)


def main():
    taxi()


if __name__ == '__main__':
    main()
