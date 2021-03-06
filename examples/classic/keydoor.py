"""Learn policy in KeyDoor world using GenRecProp."""
import gym
import gym_minigrid
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX, Key, Door, Goal

from py_trees.trees import BehaviourTree

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt


def env_setup(env, agent=(None, None, None, None)):
    env_name = 'MiniGrid-DoorKey-8x8-v0'
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    env.seed(12345)
    env.reset()
    return env


def reset_env(env, seed=12345):
    env.seed(12345)
    env.reset()


class GenRecPropKeyDoor(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        # Things that I need to make the trace
        # # Agent location
        # # Caryying 0 or 1
        # # Key or not
        # # Door or not
        # # Goal patch or not
        # # State (x, y, carry, key, door, goal, reward)
        # # Location
        pos = env.agent_pos
        ori = env.agent_dir

        fwd_pos = env.front_pos
        # Open door variable
        door_open = 0

        # # Carrying
        # if env.carrying is None or type(env.carrying).__name__ != 'Key':
        if isinstance(env.carrying, Key):
            carry = 1
        else:
            carry = 0

        # Key capture
        item = env.grid.get(fwd_pos[0], fwd_pos[1])
        if isinstance(item, Key):
            key = 1
        else:
            key = 0

        # Door capture
        if isinstance(item, Door):
            door = 1
            # print(item, item.is_open, env.carrying)
            if item.is_open:
                door_open = 1
            else:
                door_open = 0
        else:
            door = 0

        # Goal
        if isinstance(item, Goal):
            goal = 1
        else:
            goal = 0

        return (
            str(pos[0]) + str(pos[1]) + str(ori),
            str(fwd_pos[0]) + str(fwd_pos[1]), str(key), str(door),
            str(carry), str(goal), str(door_open)
            )

    # Need to work on this
    def set_state(self, env, trace, i):
        for k in self.keys:
            # print('set trace', k, trace[k][i])
            if k == 'L':
                x, y, d = trace[k][i][-1]
                env.agent_pos = (int(x), int(y))
                env.agent_dir = int(d)
            # elif k == 'F':
            #    fx, fy = trace[k][i][-1]
            elif k == 'O':
                o = trace[k][i][-1]
                if o == '0':
                    status = False
                else:
                    status = True
                for i in range(1, 7):
                    for j in range(1, 7):
                        item = env.grid.get(i, j)
                        if isinstance(item, Door):
                            item.is_open = status

            # Need to work on this function
            elif k == 'C':
                c = trace[k][i][-1]
                if c == '0':
                    status = False
                elif c == '1':
                    status = True
                    # print('set state', env.carrying)
                    if env.carrying is None:
                        for i in range(1, 7):
                            for j in range(1, 7):
                                item = env.grid.get(i, j)
                                if isinstance(item, Key):
                                    env.carrying = item
                    # print('set state', env.carrying)

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


def keydoor():
    env_name = 'MiniGrid-DoorKey-8x8-v0'
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    env.seed(12345)
    env.reset()
    env = env_setup(env)

    state = (env.agent_pos, env.agent_dir)
    print(state)
    # Find the key
    goalspec = 'F P_[K][1,none,==]'
    keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    child = behaviour_tree.root

    planner = GenRecPropKeyDoor(
        env, keys, child.name, dict(), actions=actions,
        max_trace=40, seed=None)

    child.setup(0, planner, True, 100)
    print(child.goalspec, child.planner.goalspec, type(child.planner.env))
    # Train
    for i in range(50):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Training', behaviour_tree.root.status)

    child.train = False

    # Inference
    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', behaviour_tree.root.status)


def keydoor1():
    env_name = 'MiniGrid-DoorKey-8x8-v0'
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    env.seed(12345)
    env.reset()
    env = env_setup(env)

    state = (env.agent_pos, env.agent_dir)
    print(state)
    # Find the key and carry it
    goalspec = 'F(P_[K][1,none,==]) U F P_[C][1,none,==]'

    keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    epoch = [60, 20]
    j = 0
    for child in behaviour_tree.root.children:
        planner = GenRecPropKeyDoor(
            env, keys, child.name, dict(), actions=actions,
            max_trace=40, seed=None)
        child.setup(0, planner, True, epoch[j])
        j += 1
        print(child.goalspec, child.planner.goalspec, type(child.planner.env))
    for i in range(70):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Training', behaviour_tree.root.status)

    # Inference
    for child in behaviour_tree.root.children:
        child.train = False

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', behaviour_tree.root.status)


def keydoor2():
    env_name = 'MiniGrid-DoorKey-8x8-v0'
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    env.seed(12345)
    env.reset()
    env = env_setup(env)

    state = (env.agent_pos, env.agent_dir)
    print(state)
    # Find the key and carry it
    goalspec = '(F(P_[K][1,none,==]) U F(P_[C][1,none,==])) U (F(P_[D][1,none,==]))'    # noqa: E501
    # goalspec = '(F(P_[K][1,none,==]) U F(P_[C][1,none,==]))'    # noqa: E501
    keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    actions = [[0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3]]

    root = goalspec2BT(goalspec, planner=None)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree, save=True)
    epoch = [70, 40, 70]
    j = 0
    for child in behaviour_tree.root.children:
        planner = GenRecPropKeyDoor(
            env, keys, child.name, dict(), actions=actions[j],
            max_trace=40, seed=None)
        child.setup(0, planner, True, epoch[j])
        j += 1
        print(child.goalspec, child.planner.goalspec, type(child.planner.env))
    for i in range(150):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
        print(i, 'Training', behaviour_tree.root.status)

    # Inference
    for child in behaviour_tree.root.children:
        child.train = False

    for i in range(2):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', behaviour_tree.root.status)


def main():
    keydoor2()


if __name__ == '__main__':
    main()
