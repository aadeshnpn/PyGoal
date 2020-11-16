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
from pygoal.utils.bt import goalspec2BT
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper
from pygoal.lib.genrecprop import GenRecProp


class GenRecPropKeyDoor(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0
        self.door_history = 0

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
        carryball = 1 if isinstance(env.carrying, Ball) else 0

        # # Box
        carrybox = 1 if isinstance(env.carrying, Box) else 0

        # Room 1 or 2
        room1 = 1

        # Front item
        item = env.grid.get(fwd_pos[0], fwd_pos[1])

        # Key near
        key = 1 if isinstance(item, Key) else 0
        # Goal near
        goal = 1 if isinstance(item, Goal) else 0
        # Box near
        box = 1 if isinstance(item, Box) else 0
        # Ball near
        ball = 1 if isinstance(item, Ball) else 0
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

        return (
            str(pos[0]) + str(pos[1]) + str(ori),
            str(fwd_pos[0]) + str(fwd_pos[1]),
            str(key), str(door), str(box), str(ball), str(lava), str(goal),
            str(carrykey), str(carrybox), str(carryball),
            str(door_open), str(room1)
            )

    # Need to work on this
    def set_state(self, env, trace, i):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

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
                full_grid[:, :, 0] == 4
                i, j = np.where(full_grid['image'][:, :, 0] == 4)
                item = env.grid.get(i[0], j[0])
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
                        full_grid[:, :, 0] == 4
                        i, j = np.where(full_grid['image'][:, :, 0] == 4)
                        item = env.grid.get(i, j)
                        if isinstance(item, Key):
                            env.carrying = item
                            env.carrying.cur_pos = np.array([-1, -1])
                            env.grid.set((i, j), None)
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
        return self.run_policy(policy, self.max_trace_len, verbose=True)


def reset_env(env, seed=12345):
    # env.seed(12345)
    env.reset()


def goals():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    state, reward, done, _ = env.step(2)
    print(state['image'].shape, reward, done, _)
    print(OBJECT_TO_IDX)
    print(state['image'][3][1])
    print(state['image'][6][1])
    print(state['image'][:,:,0].shape)
    i, j = np.where(state['image'][:,:,0]==4)
    print(i[0], j[0])
    i, j = np.where(state['image'][:,:,0]==6)
    print(i, j )
    # print(np.where(state['image'][:,:,0]==9))
    # time.sleep(15)
    # print(state['image'].shape)

    agent_state = (env.agent_pos, env.agent_dir)
    # print(agent_state)
    # print(dir(env.observation_space))
    exit()
    # Find the key
    goalspec = 'F P_[KE][1,none,==]'
    # keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    keys = [
        'LO', 'FW', 'KE', 'DR',
        'BO1', 'BO2', 'BA1', 'BA2'
        'LV', 'GO', 'CK',
        'CB1', 'CB2', 'CA1', 'CA2'
        'DO', 'RM']

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


if __name__ == "__main__":
    goals()
