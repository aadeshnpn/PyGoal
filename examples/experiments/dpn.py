"""Experiment fom Keydoor problem with DNNs.

Expreriments and reults for KeyDoor problem using
DNNs.
"""

import gym
import copy
import numpy as np
import pickle
# from joblib import Parallel, delayed

import gym_minigrid
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX, Key, Door, Goal
from gym_minigrid.wrappers import RGBImgObsWrapper
from py_trees.trees import BehaviourTree
from py_trees import Status

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt
from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple


class State2Action(nn.Module):
    '''
    State to action function that produces actions based on a trace and current actions.

    Params:
        state_size (int)
        action_size (int)
        hidden_size (int)
    '''

    def __init__(self,
                 input_shape,
                 num_actions
                 ):
        super().__init__()

        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    # def forward(self, x):
    #    x = self.features(x).view(x.size()[0], -1)
    #    return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)


    def forward(self, x):
        '''Forward pass

        Params:
            state (List of float)
            command (List of float)

        Returns:
            FloatTensor -- action logits
        '''
        x = x.view(1, *x.shape)
        x = x/225.0
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    def action(self, state):
        '''
        Params:
            state (List of float)
            command (List of float)

        Returns:
            int -- stochastic action
        '''

        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        # print(probs)
        # dist = Categorical(probs)
        # return probs[0]
        return probs[0]
        # probs = probs[0].detach().cpu().numpy()
        # return dist.sample().item()probs[0].detach().cpu().numpy()

    def greedy_action(self, state):
        '''
        Params:
            state (List of float)
            command (List of float)

        Returns:
            int -- greedy action
        '''
        # if np.random.rand() < 0.5:
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        probs = probs.to(torch.float16)
        # print(type(probs))
        # probs = probs[0] / torch.sum(probs[0])
        # return np.argmax(probs[0].detach().cpu().numpy())
        probs = probs[0].detach().cpu().numpy()
        # print(probs, np.sum(probs))
        return np.random.choice(
            list(range(7)), p = probs)
        #else:
        #    return np.random.randint(0, 6)

    def init_optimizer(self, optim=Adam, lr=0.003):
        '''Initialize GD optimizer

        Params:
            optim (Optimizer) -- default Adam
            lr (float) -- default 0.003
        '''

        self.optim = optim(self.parameters(), lr=lr)
        self.error = nn.MSELoss()

    def feedback(self, state, label):
        self.optim.zero_grad()
        outputs = self.action(state)
        # outputs = outputs.view(1, *outputs.shape)
        label = torch.tensor(label)
        # label = label.view(1, *label.shape)
        # print(outputs, label)
        loss = self.error(outputs, label)
        loss.backward()
        self.optim.step()

    def save(self, filename):
        '''Save the model's parameters
        Param:
            filename (str)
        '''

        torch.save(self.state_dict(), filename)

    def load(self, filename):
        '''Load the model's parameters

        Params:
            filename (str)
        '''

        self.load_state_dict(torch.load(filename))


def env_setup(name='MiniGrid-Empty-8x8-v0'):
    env_name = name
    env = gym.make(env_name)
    env = RGBImgObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.seed(12345)
    env.reset()
    return env


def run_episode(env, s0):
    states = []
    # print(s0)
    while True:
        action = np.random.randint(0, 6)
        state, reward, done, info = env.step(action)
        experience = (s0, action, state, reward)
        if done:
            break


# GenRecProp algorithm
class GenRecPropKeyDoor:
    def __init__(
        self, env, keys, goalspec, gtable, max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.gtable = gtable
        self.max_trace_len = max_trace
        self.actionsidx = actions
        self.tcount = 0
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)

    def get_action(self, state):
        # return self.nprandom.choice(
        #    self.actionsidx,
        #    p=list(self.gtable[state].values())
        #    )
        # print(state[0].shape)
        return self.gtable.greedy_action(state[0])
        # self.gtable.action()

    def create_gtable_indv(self, state):
        p = np.ones(len(self.actionsidx), dtype=np.float64)
        p = p / (len(self.actionsidx) * 1.0)

        self.gtable[state] = dict(
                        zip(self.actionsidx, p))

    def trace_accumulator(self, trace, state):
        for j in range(len(self.keys)):
            # Add the state variables to the trace
            temp = trace[self.keys[j]][-1].copy()
            temp.append(state[j])
            trace[self.keys[j]].append(temp)
        return trace

    def evaluate_trace(self, goalspec, trace):
        # Test the prop algorithm
        parser = LTLfGParser()
        parsed_formula = parser(goalspec)
        # Quick fix. create_trace_float requires action to be list of list
        temp = trace['A']
        trace['A'] = [temp]
        # Create a trace compatiable with Flloat library
        t = self.create_trace_flloat(self.list_to_trace(trace), 0)
        result = parsed_formula.truth(t)
        return result

    def list_to_trace(self, trace):
        for j in range(len(self.keys)):
            temp = trace[self.keys[j]]
            trace[self.keys[j]] = [temp]

        return trace

    def create_trace_flloat(self, traceset, i):
        setslist = [self.create_sets(traceset[k][i]) for k in self.keys]
        a = self.create_sets(traceset['A'][i])
        setslist.append(a)
        dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
        keydictlist = dict()
        keydictlist['A'] = dictlist[-1]
        j = 0
        for k in self.keys:
            keydictlist[k] = dictlist[j]
            j += 1
        t = FiniteTraceDict.fromDictSets(keydictlist)
        return t

    def create_sets(self, trace):
        if len(trace) == 1:
            return [set(trace)]
        else:
            return [set([l]) for l in trace]

    def create_trace_dict(self, trace, i):
        tracedict = dict()
        for k in self.keys:
            tracedict[k] = trace[k][i]
        tracedict['A'] = trace['A'][i]
        return tracedict

    def parse_goalspec(self):
        """This class will parse goal specification
        and develope a BT structure. """
        pass

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

    def generator(self, env_reset=False):
        if env_reset:
            self.env.restart()
        state = self.get_curr_state(self.env)
        trace = self.create_trace_skeleton(state)

        done = False
        # Trace generator and accumulator
        j = 0

        while not done:
            # Explore action or exploit
            action = self.get_action(state)
            # Addd action to the trace
            try:
                temp = trace['A'][-1].copy()
                temp.append(action)
                trace['A'].append(temp)
            except IndexError:
                trace['A'].append(action)
            # Map the action to env_action
            # next_state, reward, done, info = self.env.step(
            #     self.env.env_action_dict[action])
            next_state, reward, done, info = self.env.step(
                action)
            self.env.render()
            # print(action)
            nstate = self.get_curr_state(self.env)
            trace = self.trace_accumulator(trace, nstate)
            state = nstate
            if j >= self.max_trace_len:
                break
            j += 1

        return trace

    def recognizer(self, trace):
        """Recognizer.

        Which will reconize traces from the generator system."""
        # parse the formula
        parser = LTLfGParser()

        # Define goal formula/specification
        parsed_formula = parser(self.goalspec)

        # Change list of trace to set
        traceset = trace.copy()
        akey = list(traceset.keys())[0]
        # Loop through the trace to find the shortest best trace
        for i in range(0, len(traceset[akey])):
            t = self.create_trace_flloat(traceset, i)
            result = parsed_formula.truth(t)
            if result is True:
                self.set_state(self.env, trace, i)
                return True, self.create_trace_dict(trace, i)

        return result, self.create_trace_dict(trace, i)


    def propagate(self, result, trace):
        """Propagate the error to shape the probability."""

        traces = [trace[k][::-1] for k in self.keys]
        tracea = trace['A'][::-1]
        psi = 0.9
        j = 1
        # print(traces)
        for i in range(0, len(traces[0])-1, 1):
            a = tracea[i]
            tempvals = [t[i+1] for t in traces]
            distribution = self.gtable.action(tempvals[0]).detach().numpy()
            # print(tempvals[0], a)
            # ss = self.gtable_key(tempvals)
            # try:
            #     prob = self.gtable[ss][a]
            # except KeyError:
            #     self.create_gtable_indv(self.gtable_key(ss))
            #     prob = self.gtable[ss][a]
            prob = distribution[a]
            Psi = pow(psi, j)
            j += 1
            if result is False:
                new_prob = prob - (Psi * prob)
            else:
                new_prob = prob + (Psi * prob)

            # self.gtable[ss][a] = new_prob
            distribution[a] = new_prob
            probs = np.array(distribution)
            probs = probs / probs.sum()
            # Train the neural net with this new probability
            # print(a, probs)
            # Backprop probs
            self.gtable.feedback(tempvals[0], probs)
            # self.gtable[ss] = dict(zip(self.gtable[ss].keys(), probs))
        # print(j)

    def gen_rec_prop(self, epoch=50):
        # Run the generator, recognizer loop for some epocs
        for _ in range(epoch):
            # Generator
            trace = self.generator()

            # Recognizer
            result, trace = self.recognizer(trace)

            # Progagrate the error generate from recognizer
            self.propagate(result, trace)

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        # Things that I need to make the trace
        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=env.tile_size
        )
        # return rgb_img

        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)
        if fwd_cell is not None:
            goal = fwd_cell.type == 'goal'
        else:
            goal = False
        return torch.tensor(rgb_img.reshape(
            rgb_img.shape[2], rgb_img.shape[0],
            rgb_img.shape[1])), goal

    # Need to work on this
    def set_state(self, env, trace, i):
        pass

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
        # return tuple(ss)

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
        while self.tcount <= epoch:
            # Generator
            trace = self.generator()
            # print(trace['I'][1])
            # Recognizer
            result, trace = self.recognizer(trace)
            # print(result, trace['G'])
            # No need to propagate results after exciding the train epoch
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1
            print(self.tcount)
        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        return self.run_policy(policy, self.max_trace_len)


def main():
    env = env_setup('MiniGrid-Empty-5x5-v0')
    # env = env_setup('MiniGrid-Empty-5x5-v0')

    # Find the goal
    goalspec = 'F P_[G][True,none,==]'

    keys = ['I', 'G']
    actions = [
        [0, 1, 2, 3, 4, 5, 6]
        ]

    planner = GenRecPropKeyDoor(
        env, keys, goalspec, gtable=None, actions=actions, epoch=1, max_trace=20)
    state, goal = planner.get_curr_state(env)
    model = State2Action(state.shape, 7)
    model.init_optimizer()
    planner.gtable = model
    # print(state.shape)

    # state = torch.tensor(state)
    # print(model.action(state))
    print(planner.train(0))
    # print(state.shape)


if __name__ == "__main__":
    main()