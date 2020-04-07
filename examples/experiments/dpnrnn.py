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
import torchvision
from torchvision import datasets, models, transforms


class State2ActionRnn:
    def __init__(self, resnet, rnn):
        self.resnet = resnet
        self.rnn = rnn

    def forward(self, x):
        # x = x.view(1, *x.shape)
        x = x/225.0
        # print(x.shape)
        embeddings = self.resnet(x)
        embeddings = embeddings.view(
            embeddings.shape[0], 1, embeddings.shape[1])
        actions = self.rnn(embeddings)
        return actions

    def action(self, state):
        probs = self.forward(state)
        # probs = F.softmax(logits, dim=-1)
        return probs # [:state.size(0)]

    def greedy_action(self, state):
        state = state.view(1, *state.shape)
        # print(state.shape, state)
        probs = self.forward(state)
        # probs = F.softmax(logits, dim=-1)
        probs = probs.to(torch.float16)
        probs = probs[0].detach().cpu().numpy()
        return np.random.choice(
            list(range(4)), p = probs)

    def init_optimizer(self, optim=Adam, lr=0.003):
        self.optim = optim(
            list(
                self.resnet.parameters())
                + list(self.rnn.parameters())
            , lr=lr)
        self.error = nn.MSELoss(reduction='sum')

    def feedback(self, state, label):
        self.optim.zero_grad()
        outputs = self.action(state)
        label = torch.tensor(label) #.to(device)
        # print(outputs.shape, label.shape)
        loss = self.error(outputs, label)
        loss.backward(retain_graph=True)
        # print ("epoch : %d, loss: %1.3f" %(epoch+1, loss.item()))
        self.optim.step()
        return loss.item()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model = self.load_state_dict(torch.load(filename))



class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        # self.fc = nn.Linear(hidden_dim, output_dim)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(2)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) # .to(device)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) # .to(device)

        # One time step
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # print('out', out.shape, hn.shape)
        # out = self.fc1(out[:, 0, :])
        # out = self.fc1(out)
        out = self.fc(out)
        # print(out.shape)
        out = self.softmax(out)
        # out.size() --> 100, 10
        return out.reshape(out.shape[0], out.shape[2]) #, hn


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
        return self.gtable.greedy_action(state[0])

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

    def run_policy(self, max_trace_len=20, verbose=False):
        state = self.get_curr_state(self.env)
        # action = self.get_action_policy(self.gtable, state)
        action = self.get_action(state)

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
            self.env.render()
            next_state, reward, done, info = self.env.step(
                action)

            next_state = self.get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            try:
                action = self.get_action(state)
                # action = self.get_action_policy(self.gtable, state)
                trace['A'].append(action)
            # Handeling terminal state
            except KeyError:
                trace['A'].append(9)
            # Run the policy as long as the goal is not achieved or less than j
            traceset = trace.copy()
            if self.evaluate_trace(self.goalspec, traceset):
                print(traceset['G'])
                return True
            if j > max_trace_len:
                return False
            j += 1
        return False

    def generator(self, env_reset=False):
        self.env.env.reset()
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
            # self.env.render()
            # print(action)
            nstate = self.get_curr_state(self.env)
            trace = self.trace_accumulator(trace, nstate)
            state = nstate
            if j >= self.max_trace_len or nstate[1] is True:
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
        # print('recognizer',traceset)
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
        images = torch.stack(traces[0])
        # self.gtable.feedback(images, result)

        psi = 0.9
        j = 1
        # print(traces)
        targets = [torch.ones(4) / 4.0 ]
        # targets = []
        for i in range(0, len(traces[0])-1, 1):
            a = tracea[i]
            tempvals = [t[i+1] for t in traces]
            tempvals = tempvals[0].reshape(1, *tempvals[0].shape)
            # print(tempvals.shape)
            distribution = self.gtable.action(tempvals).detach().numpy()[0]
            # print(distribution)
            prob = distribution[a]
            Psi = pow(psi, j)
            j += 1
            if result is False:
                new_prob = prob - (Psi * prob)
            else:
                new_prob = prob + (Psi * prob)

            distribution[a] = new_prob
            probs = np.array(distribution)
            probs = probs / probs.sum()
            # Train the neural net with this new probability
            # Backprop probs
            # self.gtable.feedback(tempvals[0], probs)
            probs = torch.tensor(probs)
            targets.append(probs)

        targets = torch.stack(targets)
        # print(images.shape, targets.shape)
        self.gtable.feedback(images, targets)

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
        # action = policy[tuple(state)]
        action = self.gtable.action(state[0])
        return np.argmax(action.detach().cpu().numpy())

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
            # print(self.tcount, trace['G'][-1])
            # Recognizer
            result, trace = self.recognizer(trace)
            # print(result, trace['G'])
            # No need to propagate results after exciding the train epoch
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1
            print(self.tcount, result)
            # print(trace['A'])
            # print(trace['I'][1]
        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        return self.run_policy(policy, self.max_trace_len)


def main():
    env = env_setup('MiniGrid-Empty-8x8-v0')
    # Find the goal
    goalspec = 'F P_[G][True,none,==]'
    keys = ['I', 'G']
    actions = [
        # [0, 1, 2, 3, 4, 5, 6]
        [0, 1, 2, 3]
        ]

    planner = GenRecPropKeyDoor(
        env, keys, goalspec, gtable=None, actions=actions, epoch=500, max_trace=40)
    state, goal = planner.get_curr_state(env)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 64)
    batch_size = 1
    # Five images stack together
    images = torch.rand(batch_size, 3, 40, 40)

    rnn = RNNModel(64, batch_size, 50, 4)
    algo = State2ActionRnn(model, rnn)
    algo.init_optimizer()
    planner.gtable = algo
    planner.train(5)

    env.env.reset()
    planner.run_policy(max_trace_len=50)
    # target = torch.ones(batch_size, 4) / 4.0
    # # Train the model
    # for epoch in range(40):
    #     # loss = 0
    #     # outputs = algo.forward(images)
    #     # batch_size = np.random.randint(1, 10)
    #     images = torch.rand(batch_size, 3, 40, 40)
    #     loss = algo.feedback(images, target)
    #     print ("epoch : %d, loss: %1.3f" %(epoch+1, loss))

    # print ("Learning finished!")
    # imag1 = torch.rand(10, 3, 40, 40)
    # print('actions', algo.action(images)[0])



if __name__ == "__main__":
    main()
