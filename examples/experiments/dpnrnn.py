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

# device='cpu'

input_dim = 512
hidden_dim = 128
layer_dim = 4  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 2


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
        # out = self.softmax(out)
        # out.size() --> 100, 10
        return out.reshape(out.shape[0], out.shape[2]) #, hn


def main():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Embeddings 512
    model.fc = nn.Linear(num_ftrs, 64)
    # Five images stack together
    images = torch.rand(5, 3, 40, 40)

    # print(embeddings.shape)
    input_size = 64
    batch_size = 5
    seq_len = 1
    hidden_size = 5
    num_layers = 1
    embeddings = model(images)
    embeddings = embeddings.view(embeddings.shape[0], 1, embeddings.shape[1])
    # embeddings = embeddings.view(1, *embeddings.shape)
    # print(embeddings.shape)
    # pass the embeddings to RNN
    rnn = RNNModel(64, 5, 50, 4)
    actions = rnn(embeddings)
    # actions = actions.view(actions.shape[0], actions.shape[2])
    # print('actions', actions, actions.shape)
    target = torch.ones(5, 4) / 4.0

    # print('target', target, target.shape)

    criterion = torch.nn.MSELoss(reduction='sum')
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(rnn.parameters()), lr=0.1)

    # Train the model
    for epoch in range(100):
        loss = 0
        embeddings = model(images)
        embeddings = embeddings.view(embeddings.shape[0], 1, embeddings.shape[1])
        outputs = rnn(embeddings)
        optimizer.zero_grad()
        # print(outputs, outputs.shape)
        loss = criterion(outputs, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        print ("epoch : %d, loss: %1.3f" %(epoch+1, loss.item()))

    print ("Learning finished!")
    print('actions',rnn(embeddings))


if __name__ == "__main__":
    main()