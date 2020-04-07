import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict

keys = ['S', 'I']


class EnvMNIST:
    def __init__(self, seed=123, render=False):
        self.nprandom = np.random.RandomState(seed)
        use_cuda = True
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=60000, shuffle=False, **kwargs)
        images, labels = None, None
        for image, label in self.train_loader:
            images = image
            labels = label
        idxs = []
        for i in range(10):
            idx = torch.where(labels==i)[0].data.numpy()
            # print(idx, idx.shape)
            idxs.append(idx)
        self.images = images
        self.labels = labels
        self.idxs = idxs
        self.idxs_back = idxs.copy()
        self.state = 0
        self.render = render

    def step(self, action):
        done = False
        curr_state_image = self.get_images(self.state)
        if action == 0:
            new_state = self.state - 1
        elif action == 1:
            new_state = self.state + 1
        else:
            new_state = self.state
        new_state = np.clip(new_state, 0, 9)
        self.state = new_state
        if self.state == 9:
            done = True
        if self.render == True:
            # plt.imshow(curr_state_image.view(28,28))
            # plt.show()
            pass
        return self.state, 1, None, done

    def reset(self):
        self.state = 0
        self.idxs = self.idxs_back.copy()


    def get_images(self, label):
        i = self.nprandom.choice(self.idxs[label], replace=False)
        image = self.images[i]
        return image.view(1, *image.shape)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x = F.relu(x1)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        output = F.softmax(x, dim=1)
        return output, x1


class Recognizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Recognizer, self).__init__()
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim) # .to(device)
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        out = self.sigmoid(out)
        return out


def modify_mnistnet():
    model = Generator()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    # model = torch.load("mnist_cnn.pt")
    model.fc2 = nn.Linear(128, 4)
    return model

def greedy_action(prob):
    # print(prob)
    return np.random.choice([0, 1, 2, 3], p=prob[0])

def get_current_state(env, generator):
    image = env.get_images(env.state)
    with torch.no_grad():
        _, fc = generator(image)
    return env.state, fc

def generation(generator, env):
    # print(generator)
    actions = [0, 1, 2, 3]
    action = env.nprandom.choice(actions)
    j = 0
    # states = []
    # images = []
    curr_state = get_current_state(env, generator)
    trace = create_trace_skeleton(curr_state)
    while True:
        # print(j, env.state, action)
        s, _, _, done = env.step(action)
        if done:
            break
        if j > 20:
            break
        j += 1
        # states.append(s)
        image = env.get_images(s)
        actions, fc = generator(image)
        state = get_current_state(env, generator)
        trace = trace_accumulator(trace, state)
        action = greedy_action(actions.data.numpy())
        # images.append(fc)
        # print(j, action)
    # print(actions, torch.sum(actions))
    # print(states)
    return trace

def recognition(trace):
    goalspec = 'F P_[S][9,none,==]'
    # parse the formula
    parser = LTLfGParser()

    # Define goal formula/specification
    parsed_formula = parser(goalspec)

    # Change list of trace to set
    traceset = trace.copy()
    akey = list(traceset.keys())[0]
    # print('recognizer', traceset)
    # Loop through the trace to find the shortest best trace
    for i in range(0, len(traceset[akey])):
        t = create_trace_flloat(traceset, i)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, create_trace_dict(trace, i)

    return result, create_trace_dict(trace, i)


def propogation(label, trace, generator, recoginzer, optim, error):
    input_images = trace['I']
    # print('input images',len(input_images))
    input_batch = torch.stack(input_images)
    # print('input batch size', input_batch.shape)
    output = recoginzer(input_batch)
    # print(output)
    # if output[0][-1] > 0.5:
    #     output = 1
    # else:
    #    output = 0
    # Define loss
    optim.zero_grad()
    # outputs = self.action(state)
    label = torch.tensor(label) #.to(device)
    # print(outputs.shape, label.shape)
    # print('error', type(output[0][-1]), type(label))
    loss = error(output[0][-1], label)
    loss.backward(retain_graph=True)
    # print ("epoch : %d, loss: %1.3f" %(epoch+1, loss.item()))
    optim.step()
    return loss.item()


def create_trace_skeleton(state):
    # Create a skeleton for trace
    trace = dict(zip(keys, [[list()] for i in range(len(keys))]))
    j = 0
    for k in keys:
        trace[k][0].append(state[j])
        j += 1
    return trace

def trace_accumulator(trace, state):
    for j in range(len(keys)):
        # Add the state variables to the trace
        temp = trace[keys[j]][-1].copy()
        temp.append(state[j])
        trace[keys[j]].append(temp)
    return trace

def create_trace_flloat(traceset, i):
    setslist = [create_sets(traceset[k][i]) for k in keys]
    # a = self.create_sets(traceset['A'][i])
    # setslist.append(a)
    dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
    keydictlist = dict()
    # keydictlist['A'] = dictlist[-1]
    j = 0
    for k in keys:
        keydictlist[k] = dictlist[j]
        j += 1
    t = FiniteTraceDict.fromDictSets(keydictlist)
    return t

def create_sets(trace):
    if len(trace) == 1:
        return [set(trace)]
    else:
        return [set([l]) for l in trace]

def create_trace_dict(trace, i):
    tracedict = dict()
    for k in keys:
        tracedict[k] = trace[k][i]
    # tracedict['A'] = trace['A'][i]
    return tracedict


def main():
    # Define neural nets
    generator = modify_mnistnet()
    recognizer = Recognizer(128, 100, 1, 1)
    optim = Adam(
        list(
            generator.parameters())
            + list(recognizer.parameters())
        , lr=0.003)
    error = nn.MSELoss()
    # error = nn.BCELoss()
    env = EnvMNIST(render=False)

    for epoch in range(1):
        env.reset()
        trace = generation(generator, env)
        # print(epoch, trace['S'])
        result, trace = recognition(trace)
        print(epoch, result, trace['S'])
        # Define loss, optimizer
        loss = propogation(result * 1.0, trace, generator, recognizer , optim, error)
        print(epoch, loss)


if __name__ == "__main__":
    main()