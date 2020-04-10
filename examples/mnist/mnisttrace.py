import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
from torchviz import make_dot

import numpy as np
import matplotlib.pyplot as plt

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict
import pickle

keys = ['S', 'I']

# torch.manual_seed(none)
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.__dict__.update(locals())

    def forward(self, input, target, numbers):
        valid, invalid = numbers
        # total = valid + invalid
        loss = F.binary_cross_entropy_with_logits(input, target)
        totloss = invalid * loss
        return loss


class EnvMNIST:
    def __init__(self, seed=2, render=False):
        self.nprandom = np.random.RandomState(None)
        use_cuda = True
        kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
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
        new_state = np.clip(new_state, 0, 5)
        self.state = new_state
        if self.state == 5:
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
        # self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x1 = torch.tanh(x1) #F.tanh(x1)
        # x1 = self.bn(x1)
        # x = self.dropout2(x1)
        x = self.fc2(x1)
        # output = F.log_softmax(x, dim=1)
        output = F.softmax(x, dim=1)
        return output, x1


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size= 132, #128,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=False,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        out = self.sigmoid(out)
        return out


class Recognizer1(nn.Module):
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
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(128, 4)
    model.fc2.requires_grad = True
    return model


def greedy_action(prob, nprandom):
    # print(prob)
    return nprandom.choice([0, 1, 2, 3], p=prob[0])
    # return np.argmax(prob)

def get_current_state(env, generator):
    image = env.get_images(env.state)
    # with torch.no_grad():
    actions, fc = generator(image)
    # return env.state, fc
    # print(fc.shape, actions.shape)
    return env.state, torch.cat([fc, actions], dim=1)

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
        j += 1
        # states.append(s)
        # print(image.shape)
        image = env.get_images(s)
        actions, fc = generator(image)
        state = get_current_state(env, generator)
        trace = trace_accumulator(trace, state)
        action = greedy_action(actions.data.numpy(), env.nprandom)
        if done:
            break
        if j > 15:
            break
        # images.append(fc)
        # print(j, action)
    # print(actions, torch.sum(actions))
    # print(states)
    return trace

def recognition(trace):
    goalspec = 'F P_[S][5,none,==]'
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


def propogation(label, trace, generator, recoginzer, optim, error, numbers):
    input_images = trace
    # print('input images',len(input_images))
    # print(input_images[0].requires_grad)
    input_batch = torch.stack(input_images)
    # input_batch = torch.view()
    # print('rrn input', input_batch.shape)
    # print('input batch size', input_batch.shape)
    # print(input_batch.shape, input_batch.requires_grad)
    # exit()
    output = recoginzer(input_batch)
    # dot1 = make_dot(
    #     input_batch, params=dict(generator.named_parameters()))

    # dot2 = make_dot(
    #     output, params=dict(recoginzer.named_parameters()))

    # dot1.render('/tmp/generator.png', view=True)
    # dot2.render('/tmp/recognizer.png', view=True)

    # exit()
    # print(output)
    # if output[0][-1] > 0.5:
    #     output = 1
    # else:
    #    output = 0
    # Define loss
    optim.zero_grad()
    # outputs = self.action(state)
    # label = torch.tensor(label) #.to(device)
    label = torch.ones(output.shape[0]) * label
    label = label.view(output.shape[0], 1)
    # print(outputs.shape, label.shape)
    # print('error', type(output[0][-1]), type(label))
    # print('output', output.shape, label.shape, output, label)
    loss = error(output, label, numbers)
    # loss.backward(retain_graph=True)
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
    recognizer = Recognizer(128, 20, 1, 1)
    optim = SGD(
        list(
            generator.parameters())
            + list(recognizer.parameters())
        , lr=0.003)
    # error = nn.MSELoss()
    error = nn.BCELoss()
    env = EnvMNIST(render=False)

    # Store all the good traces that recognizer validates
    valid_traces = []
    print('Training Time')
    for epoch in range(1000):
        env.reset()
        trace = generation(generator, env)
        # print(epoch, trace['S'])
        result, trace = recognition(trace)
        if result:
            valid_traces.append(trace)
        # print(epoch, result, trace['S'])
        # Define loss, optimizer
        loss = propogation(result * 1.0, trace, generator, recognizer , optim, error)
        # print(epoch, result, trace['S'])
        # print(epoch, loss)

    print('valid trace', len(valid_traces))

    # Train valid traces for 50 epochs:
    print('Training only valid traces')
    for epoch in range(1000):
        for trace in valid_traces:
            loss = propogation(1.0, trace, generator, recognizer, optim, error)

    # Inference
    print('inference time')
    for i in range(10):
        env.reset()
        j = 0
        states = []
        act = []
        while True:
            image = env.get_images(env.state)
            actions, fc = generator(image)
            action = np.argmax(actions.detach().numpy())
            # print(env.state, action)
            s, _, _, done = env.step(action)
            states.append(env.state)
            act.append(action)
            if done:
                break
            if j > 20:
                break
            j += 1
        print(i, states, act)
        # trace = generation(generator, env)
        # print(trace['S'][-1])

    # Save both the recognizer and generator
    torch.save(generator.state_dict(), "generator.pt")
    torch.save(recognizer.state_dict(), "recognizer.pt")


def create_valid_traces(env, generator, n=100):
    valid_traces = []
    for i in range(n):
        env.reset()
        curr_state = get_current_state(env, generator)
        trace = create_trace_skeleton(curr_state)
        j = 0
        while True:
            # image = env.get_images(env.state)
            # actions, fc = generator(image)
            # We know the valid action is to move right
            action = 1
            s, _, _, done = env.step(action)
            state = get_current_state(env, generator)
            trace = trace_accumulator(trace, state)
            if done:
                break
            if j > 10:
                break
            j += 1
        valid_traces.append(trace)

    return valid_traces


def create_traces(env, generator):
    valid_trace = []
    invalid_trace = []
    for epoch in range(1000):
        print(epoch)
        env.reset()
        trace = generation(generator, env)
        result, trace = recognition(trace)
        if result:
            valid_trace.append(trace)
        else:
            invalid_trace.append(trace)

        # if len(invalid_trace) >= 100:
        #    break

    return valid_trace, invalid_trace


def recognizer_performance():
    ## Need to check if the trained recognizer is a good automaton
    ## Manually create valid and invalid traces and evaluate the recognizer
    # Creating valid traces
    pass


def test():
    env = EnvMNIST(render=False)
    generator = modify_mnistnet()
    while True:
        s,_,_,done = env.step(1)
        print(get_current_state(env, generator))
        # print(s)
        if done:
            break

def train_hardcoded():
    # Define neural nets
    generator = modify_mnistnet()
    recognizer = Recognizer()
    optimvalid = Adam(
        recognizer.parameters(),
        lr=0.003)

    optiminvalid = Adam(
        list(generator.parameters()) + list(recognizer.parameters()),
        lr=0.003)

    # error = nn.BCELoss()
    error = TotalLoss()
    env = EnvMNIST(render=False)
    vloss = []
    iloss = []
    # for epoch in range(15):
    valid_traces, invalid_traces = create_traces(env, generator)
    pickle.dump((valid_traces,invalid_traces), open( "tracestest.pk", "wb" ))
        # print(epoch, 'valid','invalid', len(valid_traces), len(invalid_traces))
        # numbers = (len(valid_traces), len(invalid_traces))
        # if len(valid_traces) >= 1:
        #     for v in range(5):
        #         # Get valid traces
        #         # valid_traces = create_valid_traces(env, generator)
        #         # print(valid_traces[0]['I'][-1])
        #         losses = 0
        #         for trace in valid_traces:
        #             trace = trace['I']
        #             loss = propogation(1.0, trace, generator, recognizer, optimvalid, error, numbers)
        #             losses += loss
        #         print(epoch, v, 'valid', np.mean(losses))
        #         vloss.append(np.mean(losses))

        # for i in range(1):
        #     # Gen invalid traces
        #     losses = 0
        #     # invalid_traces = create_invalid_traces(env, generator)
        #     # invalid_traces = invalid_traces[:len(valid_traces)]
        #     for trace in invalid_traces:
        #         trace = trace['I']
        #         # print(trace)
        #         loss = propogation(0.0, trace, generator, recognizer, optiminvalid, error, numbers)
        #         losses += loss
        #     print(epoch, i, 'invalid', np.mean(losses))
        #     iloss.append(np.mean(losses))
            # print(epoch, 'invalid', np.mean(losses))

    # Save both the recognizer and generator
    # torch.save(generator.state_dict(), "generatormain.pt")
    # torch.save(recognizer.state_dict(), "recognizermain.pt")

    # # Plot the loss curves
    # plt.plot(vloss, 'g')
    # plt.plot(iloss, 'r')
    # plt.show()


if __name__ == "__main__":
    # main()
    # test()
    train_hardcoded()