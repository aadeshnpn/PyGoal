import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
# from torch.optim import Adam, SGD
# from torchviz import make_dot

from torch.utils.data import Dataset
# from torchvision import transforms, utils, datasets
import numpy as np
import pickle
import os


# import matplotlib.pyplot as plt

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict

device = 'cpu'

# Hyper-parameters
keys = ['S', 'I']

sequence_length = 128  # 1
input_size = 128
hidden_size = 10
num_layers = 1
num_classes = 2
batch_size = 4
num_epochs = 1
learning_rate = 0.01


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.error = nn.MSELoss()

    def forward(self, action, label):
        actions = action.clone().detach()
        # print('generator loss', actions.shape, actions[0].sum())
        # print(label)
        if label.item() == 1:
            val, argmax = actions.max(-1)
            actions[range(actions.shape[0]), argmax] = val + val * 0.25
            actions = actions.T / actions.sum(1)
            actions = actions.T
            # print(action, actions)
        else:
            val, argmax = actions.max(-1)
            actions[range(actions.shape[0]), argmax] = val - val * 0.1
            actions = actions.T / actions.sum(1)
            actions = actions.T
            # print(actions.shape, actions[0].sum())

        loss = self.error(torch.log(action), torch.log(actions))
        # print (loss)
        return loss


class TraceEmbDS(Dataset):
    def __init__(self, valid, invalid):
        # data = pickle.load(open(fname,'rb'))
        # valid, invalid = data
        invalid = [v['I'] for v in invalid]
        valid = [v['I'] for v in valid]

        for i in range(len(invalid)):
            # print(i, invalid[i])
            invalid[i] = torch.stack(invalid[i])

        for v in range(len(valid)):
            val = [valid[v][0]]
            if len(valid[v]) != 17:
                diff = 17-len(valid[v])
                val = val * diff
                valid[v] = val + valid[v]
            valid[v] = torch.stack(valid[v])

        if len(valid) >= 1:
            self.vdata = torch.stack(valid)
            self.vdata = self.vdata.float()
            shape = self.vdata.shape
            self.vdata = self.vdata.view(
                shape[0], shape[1], shape[4], shape[5]
                )
        else:
            self.vdata = []
        # print(self.vdata.shape)
        self.idata = torch.stack(invalid)
        self.idata = self.idata.float()
        shape = self.idata.shape
        self.idata = self.idata.view(shape[0], shape[1], shape[4], shape[5])
        # print(self.idata.shape)
        # print(self.vdata.shape, self.idata.shape)

    def __getitem__(self, index):
        # print(index)
        if index < len(self.vdata):
            try:
                d = self.vdata[index]
                l = 1.0     # noqa: E741
            except KeyError:
                d = self.vdata[0]
                l = 1.0     # noqa: E741
        else:
            index -= len(self.vdata)
            try:
                d = self.idata[index]
                l = 0.0     # noqa: E741
            except KeyError:
                d = self.idata[index]
                l = 0.0     # noqa: E741
        # return d, l
        # d = self.vdata[index]
        # l = 1.0
        return d, l

    def __len__(self):
        return len(self.idata) // 2  # * 2 # + len(self.idata)


class TraceEmbDSOne(Dataset):
    def __init__(self, trace, label=1.0):
        # data = pickle.load(open(fname,'rb'))
        # valid, invalid = data
        self.label = label
        trace = [v['I'] for v in trace]
        valid = trace
        for v in range(len(valid)):
            val = [valid[v][0]]
            if len(valid[v]) != 17:
                diff = 17-len(valid[v])
                val = val * diff
                valid[v] = val + valid[v]
            valid[v] = torch.stack(valid[v])

        self.vdata = torch.stack(valid)
        self.vdata = self.vdata.float()
        shape = self.vdata.shape
        self.vdata = self.vdata.view(shape[0], shape[1], shape[4], shape[5])

    def __getitem__(self, index):
        d = self.vdata[index]
        return d, self.label

    def __len__(self):
        return len(self.vdata)


class EnvMNIST:
    def __init__(self, seed=2, render=False):
        self.nprandom = np.random.RandomState(None)
        use_cuda = True
        kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([  # noqa: E128
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
            idx = torch.where(labels == i)[0].data.numpy()
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
        # curr_state_image = self.get_images(self.state)
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
        if self.render is True:
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
        # return image


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
        x1 = torch.tanh(x1)
        x = self.fc2(x1)
        output = F.softmax(x, dim=1)
        return output, x1


# Recurrent neural network (many-to-one)
class Recognizer(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers, num_classes, generator):
        super(Recognizer, self).__init__()
        self.generator = generator
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass the image through the generator
        # Set initial hidden and cell states
        # torch.Size([10, 132, 17])
        # torch.Size([10, 132, 17])
        # torch.Size([10, 132, 17])
        # torch.Size([10, 132, 17])
        # torch.Size([10, 132, 17])
        # torch.Size([8, 132, 17])
        # print(x.shape)
        print('image', x.shape)
        _, x = self.generator(x)
        # print(actions.shape, fc.shape)
        # x = torch.cat([fc, actions], dim=1)
        x = x.reshape(1, x.shape[0], x.shape[1])
        print('image reshape size', x.shape)
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        print('LSTM layer', out.shape)
        exit()
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out, actions


def modify_mnistnet():
    model = Generator()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    for param in model.parameters():
        param.requires_grad = False
    # model.fc2 = nn.Linear(128, 4)
    # model.fc2.requires_grad = True
    # model.fc2.requires_grad = True
    return model


def greedy_action(prob, nprandom):
    # prob = F.softmax(prob.data)
    # print('prob', prob)
    # return nprandom.choice([0, 1, 2, 3], p=prob)
    # return np.argmax(prob)
    return prob


def get_current_state(env, generator):
    image = env.get_images(env.state)
    # actions, fc = generator(image)
    # return env.state, torch.cat([fc, actions], dim=1), actions
    return env.state, image


def generation(generator, env):
    # print(generator)
    actions = [0, 1, 2, 3]
    action = env.nprandom.choice(actions)
    j = 0
    curr_state = get_current_state(env, generator)
    # print(curr_state)
    trace = create_trace_skeleton(curr_state)
    while True:
        # print(j, env.state, action)
        s, _, _, done = env.step(action)
        j += 1
        image = env.get_images(s)
        # print(image.shape)
        actions, fc = generator(image)
        # actions, _ = generator(image)
        state = get_current_state(env, generator)
        trace = trace_accumulator(trace, state)
        action = greedy_action(actions.data.item(), env.nprandom)
        if done:
            break
        if j > 15:
            break
    # print(trace['S'])
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
    for i in range(0, len(traceset[akey])+1):
        t = create_trace_flloat(traceset, i)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, create_trace_dict(trace, i)

    return result, create_trace_dict(trace, i)


def propogation(train_loader, recoginzer, optim, optimgen, error, gerror):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # images = images.reshape(
            # -1, sequence_length, input_size).to(device)
            # print(images.shape, labels.shape)
            shape = images.shape
            images = images.reshape(shape[1], shape[0], shape[2], shape[3])
            images = images.float()
            labels = labels.to(device)
            labels = labels.long()
            # Forward pass
            # print(images.grad_fn, images.grad_fn.next_functions)
            outputs, actions = recoginzer(images)
            # print('recognizer', outputs.shape, actions.shape, labels.shape)
            # gloss = gerror(actions, labels)

            # Recognizer loss
            loss = error(outputs, labels)   # * sum(labels==0.0)

            # Backward and optimize
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()

            # optimgen.zero_grad()
            # gloss.backward()
            # optimgen.step()

            if (i+1) % 10 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch+1, num_epochs, i+1, total_step, loss.item()))
        # print(epoch, generator.fc2.weight, generator.fc2.bias)
        # print(epoch, generator.fc2.bias)


def create_trace_skeleton(state):
    # Create a skeleton for trace
    trace = dict(zip(keys, [list() for i in range(len(keys))]))
    j = 0
    for k in keys:
        trace[k].append(state[j])
        j += 1
    return trace


def trace_accumulator(trace, state):
    for j in range(len(keys)):
        # Add the state variables to the trace
        # temp = trace[keys[j]][-1].copy()
        # temp.append(state[j])
        trace[keys[j]].append(state[j])
    return trace


def create_trace_flloat(traceset, i):
    setslist = [create_sets(traceset[k][:i]) for k in keys]
    dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
    keydictlist = dict()
    j = 0
    for k in keys:
        keydictlist[k] = dictlist[j]
        j += 1
    t = FiniteTraceDict.fromDictSets(keydictlist)
    return t


def create_sets(trace):
    return [set([l]) for l in trace]


def create_trace_dict(trace, i):
    tracedict = dict()
    for k in keys:
        tracedict[k] = trace[k][:i+1]
    return tracedict


def create_traces(env, generator, n=500):
    valid_trace = []
    invalid_trace = []
    for epoch in range(n):
        env.reset()
        trace = generation(generator, env)
        result, trace = recognition(trace)
        if result:
            valid_trace.append(trace)
        else:
            invalid_trace.append(trace)

    return valid_trace, invalid_trace


def test_hardcoded(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # images = images.reshape(
            # -1, sequence_length, input_size).to(device)
            shape = images.shape
            images = images.reshape(shape[1], shape[0], shape[2], shape[3])
            images = images.float()
            labels = labels.to(device)
            labels = labels.long()
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted, labels)
            correct += (predicted == labels).sum().item()

        print(
            'Test Accuracy of the model on the {} test images: {} %'.format(
                total, 100 * correct / total))


def main():
    # Define neural nets
    generator = modify_mnistnet()
    recognizer = Recognizer(
        input_size, hidden_size, num_layers, num_classes, generator).to(device)

    gerror = GeneratorLoss()
    # Loss and optimizer
    error = nn.CrossEntropyLoss()

    optimgen = torch.optim.Adam(
        generator.parameters(), lr=learning_rate)

    optim = torch.optim.Adam(
        list(recognizer.parameters()) +
        list(generator.parameters()), lr=learning_rate
        )

    env = EnvMNIST(render=False)
    # vloss = []
    # iloss = []
    for epoch in range(10):
        fname = 'data/'+str(epoch)+'t.pt'
        if os.path.isfile(fname):
            # valid_traces, invalid_traces = pickle.load(open(fname, 'rb'))
            valid_traces, invalid_traces = torch.load(fname)
        else:
            # valid_traces, invalid_traces = create_traces(env, generator, 100)
            valid_traces, invalid_traces = create_traces(env, recognizer, 100)
            # pickle.dump((valid_traces, invalid_traces), open(fname, "wb"))
            torch.save((valid_traces, invalid_traces), fname)

        print(epoch, len(valid_traces), len(invalid_traces))
        # print(valid_traces)
        # print(invalid_traces)
        # name = 'data/'+str(epoch)+'.pk'
        # pickle.dump([valid_traces, invalid_traces], open(name, "wb" ))
        # exit()
        train_dataset = TraceEmbDS(valid_traces, invalid_traces)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True)
        propogation(
            train_loader, recognizer,
            optim, optimgen, error, gerror)

        # valid_traces, invalid_traces = create_traces(env, generator, 50)
        # name = 'data/'+str(epoch)+'t.pk'
        # pickle.dump((valid_traces,invalid_traces), open(name, "wb" ))
        print(epoch, 'test', len(valid_traces), len(invalid_traces))
        test_dataset = TraceEmbDS(valid_traces, invalid_traces)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=True)
        test_hardcoded(recognizer, test_loader)

    # Save both the recognizer and generator
    torch.save(generator.state_dict(), "generatormain.pt")
    torch.save(recognizer.state_dict(), "recognizermain.pt")

    # # Plot the loss curves
    # plt.plot(vloss, 'g')
    # plt.plot(iloss, 'r')
    # plt.show()


def test():
    # train_dataset = TraceEmbDS(name)
    pass


if __name__ == "__main__":
    main()
