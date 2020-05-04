"""Script to test new Generator-Recognizer Architecture."""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
import torch.nn.functional as F
from generatorloss import (
    EnvMNIST, create_trace_skeleton, create_trace_flloat,
    create_trace_dict, trace_accumulator, greedy_action
    )

from flloat.parser.ltlfg import LTLfGParser
# from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


device = 'cpu'


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
        # print('shape',shape)
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


def modify_mnistnet():
    model = Generator()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    for param in model.parameters():
        param.requires_grad = False
    return model


# Recurrent neural network (many-to-one)
class RNNModel(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            num_classes, generator):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.fc1 = nn.Linear(hidden_size, 2)
        self.fc1 = nn.Linear(num_classes, 2)
        self.generator = generator

    def forward(self, x):
        # print(x.shape)
        # print(x.shape)
        _, x = self.generator(x)
        x = x.view(1, x.shape[0], x.shape[1])
        # print(x.shape)
        # Set initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        # print(out.shape)
        out1 = self.fc(out)
        # print(out1.shape)
        # out = self.fc1(out[:, -1, :])
        out = self.fc1(out1)
        # print(out.shape)
        # exit()
        x = F.adaptive_avg_pool2d(out, (1, 2)).view((1, 2))
        return out1.squeeze(), x
        # return out1.squeeze(), F.softmax(out, dim=1)


def init_model():
    input_dim = 128
    hidden_dim = 64
    layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    output_dim = 4

    generator = modify_mnistnet()
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, generator)
    model = model.to(device)
    # JUST PRINTING MODEL & PARAMETERS
    criterion = nn.CrossEntropyLoss()
    # criterion1 = nn.CrossEntropyLoss()

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


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
    generator.eval()
    while True:
        # print(j, env.state, action)
        s, _, _, done = env.step(action)
        j += 1
        image = env.get_images(s)
        # print(j, image.shape)
        actions, _ = generator(image)
        # print(j, actions)
        # actions, _ = generator(image)
        state = get_current_state(env, generator)
        # print(state)
        trace = trace_accumulator(trace, state)
        action = greedy_action(actions.data.numpy(), env.nprandom)
        if done:
            break
        if j > 15:
            break
    # print(trace['S'])
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
    for i in range(0, len(traceset[akey])+1):
        t = create_trace_flloat(traceset, i)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, create_trace_dict(trace, i)

    return result, create_trace_dict(trace, i)


def propogation(train_loader, recoginzer, optim, error, num_epochs=1):
    # Train the model
    total_step = len(train_loader)
    recoginzer.train()
    losses = []
    for epoch in range(num_epochs):
        losses = []
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
            actions, outputs = recoginzer(images)
            # print('recognizer', outputs.shape, actions.shape, labels.shape)
            # gloss = gerror(actions, labels)
            # print(i, images.shape, outputs.shape, labels.shape, outputs, labels)
            # Recognizer loss
            loss = error(outputs, labels)   # * sum(labels==0.0)

            # Backward and optimize
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()

            # optimgen.zero_grad()
            # gloss.backward()
            # optimgen.step()
            losses.append(loss.item())
            # if (i+1) % 10 == 0:
        print(
            'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, total_step,
                sum(losses)*1.0/len(losses)))
        # print(epoch, generator.fc2.weight, generator.fc2.bias)
        # print(epoch, generator.fc2.bias)


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


def train():
    env = EnvMNIST()
    model, criterion, optimizer = init_model()
    # trace = generation(model, env)
    # valid_trace, invalid_trace = create_traces(env, model, 1)
    # print(invalid_trace[0]['S'], invalid_trace[0]['I'][0].shape)
    # print(trace)

    for epoch in range(10):
        fname = 'data/'+str(epoch)+'t.pt'
        if os.path.isfile(fname):
            # valid_traces, invalid_traces = pickle.load(open(fname, 'rb'))
            valid_traces, invalid_traces = torch.load(fname)
        else:
            # valid_traces, invalid_traces = create_traces(env, generator, 100)
            valid_traces, invalid_traces = create_traces(env, model, 100)
            # pickle.dump((valid_traces, invalid_traces), open(fname, "wb"))
            torch.save((valid_traces, invalid_traces), fname)

        print(epoch, len(valid_traces), len(invalid_traces))
        if len(invalid_traces) <= 1:
            pass
        else:
            train_dataset = TraceEmbDS(valid_traces, invalid_traces)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=1,
                shuffle=True)
            propogation(
                train_loader, model,
                optimizer, criterion, num_epochs=1)
        # Save both the recognizer and generator
        torch.save(model.state_dict(), "rnnmodel.pt")


def inference():
    pass


def main():
    train()


if __name__ == "__main__":
    main()
