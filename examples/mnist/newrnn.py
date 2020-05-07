"""Script to test new Generator-Recognizer Architecture."""
import os
import torch
import torch.nn as nn
import numpy as np
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


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.error = nn.MSELoss()
        # self.error = nn.CosineEmbeddingLoss()

    def forward(self, action, predict, label):
        # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True)
        # action = action.clone().detach()

        softmax = torch.softmax(predict, dim=1)
        indx = torch.argmax(softmax, dim=1).item()

        action = F.softmax(action)
        actions = action.clone().detach()
        # print('generator loss', actions.shape, actions[0].sum())
        # print(label)
        if indx == 1 and label.item() == 1:
            # if label.item() == 1:
            val, argmax = actions.max(-1)
            # actions[range(actions.shape[0]), argmax] = val + val * 0.5
            # actions = actions.T / actions.sum(1)
            # actions = actions.T
            # print(action, actions)
            action = action[range(action.shape[0]), argmax]
            actions = actions[range(actions.shape[0]), argmax]
            # print(1, 1, action, actions)
            # loss = 1 + torch.dot(torch.log(action), torch.log(actions))
            # loss = -1.0 * F.l1_log_loss(torch.log(action), torch.log(actions))
            loss = self.error(torch.log(action), torch.log(actions))
            # print(1, 1, action, actions, loss)
        else:
            val, argmax = actions.max(-1)
            actions[range(actions.shape[0]), argmax] = val - val * 0.2
            actions = actions.T / actions.sum(1)
            actions = actions.T
            action = action[range(action.shape[0]), argmax]
            actions = actions[range(actions.shape[0]), argmax]
            loss = self.error(torch.log(action), torch.log(actions))
            # loss = -1.0 * F.l1_log_loss(torch.log(action), torch.log(actions))
            # loss = 1 + torch.dot(torch.log(action), torch.log(actions))
            # print(indx, label.item(), action, actions, loss)
        # print(actions.shape, actions[0].sum())
        # loss = torch.dot(action, actions)   # self.error(torch.exp(action), torch.exp(actions))
        # loss = self.error(action, actions)
        # print(action, actions)
        # if indx == 1 and label.item() == 1:
        #     loss = -1.0 * F.l1_loss(action, actions)
        # # elif indx == 0 and label.item() == 0:
        # #    loss = 0.0 * F.l1_loss(action, actions)
        # else:
        #     loss = F.l1_loss(action, actions)

        return loss
        # else:
        #    return torch.tensor(0.0)


class CrossEntropyLoss1(nn.Module):
    def __init__(self, input=None, target=None):
        #  # self.init_params = locals()
        super(CrossEntropyLoss1, self).__init__()
        self.__dict__.update(locals())
        self.error = nn.CrossEntropyLoss()

    def forward(self, prediction, labels):
        softmax = torch.softmax(prediction, dim=1)
        indx = torch.argmax(softmax, dim=1).item()
        mloss = (1-indx) * torch.log(softmax) + indx * (torch.log(softmax))
        mloss = -1.0 * mloss
        if indx == 1 and labels.item() == 1:
            mloss = -1.0 * mloss[0][labels.item()]
        else:
            mloss = torch.sum(mloss).squeeze() * 2
        return mloss


class MainLoss(nn.Module):
    def __init__(self):
        super(MainLoss, self).__init__()
        self.error = nn.CrossEntropyLoss()

    def forward(self, prediction, labels):
        print(prediction, labels)
        softmax = torch.softmax(prediction, dim=1)
        print('softmax', softmax)
        indx = torch.argmax(softmax, dim=1).item()
        # score = softmax[0][indx]
        # if indx.item() == 0:
        # match = indx != labels
        # match = match * 1.0
        # mloss = -1.0 * (match * torch.log(score))
        print(1-indx, softmax[0][0])
        mloss = (1-indx) * torch.log(softmax) + indx * (1-torch.log(softmax))
        mloss = -1.0 * mloss[0][indx]
        loss = self.error(prediction, labels)
        print('loss', mloss, loss)
        exit()
        predict = torch.argmax(prediction)

        if predict.item() == 1 and labels.item() == 1:
            loss = loss / 2.0
        else:   # predict.item() == 0 and labels.item() == 0:
            loss = ((1 - predict.item()) + (1 - labels.item())) + loss
        # else:
        #    loss = ((1 - predict.item()) + (1 - labels.item())) * loss
        # if labels.item() == 1:
        #     loss = self.error(prediction, labels) * 0.0
        # else:
        #     loss = self.error(prediction, labels)
        print('predict, true, loss:',predict.item(), labels.item(), loss)
        return loss


class TraceEmbDSV(Dataset):
    def __init__(self, valid):
        valid = [v['I'] for v in valid]

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

    def __getitem__(self, index):
        d = self.vdata[index]
        l = 1.0     # noqa: E741
        return d, l

    def __len__(self):
        return len(self.vdata)  # * 2 # + len(self.idata)


class TraceEmbDSI(Dataset):
    def __init__(self, invalid):
        invalid = [v['I'] for v in invalid]

        for i in range(len(invalid)):
            invalid[i] = torch.stack(invalid[i])

        self.idata = torch.stack(invalid)
        self.idata = self.idata.float()
        shape = self.idata.shape
        self.idata = self.idata.view(shape[0], shape[1], shape[4], shape[5])

    def __getitem__(self, index):
        d = self.idata[index]
        l = 0.0     # noqa: E741
        return d, l

    def __len__(self):
        return len(self.idata)


class TraceEmbDS(Dataset):
    def __init__(self, valid, invalid):
        # data = pickle.load(open(fname,'rb'))
        # valid, invalid = data
        invalid = [v['I'] for v in invalid]
        valid = [v['I'] for v in valid]

        for i in range(len(invalid)):
            # print(i, invalid[i])
            invalid[i] = torch.stack(invalid[i])

        if len(invalid) >= 1:
            # print(self.vdata.shape)
            self.idata = torch.stack(invalid)
            self.idata = self.idata.float()
            shape = self.idata.shape
            self.idata = self.idata.view(shape[0], shape[1], shape[4], shape[5])
            ilabel = torch.zeros(self.idata.shape[0])
        else:
            self.idata = None

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
            vlabel = torch.ones(self.vdata.shape[0])
        else:
            self.vdata = None
        if self.idata is not None and self.vdata is not None:
            self.data = torch.cat((self.idata, self.vdata), dim=0)
            self.labels = torch.cat((ilabel, vlabel), dim=0)
        elif self.idata is None:
            self.data = self.vdata
            self.labels = vlabel
        elif self.vdata is None:
            self.data = self.idata
            self.labels = ilabel
        else:
            print('Zero data')
            exit()
        # print(self.idata.shape)
        # print(self.vdata.shape, self.idata.shape)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


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
        self.fc1 = nn.Linear(hidden_size, 2)
        # self.fc1 = nn.Linear(num_classes, 2)
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
        actions = self.fc(out)
        # print(out1.shape)
        # out = self.fc1(out[:, -1, :])
        out = self.fc1(out)
        # print(out.shape)
        # exit()
        x = F.adaptive_avg_pool2d(out, (1, 2)).view((1, 2))
        # print(out.shape)
        # x = F.conv2d()
        return actions.squeeze(), x
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
    # criterion = MainLoss()
    # criterion = CrossEntropyLoss1()
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


def propogation(train_loader, recoginzer, optim, error, errorgen, num_epochs=1):
    # Train the model
    total_step = len(train_loader)
    recoginzer.train()
    losses = []
    for epoch in range(num_epochs):
        acc = []
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
            # print(images.shape)
            actions, outputs = recoginzer(images)
            # print(i, actions)
            # print('recognizer', outputs.shape, actions.shape, labels.shape)
            # gloss = gerror(actions, labels)
            # print(i, images.shape, outputs.shape, labels.shape, outputs, labels)
            # Recognizer loss
            # if genprop:
            #    loss = error(actions, outputs, labels)
            # else:
            loss2 = error(outputs, labels)   # * sum(labels==0.0)
            loss1 = errorgen(actions, outputs, labels)
            # loss2 = error(outputs, labels)
            loss = loss2 + loss1
            # print(loss, loss2, loss1)
            # Next loss GenRecProp
            # if error1 is not None:
            #     loss1 = error1(actions, labels)   # * sum(labels==0.0)
            #     loss = loss + loss1
            # Backward and optimize
            # optim.zero_grad()
            # loss.backward()
            # optim.step()

            losses.append(loss)
            # print(outputs, labels)
            corr = torch.argmax(outputs) == labels
            corr = corr * 1
            acc.append(corr.item())
            # print(corr)
            # acc = torch.sum(corr)*100 / corr.shape[0]
            # if (i+1) % 10 == 0:
        optim.zero_grad()
        losses = torch.stack(losses)
        # print(losses)
        loss = torch.mean(losses)
        # print('loss', loss)
        loss.backward()
        optim.step()
        acc = sum(acc)*100 / len(acc)
        print(
            'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                epoch+1, num_epochs, i+1, total_step,
                loss.item(), acc))
        # print(epoch, generator.fc2.weight, generator.fc2.bias)
        # print(epoch, generator.fc2.bias)
        # return losses


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


def random_sample(lists, n):
    alist = []
    index = np.random.randint(0, len(lists), n)
    for i in index:
        alist.append(lists[i])
    return alist


def grad_option_model(model, default=True):
    if default:
        model.fc.requires_grad = True
        model.lstm.requires_grad = True
        model.fc1.requires_grad = True
    else:
        model.fc.requires_grad = True
        model.lstm.requires_grad = False
        model.fc1.requires_grad = False
    return model


def train():
    env = EnvMNIST()
    model, criterion, optimizer = init_model()
    # trace = generation(model, env)
    # valid_trace, invalid_trace = create_traces(env, model, 1)
    # print(invalid_trace[0]['S'], invalid_trace[0]['I'][0].shape)
    # print(trace)
    # learning_rate = 0.01
    # model = grad_option_model(model, False)
    # optimizergen = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criteriongen = GeneratorLoss()

    allvalid = []
    allinvalid = []
    # valid = 5
    # invalid = 0
    for epoch in range(40):
        # model = grad_option_model(model, True)
        fname = 'data/'+str(epoch)+'t.pt'
        if os.path.isfile(fname):
            # valid_traces, invalid_traces = pickle.load(open(fname, 'rb'))
            valid_traces, invalid_traces = torch.load(fname)
        else:
            # valid_traces, invalid_traces = create_traces(env, generator, 100)
            valid_traces, invalid_traces = create_traces(env, model, 20)
            # pickle.dump((valid_traces, invalid_traces), open(fname, "wb"))
            # torch.save((valid_traces, invalid_traces), fname)

        # if len(valid_traces) >= 1:
        #     allvalid += valid_traces
        #     train_dataset = TraceEmbDSV(allvalid)
        #     train_loader = torch.utils.data.DataLoader(
        #         dataset=train_dataset,
        #         batch_size=1,
        #         shuffle=True)
        #     propogation(
        #         train_loader, model,
        #         optimizer, criterion, None, num_epochs=1)

        # else:   # len(invalid_traces) >= 1:
        #     if invalid <= 5:
        #         train_dataset = TraceEmbDSI(invalid_traces)
        #         train_loader = torch.utils.data.DataLoader(
        #             dataset=train_dataset,
        #             batch_size=1,
        #             shuffle=True)
        #         propogation(
        #             train_loader, model,
        #             optimizer, criterion, criterion1, num_epochs=1)
        #         invalid += 1
        allvalid += valid_traces
        allinvalid += invalid_traces
        print(epoch, len(valid_traces), len(invalid_traces), len(allvalid))
        # n = np.clip(len(allinvalid), 1, 20)
        if len(allvalid) >= 1:
            n = np.clip(len(allvalid), 1, 20)
            randvalid = random_sample(allvalid, n)
        else:
            randvalid = valid_traces

        if len(allinvalid) >= 1:
            n = np.clip(len(allvalid), 1, 20)
            randinvalid = random_sample(allinvalid, n)
        else:
            randinvalid = invalid_traces

        # print(len(randvalid), len(randinvalid))
        train_dataset = TraceEmbDS(randvalid, randinvalid)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True)
        propogation(
            train_loader, model,
            optimizer, criterion, criteriongen, num_epochs=1)

        # model = grad_option_model(model, False)
        # propogation(
        #     train_loader, model,
        #     optimizergen, criteriongen, genprop=True, num_epochs=1)
        # Save both the recognizer and generator
        torch.save(model.state_dict(), "rnnmodel.pt")


def inference():
    pass


def main():
    train()


def test_model():
    # Load model parameter
    input_dim = 128
    hidden_dim = 64
    layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    output_dim = 4
    generator = modify_mnistnet()
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, generator)
    model.load_state_dict(torch.load("rnnmodel.pt"))
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    env = EnvMNIST()
    valid, invalid = create_traces(env, model, n=10)
    print(len(valid), len(invalid))


if __name__ == "__main__":
    main()
    # test_model()
