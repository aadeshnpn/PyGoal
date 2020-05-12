import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F

from generatorloss import (
    EnvMNIST
    )

device = 'cpu'

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 9000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = 10  # int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def shuffel_labels(images, labels, i, model):
    if torch.rand(1)[0] <= 0.5:
        indx = labels == i
        images = images[indx][:5]
        labels = labels[indx][:5]
        result = torch.tensor([1])
        actions = model(images)
        # return images[indx][:5], labels[indx][:5], torch.tensor([1])
    else:
        images = images[:5]
        labels = labels[:5]
        result = torch.tensor([0])
        actions = model(images)
        # return images[:5], labels[:5], torch.tensor([0])

    return images, labels, result, actions


def sequence_labels(images, labels, i, model):
    if torch.rand(1)[0] <= 0.5:
        image = []
        label = []
        for j in range(5):
            indx = labels == j
            image.append(images[indx][:1])
            label.append(labels[indx][:1])
        images = torch.cat(image)
        labels = torch.cat(label)
        result = torch.tensor([1])
        # print('sequence', labels, images.shape)
        actions = model(images)
        # return images[indx][:5], labels[indx][:5], torch.tensor([1])
    else:
        images = images[:5]
        labels = labels[:5]
        result = torch.tensor([0])
        actions = model(images)
        # return images[:5], labels[:5], torch.tensor([0])
    return images, labels, result, actions


class CrossEntropyLoss1(nn.Module):
    def __init__(self):
        #  # self.init_params = locals()
        super(CrossEntropyLoss1, self).__init__()
        self.__dict__.update(locals())
        # self.error = nn.CrossEntropyLoss()

    def forward(self, loss, result):
        # loss = self.error(temp, result)
        # print(loss)
        # loss = loss * torch.exp(-1.0 * result)
        # print(loss)
        return loss
'''
STEP 3: CREATE MODEL CLASS
'''

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


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
class RNNModelGen(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, generator):
        super(RNNModelGen, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.fc1 = nn.Linear(hidden_size*2, 2)
        self.generator = generator
        # self.attention_layer = Attention(128,  5)
        self.al = Attention(128,  1)

    def forward(self, x):
        _, x = self.generator(x)
        x = x.view(1, x.shape[0], x.shape[1])

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = F.relu(self.al(out))
        out = self.fc(out)
        # out = F.relu(self.attention_layer(out))
        # out = self.fc1(out)
        return out


# Recurrent neural network (many-to-one)
class RNNModelRec(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, generator):
        super(RNNModelRec, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm_label = nn.LSTM(10, 50, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # self.fc1 = nn.Linear(hidden_size*2, 2)
        self.fc1 = nn.Linear(100, 2)
        self.generator = generator
        self.attention_layer = Attention(100,  5)
        # self.al = Attention(128,  1)

    def forward(self, x, actions):
        _, x = self.generator(x)
        # x = torch.cat((actions, x), dim=1)
        x = x.view(1, x.shape[0], x.shape[1])
        # print(x.shape)
        # print(x.shape, actions.shape, torch.cat((actions, x), dim=1).shape)
        # exit()
        # print(x.shape, actions.shape, .shape)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out1 = F.relu(self.al(out))
        out = self.fc(out)
        # print(out.shape)
        out, _ = self.lstm_label(out)
        out = F.relu(self.attention_layer(out))
        # print('after lstm label',out.shape)
        # exit()
        out = self.fc1(out)
        # print(out.shape)
        # exit()
        # return out1, out
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 128
hidden_dim = 64
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

generator = modify_mnistnet()
model = RNNModelRec(128, hidden_dim, layer_dim, output_dim, generator)
modelgen = RNNModelGen(input_dim, hidden_dim, layer_dim, output_dim, generator)
model = model.to(device)
modelgen = modelgen.to(device)
# JUST PRINTING MODEL & PARAMETERS
print(model)
# print(len(list(model.parameters())))
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()
criterion1 = CrossEntropyLoss1()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001

optimizergen = torch.optim.SGD(modelgen.parameters(), lr=learning_rate)
#  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(modelgen.parameters()),
    lr=learning_rate
    )

'''
STEP 7: TRAIN THE MODEL
'''

# Number of steps to unroll
seq_dim = 1

iter = 0
print(num_epochs)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # print(images.shape)
        # Load images as tensors with gradient accumulation abilities
        # images = images.view(-1, seq_dim, input_dim).requires_grad_()
        # print(images.shape)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        optimizergen.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        # print(images.shape)
        # images, labels =  100, 10
        # print(images.shape)
        # images, labels, result, actions = shuffel_labels(images, labels, i % 9, modelgen)
        images, labels, result, actions = sequence_labels(images, labels, i % 9, modelgen)
        if images.shape[0] != 5:
            # print(images.shape)
            break
        # pyt   pprint(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        result = result.to(device)
        actions = actions.to(device)
        # print(i, images.shape)
        # print(epoch, i, labels.shape, labels)
        temp = model(images.requires_grad_(), actions)
        # print(temp.shape, temp)
        # Calculate Loss: softmax --> cross entropy loss
        # print('temp, result', outputs.shape, labels.shape)
        # loss1 = criterion(outputs, labels)
        loss = criterion(temp, result)
        # loss = criterion1(loss1, result)
        # print('temp, result', i, labels, result)

        # loss = loss1 + loss2
        # loss = loss2
        # Getting gradients w.r.t. parameters
        # loss.backward(retain_graph=True)
        loss.backward()

        # Updating parameters
        optimizer.step()
        # optimizergen.step()
        iter += 1

        if iter % 500 == 0:
            model.eval()
            # Calculate Accuracy
            correct = 0
            total = 0
            real = []
            pred = []
            # Iterate through test dataset
            for images, labels in test_loader:
                # Resize images
                # images = images.view(-1, seq_dim, input_dim)
                images, labels, result, actions = sequence_labels(images, labels, i % 9, modelgen)
                if images.shape[0] != 5:
                    # print(images.shape)
                    continue
                images = images.to(device)
                labels = labels.to(device)
                result = result.to(device)
                actions = actions.to(device)
                # print(labels, result)
                real.append(result.data.detach().item())

                # Forward pass only to get logits/output
                temp = model(images, actions)
                outputs = modelgen(images)
                # print(outputs.shape)
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # _, predicted1 = torch.max(temp.data, 1)
                #if torch.rand(1)[0] > 0.8:
                # print(labels, predicted)
                pred.append(torch.argmax(temp.data).detach().item())
                # print(result.detach().item(), torch.argmax(temp.data).detach().item())
                # Total number of labels
                total += labels.size(0)
                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            # print(real, pred)
            corr = torch.tensor(real) == torch.tensor(pred)
            # print(corr)
            acc = torch.sum(corr)*100 / corr.shape[0]
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}, Acc: {}'.format(iter, loss.item(), accuracy, acc))
    # print(i)

model.eval()
# Calculate Accuracy
correct = 0
total = 0
real = []
pred = []
# Iterate through test dataset
for images, labels in test_loader:
    # Resize images
    # images = images.view(-1, seq_dim, input_dim)
    images, labels, result, actions = sequence_labels(images, labels, i % 9, modelgen)
    if images.shape[0] != 5:
        # print(images.shape)
        continue
    images = images.to(device)
    labels = labels.to(device)
    result = result.to(device)
    actions = actions.to(device)
    real.append(result.data.detach().item())

    # Forward pass only to get logits/output
    temp = model(images, actions)
    outputs = modelgen(images)
    # print(outputs)
    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)
    # _, predicted1 = torch.max(temp.data, 1)
    #if torch.rand(1)[0] > 0.8:
    # print(labels, predicted)
    pred.append(torch.argmax(temp.data).detach().item())
    # print(result.detach().item(), torch.argmax(temp.data).detach().item())
    # Total number of labels
    total += labels.size(0)
    # Total correct predictions
    print(predicted, labels, torch.argmax(temp.detach()).item(), result.item())
    correct += (predicted == labels).sum()
accuracy = 100 * correct / total
corr = torch.tensor(real) == torch.tensor(pred)
# print(corr)
acc = torch.sum(corr)*100 / corr.shape[0]
# Print Loss
print('Iteration: {}. Loss: {}. Accuracy: {}, Acc: {}'.format(iter, loss.item(), accuracy, acc))
