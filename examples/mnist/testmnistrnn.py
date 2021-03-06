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
num_epochs = 60  # int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def shuffel_labels(images, labels, i):
    if torch.rand(1)[0] <= 0.5:
        indx = labels == i
        return images[indx][:5], labels[indx][:5], torch.tensor([1])
    else:
        return images[:5], labels[:5], torch.tensor([0])

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
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, generator):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # self.fc1 = nn.Linear(num_classes*5, 2)
        self.fc1 = nn.Linear(hidden_size*2, 2)
        self.generator = generator
        # self.attention_layer = Attention(128,  5)
        self.attention_layer = Attention(128,  5)
        self.al = Attention(128,  1)

    def forward(self, x):
        # print(x.shape)
        # print(x.shape)
        _, x = self.generator(x)
        x = x.view(1, x.shape[0], x.shape[1])
        # print(x.shape)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        # print(out.shape)
        # out = out.view(out.shape[1], out.shape[0], out.shape[2])
        out1 = F.relu(self.al(out))
        # print(out.shape, out1.shape)
        out1 = self.fc(out1)
        # print(out1.shape)
        # print(out.shape)
        # out = self.fc1(out1[:, -1, :])
        # out = F.relu(self.attention_layer(out))
        # print(out.shape)
        out = F.relu(self.attention_layer(out))
        out = self.fc1(out)
        # print(out1.shape, out.shape)
        # print(out.shape)
        # print(out.shape)  # 1, 2
        # exit()
        # out = F.softmax(out, dim=1)
        # print('final layer', out.shape)
        # x = F.adaptive_max_pool2d(out, (1, 2)).view((1, 2))
        # print(x.shape, F.softmax(x, dim=1).shape)
        # print(x.shape)
        return out1, out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 128
hidden_dim = 64
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

generator = modify_mnistnet()
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, generator)
model = model.to(device)
# JUST PRINTING MODEL & PARAMETERS
print(model)
# print(len(list(model.parameters())))
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        # print(images.shape)
        # images, labels =  100, 10
        # print(images.shape)
        images, labels, result = shuffel_labels(images, labels, i % 9)

        if images.shape[0] != 5:
            # print(images.shape)
            break
        # pyt   pprint(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        result = result.to(device)
        # print(i, images.shape)
        # print(epoch, i, labels.shape, labels)
        outputs, temp = model(images.requires_grad_())
        # print(temp.shape, temp)
        # Calculate Loss: softmax --> cross entropy loss
        # print('temp, result', outputs.shape, labels.shape)
        # loss1 = criterion(outputs, labels)
        loss2 = criterion1(temp, result)
        # print('temp, result',temp.shape, result.shape, temp, result)
        # loss = loss1 + loss2
        loss = loss2
        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

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
                images, labels, result = shuffel_labels(images, labels, i % 9)
                if images.shape[0] != 5:
                    # print(images.shape)
                    continue
                images = images.to(device)
                labels = labels.to(device)
                result = result.to(device)
                real.append(result.data.detach().item())

                # Forward pass only to get logits/output
                outputs, temp = model(images)
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
    images, labels, result = shuffel_labels(images, labels, i % 9)
    if images.shape[0] != 5:
        # print(images.shape)
        continue
    images = images.to(device)
    labels = labels.to(device)
    result = result.to(device)
    real.append(result.data.detach().item())

    # Forward pass only to get logits/output
    outputs, temp = model(images)
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
