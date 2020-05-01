import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F


device = 'cuda:0'

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
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def shuffel_labels(images, labels, i):
    if torch.rand(1)[0] <= 0.6:
        indx = labels == i
        return images[indx], labels[indx], torch.tensor([1])
    else:
        return images, labels, torch.tensor([0])
    
'''
STEP 3: CREATE MODEL CLASS
'''
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


# class RNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, generator):
#         super(RNNModel, self).__init__()
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim

#         # Number of hidden layers
#         self.layer_dim = layer_dim

#         self.generator = generator
#         # Building your RNN
#         # batch_first=True causes input/output tensors to be of shape
#         # (batch_dim, seq_dim, feature_dim)
#         # self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
#         # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

#         # Readout layer
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):

#         _, x = self.generator(x)
#         # print(x.shape)
#         x = x.view(x.shape[0], 1, x.shape[1])
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

#         # We need to detach the hidden state to prevent exploding/vanishing gradients
#         # This is part of truncated backpropagation through time (BPTT)
#         out, hn = self.rnn(x, h0.detach())

#         # Index hidden state of last time step
#         # out.size() --> 100, 28, 100
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
#         out = self.fc(out[:, -1, :])
#         # out.size() --> 100, 10
#         return out


# Recurrent neural network (many-to-one)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, generator):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc1 = nn.Linear(hidden_size, 2)
        self.generator = generator

    def forward(self, x):
        # print(x.shape)
        # print(x.shape)
        _, x = self.generator(x)
        x = x.view(1 , x.shape[0], x.shape[1])
        # print(x.shape)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        # print(out.shape)
        out1 = self.fc(out)
        # print(out1.shape)
        out = self.fc1(out[:, -1, :])
        # print(out.shape)
        # exit()
        return out1.squeeze(), F.softmax(out, dim=1)

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
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''

# Number of steps to unroll
seq_dim = 1

iter = 0
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
        images = images.to(device)
        labels = labels.to(device)
        result = result.to(device)
        # print(epoch, i, labels.shape, labels)
        outputs, temp = model(images.requires_grad_())
        # print(temp.shape, temp)
        # Calculate Loss: softmax --> cross entropy loss
        # print('temp, result', outputs.shape, labels.shape)
        loss1 = criterion(outputs, labels)
        # print('temp, result',temp.shape, result.shape, temp, result)
        loss2 = criterion1(temp, result)
        loss = loss1 + loss2
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
                images = images.to(device)
                labels = labels.to(device)
                result = result.to(device)
                real.append(result.data.detach().item())
                # Forward pass only to get logits/output
                outputs,temp = model(images)

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
