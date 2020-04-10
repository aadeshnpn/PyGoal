import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import numpy as np
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(device)
# device = 'cuda:0'

# Hyper-parameters
sequence_length = 132 # 1
input_size = 17
hidden_size = 128
num_layers = 1
num_classes = 2
batch_size = 4
num_epochs = 3
learning_rate = 0.01


class TraceDS(Dataset):
    def __init__(self):
        data = pickle.load(open('traces.pk','rb'))
        valid, invalid = data
        invalid = [v['S'] for v in invalid ]
        for v in valid:
            val = [0]
            if len(v['S']) != 17:
                diff = 17-len(v['S'])
                val = val * diff
                v['S'] = val + v['S']
        valid = [v['S'] for v in valid ]
        self.vdata = torch.tensor(np.array(valid))
        self.vdata = self.vdata.float()
        self.vdata = self.vdata.view(self.vdata.shape[0], 1 , self.vdata.shape[1])
        self.idata = torch.tensor(np.array(invalid))
        self.idata = self.idata.float()
        self.idata = self.idata.view(self.idata.shape[0], 1 , self.idata.shape[1])

    def __getitem__(self, index):
        # print(index)
        if index < len(self.vdata):
            d = self.vdata[index]
            l = 1.0
        else:
            d = self.idata[index]
            l = 0.0
        return d, l

    def __len__(self):
        return  len(self.idata)


class TraceEmbDS(Dataset):
    def __init__(self, fname='traces.pk'):
        data = pickle.load(open(fname,'rb'))
        valid, invalid = data
        invalid = [v['I'] for v in invalid ]
        valid = [v['I'] for v in valid ]

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

        self.vdata = torch.stack(valid)
        self.vdata = self.vdata.float()
        self.vdata = self.vdata.view(self.vdata.shape[0], self.vdata.shape[3] , self.vdata.shape[1])
        self.idata = torch.stack(invalid)
        self.idata = self.idata.float()
        self.idata = self.idata.view(self.idata.shape[0], self.idata.shape[3] , self.idata.shape[1])
        # print(self.vdata.shape, self.idata.shape)

    def __getitem__(self, index):
        # print(index)
        if index < len(self.vdata):
            d = self.vdata[index]
            l = 1.0
        else:
            d = self.idata[index]
            l = 0.0
        return d, l

    def __len__(self):
        return  len(self.idata)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        return out


def main():
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    train_dataset = TraceEmbDS() # TraceDS()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=4,
                                            shuffle=True)

    test_dataset = TraceEmbDS() # TraceDS()

    test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=4,
                                            shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            images = images.float()
            # print('images', images.shape, images)
            # print('images', images.shape, images)
            labels = labels.to(device)
            labels = labels.long()
            # Forward pass
            outputs = model(images)
            # print('output and label',outputs.shape, labels.shape)
            # print(outputs, labels)
            # exit()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            images = images.float()
            labels = labels.to(device)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted, labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'modeltrace.ckpt')


def invalidtest():
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load("model.ckpt"))
    data = [[np.random.randint(0, 4) * 1.0 for a in range(17)] for b in range(4)]
    data = torch.tensor(np.array(data))
    data = data.view(4,1,17)
    # print(data)
    data = data.float()
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)

def validtest():
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load("model.ckpt"))
    # data = [[np.random.randint(0, 4) * 1.0 for a in range(17)] for b in range(4)]
    data = [
        [0,0,0,0,0,0,0,0,0,1,2,3,3,4,4,4,5],
        [0,0,0,0,0,0,0,1,1,1,2,3,3,4,4,4,5],
        [0,0,0,0,0,0,1,1,2,2,2,3,3,3,4,4,5],
        [0,1,1,1,1,1,1,1,1,2,2,3,3,3,3,4,5]
        ]
    # print(len(data))
    data = torch.tensor(np.array(data))
    data = data.view(4,1,17)
    # print(data)
    data = data.float()
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)


def load():
    train_dataset = TraceEmbDS() #TraceDS()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=4,
                                               shuffle=True)
    for images, label in train_loader:
        print(images.shape, label)


def test_real_trace():
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load("modeltrace.ckpt"))
    train_dataset = TraceEmbDS('tracestest.pk') # TraceDS()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=5,
                                            shuffle=False)
    for data, labels in train_loader:

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted, labels, data.shape)


if __name__ == "__main__":
    # main()
    # invalidtest()
    # validtest()
    # load()
    test_real_trace()