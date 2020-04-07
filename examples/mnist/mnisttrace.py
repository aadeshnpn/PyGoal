import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt


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
            plt.imshow(curr_state_image.view(28,28))
            plt.show()
        return self.state, 1, None, done

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
        super(RNNModel, self).__init__()
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
        out = self.fc1(out[:, -1, :])
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


def main():
    generator = modify_mnistnet()
    # print(generator)
    env = EnvMNIST(render=True)
    actions = [0, 1, 2, 3]
    action = env.nprandom.choice(actions)
    j = 0
    while True:
        s, _, _, done = env.step(action)
        if done:
            break
        if j > 20:
            break
        j += 1
        actions, fc = generator(env.get_images(s))
        action = greedy_action(actions.data.numpy())
        print(j, action)
    # print(actions, torch.sum(actions))



if __name__ == "__main__":
    main()