import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

device = 'cuda:0'


class TransFeedForwd(nn.Module):
    def __init__(self):
        super(TransFeedForwd, self).__init__()
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
        return x1


def modify_mnistnet():
    model = TransFeedForwd()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    return model


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        # print(src.shape)
        src = src.view((1, *src.shape))
        output = self.transformer_encoder(src)
        # output = self.decoder(output)
        return output


def load_dataset():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

    batch_size = 100
    # n_iters = 9000
    # num_epochs = n_iters / (len(train_dataset) / batch_size)
    # num_epochs = 20  # int(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    return train_loader, test_loader


class PolicyNetwork(nn.Module):
    """Policy Network."""

    def __init__(self, state_dim=4, action_dim=2):
        super(PolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        """Receives input x of shape [batch, state_dim].
        Outputs action distribution (categorical distribution) of shape [batch, action_dim],
        as well as a sampled action (optional).
        """
        scores = self._net(x)
        probs = self._softmax(scores)

        if not get_action:
            return probs

        batch_size = x.shape[0]
        actions = np.empty((batch_size, 1), dtype=np.uint8)
        probs_np = probs.cpu().detach().numpy()
        for i in range(batch_size):
            action_one_hot = np.random.multinomial(1, probs_np[i])
            action_idx = np.argmax(action_one_hot)
            actions[i, 0] = action_idx
        return probs, actions


class ValueNetwork(nn.Module):
    """Approximates the value of a particular state."""

    def __init__(self, state_dim=4):
        super(ValueNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        return self._net(x)


def env(images, labels, i, embedder, policy):
    image = []
    label = []
    action = []
    state = 0
    indx = labels == state
    image.append(images[indx][:1])
    label.append(labels[indx][:1])
    done = False
    j = 0
    states = []
    # 0 - left, 1 - right, 2 - down and 3 - up
    while True:
        # state =
        embedding = embedder(image[-1])
        probs = policy(embedding, False)
        # actions, _ = model(image[-1])
        act = torch.argmax(probs).item()
        # print(probs, act)
        # print(j, act)
        if act == 1:
            state = state + 1
        elif act == 0:
            state = state - 1

        state = np.clip(state, 0, 5)
        indx = labels == state
        image.append(images[indx][:1])
        label.append(labels[indx][:1])
        states.append(torch.cat((embedding, probs), dim=1))
        action.append(act)
        if state == 5:
            done = True
        if done:
            break
        if j >= 5:
            break
        j += 1

    # if done:
    #     return torch.cat(image), torch.cat(label), torch.tensor([1]), torch.cat(action)
    # else:
    #     return torch.cat(image), torch.cat(label), torch.tensor([0]), torch.cat(action)

    if done:
        return torch.cat(states).to(device), torch.tensor([1]).to(device)
    else:
        return torch.cat(states).to(device), torch.tensor([0]).to(device)


def embeddings():
    model = modify_mnistnet()
    return model


def main():
    embedder = embeddings()
    train_loader, test_loader = load_dataset()
    policy = PolicyNetwork(state_dim=128, action_dim=4)
    policy = policy.to(device)
    transformer = TransformerModel(500, 132, 2, 200, 2)
    transformer = transformer.to(device)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        states, labels = env(images, labels, i, embedder, policy)
        print(states.shape)
        output = transformer(states)
        print(output.shape)
        if True:
            break



if __name__ == '__main__':
    main()