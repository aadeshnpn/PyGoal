import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from itertools import chain
device = 'cuda:0'


class RegressionLoss(nn.Module):
    def __init__(self):
        #  # self.init_params = locals()
        super(RegressionLoss, self).__init__()
        # self.__dict__.update(locals())
        self.error = torch.nn.MSELoss()

    def forward(self, out, reward):
        # out = out.squeeze()
        out = torch.sum(out)
        out = out.view(1)
        # print('loss',out, reward)
        # print('loss',out, torch.sum(out))
        # print('loss',out, out.shape, reward.shape)
        return self.error(out, reward)


class Regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Regression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


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
        # src = src.view((1, src.shape[1], src.shape[0]))
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

    def __init__(self, state_dim=128, action_dim=2):
        super(PolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            # nn.Linear(10, 10),
            # nn.ReLU(),
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

    def __init__(
            self, transformer, selfatten, regression):
        super(ValueNetwork, self).__init__()
        self.transformer = transformer
        self.regression = regression
        self.selfatten = selfatten

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        # return self._net(x)
        hidden = self.transformer(x).squeeze()
        hidden = hidden.transpose(1, 0)
        z = torch.sigmoid(self.selfatten(hidden))
        # hidden = self.selfatten(hidden)
        # print(hidden.shape, z.shape)
        hidden_sum = hidden * z
        hidden_sum = torch.reshape(
            hidden_sum, (hidden_sum.shape[1], hidden_sum.shape[0]))
        # print(hidden_sum.shape)
        out = self.regression(hidden_sum)
        return out


def calculate_returns(tracelen, gamma, finalrew):
    # current_return = 0
    ret = finalrew
    returns = []
    returns.append(ret)
    for i in reversed(range(tracelen)):
        ret = ret * gamma
        returns.append(ret)
    return torch.tensor(returns)


def env(images, labels, i, embedder, policy):
    image = []
    label = []
    action = []
    action_dist = []
    state = 0
    indx = labels == state
    image.append(images[indx][:1])
    label.append(labels[indx][:1])
    done = False
    j = 0
    states = []
    # embedding = embedder(image[-1])
    # probs = policy(embedding, False)
    # states.append(torch.cat((embedding, probs), dim=1))
    # 0 - left, 1 - right, 2 - down and 3 - up

    while True:
        # state =
        embedding = embedder(image[-1])
        actions_dist, act = policy(embedding)
        actions_dist, act = actions_dist[0], act[0]
        # act = torch.argmax(probs).item()
        # probs = [0.10, 0.70, 0.10, 0.10]
        # act = np.random.choice(list(range(4)), p=probs)
        # probs = torch.tensor([probs]).to(device)
        # print(probs, act)
        # print(j, act)
        if act == 1:
            state = state + 1
        elif act == 0:
            state = state - 1

        state = np.clip(state, 0, 5)
        indx = labels == state
        # image.append(images[indx][:1])
        label.append(labels[indx][:1])

        actions = torch.tensor([act * 1.0]).to(device).to(torch.float32)
        actions = actions.view(1, 1)
        # print(embedding, actions)
        states.append(torch.cat((embedding, actions), dim=1))
        action.append(torch.tensor(act))
        action_dist.append(actions_dist)
        if done:
            break
        if state == 5:
            done = True
        if j >= 5:
            break
        j += 1
    # print(len(states), [l.item() for l in label])
    # if done:
    #     return torch.cat(image), torch.cat(label), torch.tensor([1]), torch.cat(action)
    # else:
    #     return torch.cat(image), torch.cat(label), torch.tensor([0]), torch.cat(action)

    if done:
        returns = calculate_returns(5, 0.99, 100.0)
        return (
            torch.cat(states).to(device),
            torch.tensor([100.0]).to(device),
            torch.cat(action).to(device),
            torch.stack(action_dist).to(device),
            returns.to(device)
            )
    else:
        returns = calculate_returns(5, 0.99, 0.0)
        return (
            torch.cat(states).to(device),
            torch.tensor([0.0]).to(device),
            torch.cat(action).to(device),
            torch.stack(action_dist).to(device),
            returns.to(device)
            )


def likelihood_fn(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def embeddings():
    model = modify_mnistnet()
    return model


def main():
    embedder = embeddings()
    train_loader, test_loader = load_dataset()
    policy = PolicyNetwork(state_dim=128, action_dim=2)
    policy = policy.to(device)
    transformer = TransformerModel(200, 129, 3, 200, 2)
    transformer = transformer.to(device)
    selfatt = Attention(6,  129)
    selfatt = selfatt.to(device)
    lregression = Regression(129, 1)
    lregression = lregression.to(device)
    crieteria = RegressionLoss()
    valuenet = ValueNetwork(transformer, selfatt, lregression)
    valuenet = valuenet.to(device)
    # modelpara = (
    #     list(lregression.parameters()) +
    #     list(transformer.parameters()) + list(selfatt.parameters())
    #     + list(valuenet.parameters()) + list(policy.parameters())
    #     )
    modelpara = chain(valuenet.parameters(), policy.parameters())
    # Parameters
    epochs = 30
    epsilon = 0.2
    gamma = 0.99
    lr = 1e-3
    betas = (0.9, 0.999)
    weight_decay = 0.01
    optimizer = torch.optim.Adam(modelpara, lr=lr, betas=betas, weight_decay=weight_decay)

    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    for epoch in range(epochs):
        losses = []
        la = []
        rewards = []
        ploss = []
        vloss = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            states, labels, action, action_dist, returns = env(images, labels, i, embedder, policy)
            # print(states.shape, labels.shape, action.shape, action_dist.shape, returns.shape)
            # exit()
            # print(len(states))
            # hidden = []
            # for i in range(1, len(states)+1):
            #     state = states[:i]
            #     print(state.shape, end=' ')
            #     h = transformer(state)
            #     print(h.shape)
            #     hidden.append(h)

            # hidden = transformer(states).squeeze()
            # hidden = hidden.transpose(1, 0)
            # # hidden = torch.cat(hidden, dim=1)
            # # print(hidden.shape)
            # z = torch.sigmoid(selfatt(hidden))
            # # z = z.view(z.shape[1], z.shape[0])
            # # print('z', z.shape, z.data.cpu())
            # print(hidden.shape, z.shape)
            # hidden_sum = hidden * z
            # # print(hidden_sum.shape)
            # hidden_sum = torch.reshape(hidden_sum, (hidden_sum.shape[1], hidden_sum.shape[0]))
            # out = lregression(hidden_sum)
            current_action_dist = policy(states[:,:128], False)
            current_action_dist = current_action_dist # .squeeze()
            action = action.view(action.shape[0], 1)
            # print('curr action dist', current_action_dist.shape, action.shape)
            current_likelihood = likelihood_fn(current_action_dist, action)
            old_likelihood = likelihood_fn(action_dist, action)
            ratio = (current_likelihood / old_likelihood)

            out = valuenet(states)
            optimizer.zero_grad()
            # print(out.shape, out)
            val_loss = crieteria(out, labels)

            advantage = returns.squeeze() - out.detach().squeeze()
            # print(ratio.shape, advantage.shape)
            lhs = ratio * advantage
            rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
            policy_loss = -torch.mean(torch.min(lhs, rhs))
            loss = val_loss + policy_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            la.append(labels.detach().cpu().item())
            vloss.append(val_loss.detach().cpu().item())
            ploss.append(policy_loss.detach().cpu().item())
            # print(loss)
            # print('optimized')
            # if True:
            #    break
        print('epoch, loss, reward, value loss, policy loss', epoch, np.mean(losses), np.mean(la), np.mean(vloss), np.mean(ploss))
    torch.save(valuenet.state_dict(), "valuenet.pt")


def test_dataset(test_loader, embedder):
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        image = []
        label = []
        action = []
        state = []
        for i in range(6):
            i = np.random.randint(0, 6)
            indx = labels == i
            img = images[indx][:1]
            lab = labels[indx][:1]
            embedding = embedder(img)
            # print(i, end=' ')
            for a in range(1):
                a = 1
                actions = torch.tensor([a * 1.0]).to(device)
                actions = actions.view(1, 1)
                states = torch.cat((embedding, actions), dim=1)
                state.append(states)
        return torch.cat(state).to(device), torch.tensor([1.0]).to(device)


def test():
    embedder = embeddings()
    train_loader, test_loader = load_dataset()
    states, _ = test_dataset(test_loader, embedder)
    transformer = TransformerModel(500, 129, 3, 200, 2)
    transformer = transformer.to(device)
    selfatt = Attention(6,  129)
    selfatt = selfatt.to(device)
    lregression = Regression(129, 1)
    lregression = lregression.to(device)
    crieteria = RegressionLoss()
    valuenet = ValueNetwork(transformer, selfatt, lregression)
    valuenet = valuenet.to(device)
    valuenet.load_state_dict(torch.load("valuenet.pt"))
    # torch.load(valuenet, "valuenet.pt")
    print('load success')
    print(valuenet(states))


if __name__ == '__main__':
    main()
    # test()