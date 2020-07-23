"""Definition of PPO algorithm"""
import imageio
from itertools import chain
import math
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from queue import Queue
import gym
import gym_minigrid
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX, Key, Door, Goal
import torchvision.models as models


from utils import (
    run_envs, ExperienceDataset, prepare_tensor_batch,
    multinomial_likelihood, EnvironmentFactory, RLEnvironment,
    DataLoader, LossPlot, prepare_numpy, make_gif
    )
import gym_super_mario_bros
from gym_super_mario_bros import actions

from environment import (
    ResizeFrameEnvWrapper,
    StochasticFrameSkipEnvWrapper,
    BinarySpaceToDiscreteSpaceEnv,

    )


class MarioEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(MarioEnvironmentFactory, self).__init__()

    def new(self):
        return MarioEnvironment()


class MarioEnvironment(RLEnvironment):
    def __init__(self):
        super(MarioEnvironment, self).__init__()
        env_name = 'SuperMarioBros-1-1-v0'
        env = gym_super_mario_bros.make(env_name)
        env = ResizeFrameEnvWrapper(env, width=224, height=224, grayscale=False)
        env = StochasticFrameSkipEnvWrapper(env, n_frames=5)
        self._env = BinarySpaceToDiscreteSpaceEnv(env, actions.SIMPLE_MOVEMENT)
        # self._env = gym.make(env_name)
        # self._env.max_steps = min(self._env.max_steps, 350)
        # self.ereward = 0
        self.coin = 0

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, _, t, info = self._env.step(action.item())
        # print(s.shape, r)
        # if info['coins'] != 0:
        #    print(_, info['coins'])
        # self.ereward += r
        # if t:
        # carry = True if isinstance(self._env.carrying, Key) else False
        temp = info['coins']
        # print(self.coin, temp)
        # if temp > 0:
        #     print(temp)
        c = 0
        if self.coin == temp:
            pass
        else:
            c = temp - self.coin
        # print(temp, self.coin, c)
        self.coin = temp
        return s, 0.0, t, c
        # else:
        #    return s, 0.0, t

    def reset(self):
        """Returns observation (np.ndarray)"""
        # self.ereward = 0
        # return_dict = self._env.reset()
        # return_dict['carry'] = False
        # return return_dict
        self.coin = 0
        return self._env.reset()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(66176, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x1 = torch.tanh(x1)
        # x = self.fc2(x1)
        # output = F.softmax(x, dim=1)
        return x1


class MarioPolicyNetwork(nn.Module):
    """Policy Network for KeyDoor."""

    def __init__(self, state_dim=512, action_dim=7):
        super(MarioPolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
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
        probs_np = probs.cpu().detach().numpy().astype('float64')
        for i in range(batch_size):
            try:
                action_one_hot = np.random.multinomial(1, probs_np[i])
            except ValueError:
                action_one_hot = np.zeros(probs_np[i].shape)
                action_one_hot[np.argmax(probs_np[i])] = 1
            action_idx = np.argmax(action_one_hot)
            actions[i, 0] = action_idx
        return probs, actions


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.error = torch.nn.MSELoss()

    def forward(self, out, reward):
        # out = torch.sum(out)
        # out = out.view(1)
        # reward = reward.view(1)
        # reward = reward.double()    # .to('cuda:0')
        # out = out.double()  # .to('cuda:0')
        # print('loss', out.shape, reward.shape)
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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.view((1, *src.shape))
        output = self.transformer_encoder(src)
        return output


### Temporal reward prediction value network
### Implemented from paper Temporal Credit assignment
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
        x = x.squeeze()
        hidden = self.transformer(x).squeeze()
        # print('hidden shape',hidden.shape)
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


def ppo(env_factory, policy, value, likelihood_fn, embedding_net=None, epochs=100,
        rollouts_per_epoch=100, max_episode_length=20, gamma=0.99, policy_epochs=5,
        batch_size=50, epsilon=0.2, environment_threads=8, data_loader_threads=4,
        device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
        gif_name='', gif_epochs=0, csv_file='latest_run.csv', valueloss= nn.MSELoss()):

    # Clear the csv file
    with open(csv_file, 'w') as f:
        f.write('avg_reward, value_loss, policy_loss\n')

    # Multi-processing
    # mp.set_start_method('spawn', force=True)

    # Move networks to the correct device
    # policy = policy.to(device)
    # policy.share_memory()
    value = value.to(device)
    # value.share_memory()
    # Collect parameters
    params = chain(policy.parameters(), value.parameters())
    if embedding_net:
        embedding_net = embedding_net.to('cpu')
        # embedding_net.share_memory()
        # params = chain(params, embedding_net.parameters())

    # Set up optimization
    # optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer = optim.Adam(params, lr=lr)
    value_criteria = valueloss

    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    loop = tqdm(total=epochs, position=0, leave=False)

    # Prepare the environments
    environments = [env_factory.new() for _ in range(environment_threads)]
    rollouts_per_thread = rollouts_per_epoch // environment_threads
    remainder = rollouts_per_epoch % environment_threads
    rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (environment_threads - remainder))

    for e in range(epochs):
        embedding_net = embedding_net.to('cpu')
        policy = policy.to('cpu')
        # Run the environments
        experience_queue = Queue()
        reward_queue = Queue()
        threads = [Thread(target=run_envs, args=(environments[i],
                                                  embedding_net,
                                                  policy,
                                                  experience_queue,
                                                  reward_queue,
                                                  rollout_nums[i],
                                                  max_episode_length,
                                                  gamma,
                                                  'cpu')) for i in range(environment_threads)]
        for x in threads:
            x.start()
        for x in threads:
            x.join()
        # experience_queue = mp.Queue()
        # reward_queue = mp.Queue()
        # threads = [mp.Process(target=run_envs, args=(environments[i],
        #                                           embedding_net,
        #                                           policy,
        #                                           experience_queue,
        #                                           reward_queue,
        #                                           1, #rollout_nums[i],
        #                                           max_episode_length,
        #                                           gamma,
        #                                           device)) for i in range(environment_threads)]
        # for x in threads:
        #     x.start()
        #     print('threads started')
        # for x in threads:
        #     x.join()
        #     print('treads ended')

        # Collect the experience
        rollouts = list(experience_queue.queue)
        avg_r = sum(reward_queue.queue) / reward_queue.qsize()
        loop.set_description('avg reward: % 6.2f' % (avg_r))

        # Make gifs
        #if gif_epochs and e % gif_epochs == 0:
        # make_gif(rollouts[0], gif_name + '%d.gif' % e)

        # Move the network to GPU
        policy = policy.to(device)
        embedding_net = embedding_net.to(device)
        # Update the policy
        # print(list(rollouts))

        experience_dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(experience_dataset, num_workers=data_loader_threads, batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0

        for _ in range(policy_epochs):
            avg_policy_loss = 0
            avg_val_loss = 0
            for state, old_action_dist, old_action, reward, ret, s1 in data_loader:
                state = prepare_tensor_batch(state, device)
                old_action_dist = prepare_tensor_batch(old_action_dist, device)
                old_action = prepare_tensor_batch(old_action, device)
                ret = prepare_tensor_batch(ret, device).unsqueeze(1)
                s1 = prepare_tensor_batch(s1, device)
                optimizer.zero_grad()
                # print(state.shape)
                if state.shape[0] != 20:
                    continue
                # If there is an embedding net, carry out the embedding
                if embedding_net:
                    # print(state.shape, embedding_net)
                    state = embedding_net(state)
                    # sp = state.shape
                    # state = state.reshape(sp[0], sp[1]*sp[2])
                # Calculate the ratio term
                current_action_dist = policy(state, False)
                # print(current_action_dist.shape)
                current_likelihood = likelihood_fn(current_action_dist, old_action)
                old_likelihood = likelihood_fn(old_action_dist, old_action)
                ratio = (current_likelihood / old_likelihood)

                # Calculate the value loss
                # print(s1.shape)
                expected_returns = value(s1)
                # print(expected_returns.shape, ret.shape)
                # print(expected_returns, ret)
                val_loss = value_criteria(expected_returns, ret)
                # val_loss = value_criteria(expected_returns, reward.sum().detach())

                # Calculate the policy loss
                advantage = ret - expected_returns.detach()
                # print(ratio.shape, advantage.shape)
                lhs = ratio * advantage
                rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))

                # For logging
                avg_val_loss += val_loss.item()
                avg_policy_loss += policy_loss.item()

                # Backpropagate
                loss = policy_loss + val_loss
                loss.backward()
                optimizer.step()

            # Log info
            avg_val_loss /= len(data_loader)
            avg_policy_loss /= len(data_loader)
            torch.save(policy.state_dict(), "policy.pt")
            # torch.save(embedding_net.state_dict(), "embedded.pt")
            # Render the mario game after training one policy
            # Render
            loop.set_description(
                'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' % (avg_r, avg_val_loss, avg_policy_loss))
        with open(csv_file, 'a+') as f:
            f.write('%6.2f, %6.2f, %6.2f\n' % (avg_r, avg_val_loss, avg_policy_loss))
        print()
        loop.update(1)


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        # print(list(original_model.children()))
        paras = list(original_model.children())
        paras[2] = paras[2][:2]
        # print(paras)
        self.features = nn.Sequential(
            *paras
            )
        # print(self.features)
        # self.features = paras
        self.maxpool = nn.MaxPool1d(8, stride=8)

    def forward(self, x):
        # print(x.shape)
        i = 0
        for paras in self.features:
            # print(paras[0], x.shape)
            if i == 2:
                x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
            x = paras(x)
            i += 1
        # print(x.shape)
        x = x.view(x.shape[0], 1 , x.shape[1])
        return self.maxpool(x).squeeze(1)


def main():
    factory = MarioEnvironmentFactory()
    policy = MarioPolicyNetwork()
    transformer = TransformerModel(500, 513, 1, 200, 2)
    selfatt = Attention(20, 513)
    lregression = Regression(513, 1)
    value = ValueNetwork(transformer, selfatt, lregression)
    # embeddnet = models.resnet18(pretrained=True)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])

    # for name, param in embeddnet.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)
    # for l in embeddnet.features.modules():
    #     print(l)

    res18_model = models.vgg11(pretrained=True)
    # res18_model = models.squeezenet1_1(pretrained=True)
    res18_model = ResNet50Bottom(res18_model)
    for para in res18_model.parameters():
        para.requires_grad = False
    # a = torch.rand(1, 3, 224, 224)
    # a = a.to('cuda:0')
    # res18_model = res18_model.to('cuda:0')
    # print(res18_model(a).shape)
    # layer2.0.downsample.0.weight
    # for para in embeddnet.parameters():
    #    print(para)
    # print(embeddnet.parameters())
    ppo(factory, policy, value, multinomial_likelihood, epochs=100,
        rollouts_per_epoch=100, max_episode_length=50,
        gamma=0.9, policy_epochs=5, batch_size=20,
        device='cuda:0', valueloss=RegressionLoss(), embedding_net=res18_model)

    # draw_losses()


def render(policy, embedding_net, device):
    env = MarioEnvironment()
    s = env.reset()
    for _ in range(300):
        env.render()
        input_state = prepare_numpy(s, device)
        input_state = embedding_net(input_state)
        action_dist, action = policy(input_state)
        action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
        s_prime, r, t, coins = env.step(action)
        s = s_prime


def draw_losses():
    fname = 'latest_run.csv'
    import os
    folder = os.getcwd()
    graph = LossPlot(folder, fname)
    graph.gen_plots()


if __name__ == '__main__':
    main()
    # draw_losses()
