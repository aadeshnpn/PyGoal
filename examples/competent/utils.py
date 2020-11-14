"""Useful methods for PPO."""
import os
import time
import math
import numpy as np
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402

# import imageio
from itertools import chain
# import math
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torch.multiprocessing as mp
from tqdm import tqdm
from queue import Queue
import gym

import gym_minigrid     # noqa: F401
from gym_minigrid.minigrid import (     # noqa: F401
    Grid, OBJECT_TO_IDX, Key, Door, Goal)

from torch.utils.data import Dataset
import pandas as pd

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


class RLEnvironment(object):
    """An RL Environment, used for wrapping environments to run PPO on."""

    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        """Takes an action x, which is the same format as
         the output from a policy network.
        Returns observation (np.ndarray), reward (float), terminal (boolean)
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the environment.
        Returns observation (np.ndarray)
        """
        raise NotImplementedError()


class EnvironmentFactory(object):
    """Creates new environment objects"""

    def __init__(self):
        super(EnvironmentFactory, self).__init__()

    def new(self):
        raise NotImplementedError()


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


class RecognizerDataset(Dataset):
    def __init__(self, experience, max_trace_len=30):
        super(RecognizerDataset, self).__init__()
        self._exp = []
        self.max_trace_len = max_trace_len
        for x in experience:
            # print(len(x), end=' ')
            x = [x[i][-1] for i in range(len(x))]
            # print(len(x), end=' ')
            x = self.recursive_fill(x)
            # print(len(x), x[0].shape)
            # x = torch.stack(x)
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length

    def recursive_fill(self, x):
        # while len(x) != self.max_trace_len:
        fix_len = self.max_trace_len - len(x)
        x = [x[0]]*fix_len + x
        return x


def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def get_log_p(data, mu, sigma):
    """get negative log likelihood from normal distribution"""
    return -torch.log(
        torch.sqrt(
            2 * math.pi * sigma ** 2)) - (data - mu) ** 2 / (2 * sigma ** 2)


def calculate_returns(trajectory, gamma, trace, keys, goalspec, r):
    result, j = recognition(trace, keys, goalspec)
    ret = r if result is True else 0
    episode_reward = r if result is True else 0
    for i in reversed(range(len(trajectory[:j]))):
        try:
            state, action_dist, action, rwd, s1 = trajectory[i]
            rets = 0
        except ValueError:
            state, action_dist, action, rwd, rets, s1 = trajectory[i]
        trajectory[i] = (state, action_dist, action, rwd, ret+rets, s1)
        ret = ret * gamma
    for i in range(j, len(trajectory)):
        ret = 0
        try:
            state, action_dist, action, rwd, s1 = trajectory[i]
            rets = 0
        except ValueError:
            state, action_dist, action, rwd, rets, s1 = trajectory[i]
        trajectory[i] = (state, action_dist, action, rwd, ret+rets, s1)
    return episode_reward, trajectory, result


def run_envs(
        env, embedding_net, policy, experience_queue, reward_queue,
        num_rollouts, max_episode_length, gamma, device):
    keys = ['C', 'D', 'DO', 'G']
    for _ in range(num_rollouts):
        current_rollout = []
        statedict = env.reset()
        s, direction, carry = (
            statedict['image'], statedict['direction'], statedict['carry'])
        # print(s.shape)
        door, door_open, goal = (
            statedict['door'], statedict['door_open'], statedict['goal'])
        s = np.reshape(s, (s.shape[0]*s.shape[1]*s.shape[2]))
        direction = np.array([direction])
        s = np.concatenate((s, direction))
        episode_reward = 0
        trace = create_trace_skeleton([carry, door, door_open, goal], keys)
        for _ in range(max_episode_length):
            # print(s.shape)
            input_state = prepare_numpy(s, device)
            # input_state = prepare_tensor_batch(s)
            # input_state = s.to(device)
            if embedding_net:
                input_state = embedding_net(input_state)

            action_dist, action = policy(input_state)
            action_dist, action = action_dist[0], action[0]
            # Remove the batch dimension
            s_prime, r, t, info = env.step(action)
            carry, door = info['carry'], info['door']
            door_open, goal = info['door_open'], info['goal']
            # print(_, s_prime, r)
            if type(r) != float:
                print('run envs:', r, type(r))
            # print(input_state.dtype)
            trace = trace_accumulator(
                trace, [carry, door, door_open, goal], keys)
            act = torch.tensor([action*1.0], dtype=torch.float32).to(device)
            s1 = torch.cat((input_state, act), dim=1)
            # s1.requires_grad_(False)
            s1 = s1.cpu().detach().numpy()
            # print(_, s, action, r, s1)
            current_rollout.append(
                (s, action_dist.cpu().detach().numpy(), action, r, s1))
            # episode_reward += r
            if t:
                break
            s = s_prime['image']
            direction = s_prime['direction']
            s = np.reshape(s, (s.shape[0]*s.shape[1]*s.shape[2]))
            direction = np.array([direction])
            s = np.concatenate((s, direction))

        # New way to assign credit for multiple goals
        # goalspecs = [
        #     'F P_[C][True,none,==]',
        #     'F(P_[D][True,none,==])',
        #     'F(P_[DO][True,none,==])',
        #     'F(P_[G][True,none,==])'
        #     ]
        goalspecs = ['F P_[C][True,none,==]']
        r = 1   # len(goalspecs)
        for goalspec in goalspecs:
            rwd, current_rollout, result = calculate_returns(
                    current_rollout, gamma, trace, keys, goalspec, r)
            episode_reward += rwd
            # r -= 1
            if not result:
                break
        # print(current_rollout)
        # episode_reward, current_rollout, result = calculate_returns(
        #     current_rollout, gamma, trace, keys, goalspecs[0], 1.0
        # )
        experience_queue.put(current_rollout)
        reward_queue.put(episode_reward)


def prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)


def prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)


# def make_gif(rollout, filename):
#     with imageio.get_writer(filename, mode='I', duration=1 / 30) as writer:
#         for x in rollout:
#             writer.append_data((x[0][:, :, 0] * 255).astype(np.uint8)

class LossPlot:

    def __init__(
            self, directory, fname,
            title="Temporal-Credit-Assignment Performance",
            pname='temporal'):
        self.__dict__.update(locals())
        plt.style.use('fivethirtyeight')
        self.data = self.load_file()
        # print(dir(self.data))
        self.vloss = self.data['value_loss']
        self.ploss = self.data['policy_loss']
        self.avg_reward = self.data['avg_reward']
        self.data = [self.vloss, self.ploss, self.avg_reward]
        self.pname = '/' + pname

    def gen_plots(self):
        fig = plt.figure(figsize=[12, 20])
        color = ['orange', 'red', 'green']
        label = ['Value Loss', 'Policy Loss', 'Average Reward']
        ylabel = ['Loss', 'Loss', 'Avg Reward']
        plt.title(self.title)
        for i in range(0, 3):
            ax1 = fig.add_subplot(3, 1, i+1)
            ax1.plot(
                range(len(self.data[i])), self.data[i],
                color=color[i], label=label[i],
                linewidth=1.0)
            ax1.legend()
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel(ylabel[i])
            # ax1.set_yticks(
            # np.arange(min(self.data[i]), max(self.data[i])+1, 10.0))
            ax1.set_yticks(
                np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
            # ax1.set_title('Value Loss in Temporal Credit Assignment')
        plt.tight_layout()
        fig.savefig(
            '/tmp' + self.pname + '.pdf')  # pylint: disable = E1101
        fig.savefig(
            '/tmp' + self.pname + '.png')  # pylint: disable = E1101
        plt.close(fig)

    def load_file(self):
        data = pd.read_csv(
            self.directory + '/' + self.fname, sep=',', skipinitialspace=True)
        return data


# Trace Related Functions

def create_trace_flloat(traceset, i, keys):
    setslist = [create_sets(traceset[k][:i]) for k in keys]
    dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
    keydictlist = dict()
    j = 0
    for k in keys:
        keydictlist[k] = dictlist[j]
        j += 1
    t = FiniteTraceDict.fromDictSets(keydictlist)
    return t


def create_sets(trace):
    return [set([l]) for l in trace]    # noqa: E741


def create_trace_dict(trace, i, keys):
    tracedict = dict()
    for k in keys:
        tracedict[k] = trace[k][:i+1]
    return tracedict


def create_trace_skeleton(state, keys):
    # Create a skeleton for trace
    trace = dict(zip(keys, [list() for i in range(len(keys))]))
    j = 0
    for k in keys:
        trace[k].append(state[j])
        j += 1
    return trace


def trace_accumulator(trace, state, keys):
    for j in range(len(keys)):
        # Add the state variables to the trace
        # temp = trace[keys[j]][-1].copy()
        # temp.append(state[j])
        trace[keys[j]].append(state[j])
    return trace


def recognition(trace, keys, goalspec):
    # goalspec = 'F P_[C][True,none,==]'
    # parse the formula
    parser = LTLfGParser()

    # Define goal formula/specification
    parsed_formula = parser(goalspec)

    # Change list of trace to set
    traceset = trace.copy()
    akey = list(traceset.keys())[0]
    # print('recognizer', traceset)
    # Loop through the trace to find the shortest best trace
    for i in range(1, len(traceset[akey])+1):
        t = create_trace_flloat(traceset, i, keys)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, i

    return result, i


class KeyDoorEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(KeyDoorEnvironmentFactory, self).__init__()

    def new(self):
        return KeyDoorEnvironment()


class KeyDoorEnvironment(RLEnvironment):
    def __init__(self):
        super(KeyDoorEnvironment, self).__init__()
        env_name = 'MiniGrid-DoorKey-16x16-v0'
        # env_name = 'MiniGrid-DoorKey-8x8-v0'
        self._env = gym.make(env_name)
        self._env.max_steps = min(self._env.max_steps, 200)
        # self._env.seed(12345)
        # self.ereward = 0

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, _, t, info = self._env.step(action.item())
        # print(s, r)
        # self.ereward += r
        # if t:
        # Logic for status variable
        fwd_pos = self._env.front_pos
        carry = True if isinstance(self._env.carrying, Key) else False
        item = self._env.grid.get(fwd_pos[0], fwd_pos[1])
        door = True if isinstance(item, Door) else False
        door_open = item.is_open if isinstance(item, Door) else False
        goal = True if isinstance(item, Goal) else False
        info['carry'] = carry
        info['door'] = door
        info['door_open'] = door_open
        info['goal'] = goal
        return s, 0.0, t, info
        # else:
        #    return s, 0.0, t

    def reset(self):
        """Returns observation (np.ndarray)"""
        # self.ereward = 0
        return_dict = self._env.reset()
        return_dict['carry'] = False
        return_dict['door_open'] = False
        return_dict['door'] = False
        return_dict['goal'] = False
        # self._env.seed(12345)
        return return_dict


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
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


# def modify_KeyDoornet():
#     model = Generator()
#     model.load_state_dict(torch.load("KeyDoor_cnn.pt"))
#     for param in model.parameters():
#         param.requires_grad = False
#     # model = model.to(device)
#     return model


class KeyDoorPolicyNetwork(nn.Module):
    """Policy Network for KeyDoor."""

    def __init__(self, state_dim=148, action_dim=6):
        super(KeyDoorPolicyNetwork, self).__init__()
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
        Outputs action distribution
        (categorical distribution) of shape [batch, action_dim],
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
            # print(i, probs_np[i])
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
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.view((1, *src.shape))
        output = self.transformer_encoder(src)
        return output


# Temporal reward prediction value network
# Implemented from paper Temporal Credit assignment
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


# Recurrent neural network (many-to-one)
class Recognizer(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            num_classes, device='cpu'):
        super(Recognizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print('LSTM output',out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        return out


def ppo(env_factory, policy, value, likelihood_fn, embedding_net=None,
        epochs=100, rollouts_per_epoch=80, max_episode_length=20,
        gamma=0.99, policy_epochs=5, batch_size=50, epsilon=0.2,
        environment_threads=4, data_loader_threads=0,
        device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999),
        weight_decay=0.01, gif_name='', gif_epochs=0,
        csv_file='latest_run.csv', valueloss=nn.MSELoss(),
        stime=time.time()):

    directory = '/tmp/goal/data/experiments/'

    # Clear the csv file
    with open(directory + csv_file+'.csv', 'w') as f:
        f.write('avg_reward, value_loss, policy_loss\n')

    # Multi-processing
    # mp.set_start_method('spawn', force=True)

    # Move networks to the correct device
    # policy = policy.to(device)
    # policy.share_memory()
    value = value.to(device)
    # value.share_memory()
    recog = Recognizer(149, max_episode_length, 10, 2, device)
    recog = recog.to(device)
    # Collect parameters
    params = chain(policy.parameters(), value.parameters())
    if embedding_net:
        # embedding_net = embedding_net.to(device)
        # embedding_net.share_memory()
        params = chain(params, embedding_net.parameters())

    # Set up optimization
    # optimizer = optim.Adam(params, lr=lr, betas=betas,
    # weight_decay=weight_decay)
    optimizer = optim.Adam(params, lr=lr)
    value_criteria = valueloss

    optimizer_rec = optim.Adam(recog.parameters(), lr=lr)
    rec_criteria = torch.nn.CrossEntropyLoss()
    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    loop = tqdm(total=epochs, position=0, leave=False)

    # Prepare the environments
    environments = [env_factory.new() for _ in range(environment_threads)]
    rollouts_per_thread = rollouts_per_epoch // environment_threads
    remainder = rollouts_per_epoch % environment_threads
    rollout_nums = (
        [rollouts_per_thread + 1] * remainder) + (
            [rollouts_per_thread] * (environment_threads - remainder))

    for e in range(epochs):
        # embedding_net = embedding_net.to('cpu')
        policy = policy.to('cpu')
        # Run the environments
        experience_queue = Queue()
        reward_queue = Queue()
        threads = [
            Thread(
                target=run_envs, args=(
                    environments[i],
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
        # embedding_net,
        # policy,
        # experience_queue,
        # reward_queue,
        # 1, #rollout_nums[i],
        # max_episode_length,
        # gamma,
        # device)) for i in range(environment_threads)]
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
        # if gif_epochs and e % gif_epochs == 0:
        #     make_gif(rollouts[0], gif_name + '%d.gif' % e)

        # Move the network to GPU
        policy = policy.to(device)
        # embedding_net = embedding_net.to(device)
        # # Update the policy
        trace_len = [len(rollouts[i]) for i in range(len(rollouts))]
        avg_trace_leng = np.mean(trace_len)
        min_trace_leng = np.min(trace_len)
        max_trace_leng = np.max(trace_len)
        rec_dataset = RecognizerDataset(rollouts, max_episode_length)
        data_loader_rec = DataLoader(
            rec_dataset,
            num_workers=data_loader_threads, batch_size=max_episode_length,
            shuffle=False, pin_memory=True)

        experience_dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(
            experience_dataset,
            num_workers=data_loader_threads, batch_size=batch_size,
            shuffle=True, pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0

        for _ in range(policy_epochs):
            avg_policy_loss = 0
            avg_val_loss = 0
            avg_rec_loss = 0
            for state, old_action_dist, old_action, reward, ret, s1 in data_loader:     # noqa: E501
                state = prepare_tensor_batch(state, device)
                old_action_dist = prepare_tensor_batch(old_action_dist, device)
                old_action = prepare_tensor_batch(old_action, device)
                ret = prepare_tensor_batch(ret, device).unsqueeze(1)
                s1 = prepare_tensor_batch(s1, device)
                optimizer.zero_grad()
                # print(state.shape)
                if state.shape[0] != 40:
                    continue
                # If there is an embedding net, carry out the embedding
                if embedding_net:
                    # print(state.shape)
                    state = embedding_net(state)

                # Calculate the ratio term
                current_action_dist = policy(state, False)
                # print(current_action_dist.shape)
                current_likelihood = likelihood_fn(
                    current_action_dist, old_action)
                old_likelihood = likelihood_fn(old_action_dist, old_action)
                ratio = (current_likelihood / old_likelihood)

                # Calculate the value loss
                # print(s1.shape)
                expected_returns = value(s1)
                # print(expected_returns.shape, ret.shape)
                # print(expected_returns, ret)
                val_loss = value_criteria(expected_returns, ret)
                # val_loss = value_criteria(
                # expected_returns, reward.sum().detach())

                # Calculate the policy loss
                advantage = ret - expected_returns.detach()
                # print(ratio.shape, advantage.shape)
                lhs = ratio * advantage
                rhs = torch.clamp(
                    ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))

                # For logging
                avg_val_loss += val_loss.item()
                avg_policy_loss += policy_loss.item()

                # Backpropagate
                loss = policy_loss + val_loss
                loss.backward()
                optimizer.step()

            labels = torch.tensor(list(reward_queue.queue)).to(device)
            t = labels > 3
            f = labels <= 3
            labels[t] = 1
            labels[f] = 0

            # labels = torch.stack(
            #     [torch.tensor([label]) for label in labels]).to(device)
            datas = list(data_loader_rec)
            datas = torch.cat(datas, axis=1)
            datas = datas.to(device)
            datas = datas.view(datas.shape[1], datas.shape[0], datas.shape[2])
            # print(datas.shape, labels.shape)
            output = recog(datas)
            # print(output.shape, torch.tensor(labels[d]))
            # label = torch.tensor([labels[d]]).to(device)
            # print(output.shape, labels.shape)
            # print(output)
            # print(labels)
            rec_loss = rec_criteria(output, labels)
            avg_rec_loss += rec_loss.to('cpu').item()
            rec_loss.backward()
            optimizer_rec.step()

            # Log info
            avg_val_loss /= len(data_loader)
            avg_policy_loss /= len(data_loader)
            # avg_rec_loss /= len(labels)
            loop.set_description(
                'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f:' % (avg_r, avg_val_loss, avg_policy_loss))     # noqa: E501
        tdiff = time.time() - stime
        with open(directory + csv_file+'.csv', 'a+') as f:
            labels = labels.to(torch.float16)
            f.write(
                '%6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f\n' % (
                    torch.mean(labels), avg_val_loss, avg_policy_loss,
                    avg_rec_loss, avg_trace_leng, max_trace_leng,
                    min_trace_leng, tdiff))
        print()
        loop.update(1)
    modelname = directory + csv_file + ".pt"
    torch.save(policy.state_dict(), modelname)