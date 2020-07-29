"""Useful methods for PPO."""
import os
import torch
import math
# i#mport imageio
import numpy as np
from torch.utils.data import Dataset    # , DataLoader
import matplotlib
from pathlib import Path

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import pandas as pd

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


class RLEnvironment(object):
    """An RL Environment, used for wrapping environments to run PPO on."""

    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        """Takes an action x, which is the same format as the output
        from a policy network.
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


# def calculate_returns(trajectory, gamma):
#     current_return = 0
#     for i in reversed(range(len(trajectory))):
#         state, action_dist, action, reward = trajectory[i]
#         ret = reward + gamma * current_return
#         trajectory[i] = (state, action_dist, action, reward, ret)
#         current_return = ret


def calculate_returns(trajectory, gamma, trace, keys):
    # ret = finalrwd
    result, j = recognition(trace, keys)
    # print(result, j)
    ret = 1 if result is True else 0
    trajectory = trajectory[:j]
    episode_reward = 1 if result is True else 0
    for i in reversed(range(len(trajectory))):
        state, action_dist, action, rwd, s1 = trajectory[i]
        # print(i, state, action, rwd, s1)
        trajectory[i] = (state, action_dist, action, rwd, ret, s1)
        # print(i, ret, end=' ')
        ret = ret * gamma
    return episode_reward, trajectory


def run_envs(
        env, embedding_net, policy, experience_queue, reward_queue,
        num_rollouts, max_episode_length, gamma, device):
    keys = ['G', 'S']
    for _ in range(num_rollouts):
        current_rollout = []
        s, g, grid = env.reset()
        episode_reward = 0
        trace = create_trace_skeleton([g, grid], keys)
        for _ in range(max_episode_length):
            # print(s.shape)
            # input_state = prepare_numpy(s, device)
            # input_state = prepare_tensor_batch(s)
            input_state = s.to(device)
            # print(input_state.shape)
            if embedding_net:
                input_state = embedding_net(input_state)

            action_dist, action = policy(input_state)
            action_dist, action = action_dist[0], action[0]
            # Remove the batch dimension
            s_prime, r, t, goal = env.step(action)
            # print(_, s_prime, r)
            if type(r) != float:
                print('run envs:', r, type(r))
            # print(input_state.dtype)
            trace = trace_accumulator(trace, [goal[0], goal[1]], keys)
            act = torch.tensor([action*1.0], dtype=torch.float32).to(device)
            s1 = torch.cat((input_state, act), dim=1)
            current_rollout.append(
                (
                    s.squeeze(0), action_dist.cpu().detach().numpy(),
                    action, r, s1))
            # episode_reward += r
            if t:
                break
            s = s_prime

        episode_reward, current_rollout = calculate_returns(
            current_rollout, gamma, trace, keys)
        # print(current_rollout)
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
            self.directory + '/' + self.fname, sep=',',  # pylint: disable=E501
            skipinitialspace=True)
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
    return [set([trc]) for trc in trace]


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


def recognition(trace, keys):
    goalspec = 'F P_[S][9,none,==]'
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
        t = create_trace_flloat(traceset, i, keys)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, i

    return result, i


def load_file_all(directory, fname):
    data = np.loadtxt(
            os.path.join(directory, fname),
            delimiter=',', unpack=True, skiprows=1)
    # data = np.sum(data == i, axis=1)
    # print(data)
    return data.T


def load_files_all(directory, fnames):
    # Find all the files with matching fname
    path = Path(directory)
    files = path.glob(fnames)
    data = [load_file_all(directory, f) for f in files]
    # print(data[0].shape)
    data = np.stack(data)
    return data


def filter_data(data, i):
    # mean = np.mean(data[:, :, i], axis=0)
    # std = np.std(data[:, :, i], axis=0)

    median = np.quantile(data[:, :, i], 0.5, axis=0)
    q1 = np.quantile(data[:, :, i], 0.25, axis=0)
    q3 = np.quantile(data[:, :, i], 0.75, axis=0)

    # return mean, std
    return median, q1, q3


# def draw_losses():

#     datas = load_files_all('/tmp', 'mnist_2_*')
#     mean, std = filter_data(datas, 0)


def draw_trace_data(data, pname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = ['blue', 'purple', 'gold']
    colorshade = ['DodgerBlue', 'plum', 'khaki']
    label = ['Mean', 'Max', 'Min']

    idx = [4, 5, 6]
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(3):
        # mean, std = filter_data(data, idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data, idx[i])
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Average Trace Length')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('$\it{m}$')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_success_prob(data, pname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = ['forestgreen']
    colorshade = ['springgreen']
    label = ['Mean']

    idx = [0]
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(1):
        mean, field_min, field_max = filter_data(data, idx[i])
        # mean, std = filter_data(data, idx[i])
        # field_max = mean + std
        # field_min = mean - std
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Goal Success Probability')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Probability')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_success_comp(data, pname, lb):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # label = ['20', '30', '40', '50', '60']
    # label = ['50', '60', '70', '80', '90']
    label = [str(l) for l in lb]
    idx = [0] * 5
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(5):
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data[i], idx[i])
        mean = mean[:30]
        field_max = field_max[:30]
        field_min = field_min[:30]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Goal Success Probability')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Probability')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_trace_comp(data, pname, lb):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    # color = ['blue', 'purple', 'gold']
    # colorshade = ['DodgerBlue', 'plum', 'khaki']
    # label = ['Mean', 'Max', 'Min']
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # label = ['20', '30', '40', '50', '60']
    # label = ['50', '60', '70', '80', '90']
    label = [str(l) for l in lb]
    # idx = [4, 5, 6]
    idx = [4] * 5
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(5):
        mean, field_min, field_max = filter_data(data[i], idx[i])
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean = mean[:30]
        field_max = field_max[:30]
        field_min = field_min[:30]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Average Trace Length')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('$\it{m}$')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_time_comp(data, pname, lb):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    # color = ['blue', 'purple', 'gold']
    # colorshade = ['DodgerBlue', 'plum', 'khaki']
    # label = ['Mean', 'Max', 'Min']
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # label = ['20', '30', '40', '50', '60']
    label = [str(l) for l in lb]

    # idx = [4, 5, 6]
    idx = [7] * 5
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(5):
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data[i], idx[i])
        mean = mean[:15]
        field_max = field_max[:15]
        field_min = field_min[:15]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    # plt.title('Trace Length')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Computation Time')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_action_comp(data, pname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    label = ['2', '4', '6', '8', '10']

    idx = [0] * 5
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(5):
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data[i], idx[i])
        mean = mean[:25]
        field_max = field_max[:25]
        field_min = field_min[:25]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Goal Success Probability')
    ax1.legend(title='Action Space')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Probability')

    # ax1.set_yticks(
    #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        '/tmp/goal/data/experiments/' + pname + '.png')
    plt.close(fig)


def sucess_comparasion(tl=[50, 60, 70, 80, 90], a=10):
    datas = []
    for i in tl:
        name = '_' + str(a) + '_' + str(i)
        data = load_files_all('/tmp/goal/data/experiments', 'mnist'+name+'_*')
        datas.append(data)
    draw_success_comp(datas, 'success_' + str(a) + '__', tl)


def trace_comparasion(tl=[50, 60, 70, 80, 90], a=10):
    datas = []
    for i in tl:
        name = '_' + str(a) + '_' + str(i)
        # print(name)
        data = load_files_all('/tmp/goal/data/experiments', 'mnist'+name+'_*')
        datas.append(data)
    draw_trace_comp(datas, 'trace_' + str(a) + '__', tl)


def time_comparasion(tl=[50, 60, 70, 80, 90], a=4):
    datas = []
    for i in [20, 30, 40, 50, 60]:
        name = '_2_' + str(i)
        data = load_files_all('/tmp/goal/data/experiments', 'mnist'+name+'_*')
        datas.append(data)
    draw_time_comp(datas, 'mnist_2_ti', tl)


def action_comparasion():
    datas = []
    for i in [2, 4, 6, 8, 10]:
        name = '_' + str(i) + '_50'
        # print(name)
        data = load_files_all('/tmp/goal/data/experiments', 'mnist'+name+'_*')
        datas.append(data)
    draw_action_comp(datas, 'mnist_2_a')


def results():
    action_comparasion()
    name = '_2_70'
    datas = load_files_all('/tmp/goal/data/experiments', 'mnist'+name+'_*')
    draw_trace_data(datas, 'traces'+name)
    draw_success_prob(datas, 'sucess'+name)
