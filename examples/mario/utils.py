"""Useful methods for PPO."""
import torch
import math
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict



class RLEnvironment(object):
    """An RL Environment, used for wrapping environments to run PPO on."""

    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        """Takes an action x, which is the same format as the output from a policy network.
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


def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def get_log_p(data, mu, sigma):
    """get negative log likelihood from normal distribution"""
    return -torch.log(torch.sqrt(2 * math.pi * sigma ** 2)) - (data - mu) ** 2 / (2 * sigma ** 2)


# def calculate_returns(trajectory, gamma):
#     current_return = 0
#     for i in reversed(range(len(trajectory))):
#         state, action_dist, action, reward = trajectory[i]
#         ret = reward + gamma * current_return
#         trajectory[i] = (state, action_dist, action, reward, ret)
#         current_return = ret


# def calculate_returns(trajectory, gamma, trace, keys):
#     # ret = finalrwd
#     tlen = len(trace)
#     result, j = recognition(trace, keys)
#     if result is False:
#         # print(result, j)
#         # ret = 1 if result == True else 0
#         # trajectory = trajectory[:j]
#         # episode_reward = 1 if result == True else 0
#         # episode_reward = trace
#         ret = -1
#         for i in reversed(range(len(trajectory))):
#             state, action_dist, action, rwd, s1 = trajectory[i]
#             # print(i, state, action, rwd, s1)
#             trajectory[i] = (state, action_dist, action, rwd, ret, s1)
#             # print(i, ret, end=' ')
#             ret = ret * gamma
#     else:
#         if result is True and  j+1 >= tlen:
#             pass
#     return episode_reward, trajectory

def slice_trace(j, trace, keys):
    for k in keys:
        trace[k] = trace[k][j:]
    return trace


def temp_fn(gamma, ret, trajectory):
    for i in reversed(range(len(trajectory))):
        state, action_dist, action, rwd, s1 = trajectory[i]
        trajectory[i] = (state, action_dist, action, rwd, ret, s1)
        ret = ret * gamma
    return trajectory

def calculate_returns(trajectory, gamma, trace, keys):
    tlen = len(trajectory)
    result, j = recognition(trace, keys)
    trajectory = trajectory[:j]
    if result is False:
        return temp_fn(gamma, 0.0, trajectory)
    else:
        return temp_fn(gamma, 1.0, trajectory)
    return trajectory
    # if result is False:
    #     ret = 0.0
    #     return temp_fn(gamma, ret, trajectory)
    # else:
    #     if result is True and  j+1 >= tlen:
    #         ret = 1.0
    #         return temp_fn(gamma, ret, trajectory)
    #     else:
    #         ret = 1.0
    #         traj = temp_fn(gamma, ret, trajectory[:j])
    #         traj += calculate_returns(trajectory[j:], gamma, slice_trace(j+1, trace, keys), keys)
    #         return traj
    # print(trajectory)
    # return trajectory

def run_envs(env, embedding_net, policy, experience_queue, reward_queue,
                num_rollouts, max_episode_length, gamma, device):
    keys = ['C']
    for _ in range(num_rollouts):
        current_rollout = []
        s = env.reset()
        # print(len(statedict), statedict)
        # s, direction, carry = statedict['image'], statedict['direction'], statedict['carry']
        s = np.reshape(s, (s.shape[0]*s.shape[1]*s.shape[2]))
        # 8640
        # direction = np.array([direction])
        # s = np.concatenate((s,direction))
        episode_reward = 0
        trace = create_trace_skeleton([0], keys)
        coin = 0
        for _ in range(max_episode_length):
            # print(s.shape)
            # print(s.shape)
            input_state = prepare_numpy(s, device)
            # print(input_state.shape)
            # input_state = prepare_tensor_batch(s)
            # input_state = s.to(device)
            if embedding_net:
                input_state = embedding_net(input_state)
            # print(input_state.shape)
            action_dist, action = policy(input_state)
            action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
            s_prime, r, t, coins = env.step(action)
            coin += coins
            # print(_, s_prime, r)
            if type(r) != float:
                print('run envs:', r, type(r))
            # print(input_state.dtype)
            trace = trace_accumulator(trace, [coins], keys)
            act = torch.tensor([action*1.0], dtype=torch.float32).to(device)
            s1 = torch.cat((input_state, act), dim=1)
            # s1.requires_grad_(False)
            s1 = s1.cpu().detach().numpy()
            # print(_, s, action, r, s1)
            current_rollout.append((s, action_dist.cpu().detach().numpy(), action, r, s1))
            # episode_reward += r
            if t:
                break
            s = s_prime
            s = np.reshape(s, (s.shape[0]*s.shape[1]*s.shape[2]))
            # direction = np.array([direction])
            # s = np.concatenate((s,direction))
        # print(current_rollout, gamma, episode_reward)
        # if episode_reward == 0:
        #     episode_reward = -1
        # elif episode_reward == 1:
        #     episode_reward = 100
        episode_reward = coin
        current_rollout = calculate_returns(current_rollout, gamma, trace, keys)
        # print(current_rollout)
        experience_queue.put(current_rollout)
        reward_queue.put(episode_reward)


def prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device) / 255.0
    # return torch.from_numpy(ndarray).float().to(device)


def prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)


def make_gif(rollout, filename):
    with imageio.get_writer(
            filename, mode='I', duration=1 / 30) as writer:
        for x in rollout:
            writer.append_data((x[0][:, :, 0] * 255).astype(np.uint8))


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
            # ax1.set_yticks(np.arange(min(self.data[i]), max(self.data[i])+1, 10.0))
            ax1.set_yticks(np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
            # ax1.set_title('Value Loss in Temporal Credit Assignment')
        plt.tight_layout()
        fig.savefig(
            '/tmp' + self.pname + '.pdf')  # pylint: disable = E1101
        fig.savefig(
            '/tmp' + self.pname + '.png')  # pylint: disable = E1101
        plt.close(fig)

    def load_file(self):
        data = pd.read_csv(
            self.directory + '/' + self.fname, sep=','  # pylint: disable=E1101
            , skipinitialspace=True)
        return data


#### Trace Related Functions

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
    return [set([l]) for l in trace]


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
    goalspec = 'F P_[C][1,none,>=]'
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
