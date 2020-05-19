"""MNIST Grid Environemnt."""
import gym
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(
            '../data', train=False, download=True,
            transform=transforms.Compose([  # noqa: E128
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
            ]))

    def __getitem__(self, index):
        return self.mnist[index]

    def __len__(self):
        return 500  # len(self.mnist)


class MNISTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed):
        self.nprandom = np.random.RandomState(seed)
        self.data = list(MNIST())
        self.images, self.labels = zip(*self.data)
        self.idxs = []
        self.state = 0
        self.labels = torch.tensor(self.labels)
        self.goal = False
        for i in range(10):
            idx = torch.where(self.labels == i)[0].data.numpy()
            # print(idx, idx.shape)
            self.idxs.append(idx)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        done = False
        reward = 0.0
        self._take_action(action)
        if self.state == 9:
            done = True
            self.goal = True
            reward = 0.0
        # self.status = self.env.step()
        # reward = self._get_reward()
        # ob = self.env.getState()
        # episode_over = self.status != hfo_py.IN_GAME
        return self.get_images(self.state), reward, done, {'goal': self.goal, 'grid':self.state}

    def reset(self):
        self.state = 0
        self.goal = False
        return self.get_images(self.state), self.goal, self.state

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        if action == 0:
            new_state = self.state - 1
        elif action == 1:
            new_state = self.state + 1
        else:
            new_state = self.state
        new_state = np.clip(new_state, 0, 9)
        self.state = new_state

    def get_images(self, label):
        i = self.nprandom.choice(self.idxs[label], replace=False)
        image = self.images[i]
        return image.view(1, *image.shape)

    def _get_reward(self):
        pass


# def main():
#     env = MNISTEnv(3)
#     s = env.reset()
#     print(env.state)
#     for i in range(10):
#         s, r, _ , _ = env.step(1)
#         # print(s, r)
#         print(env.state)


# if __name__ == '__main__':
#     main()
