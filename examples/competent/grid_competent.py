"""Compute Competency in a GridWorld"""

import os
from pathlib import Path
import numpy as np
from py_trees.trees import BehaviourTree
from py_trees.common import Status

from pygoal.lib.genrecprop import GenRecPropMDP     # GenRecPropMDPNear
from pygoal.utils.bt import goalspec2BT, reset_env
from joblib import Parallel, delayed

import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
# matplotlib.rc('text', usetex=True)

import gym
import gym_minigrid
from gym_minigrid.wrappers import ReseedWrapper, OneHotPartialObsWrapper
from gym_minigrid.minigrid import (
    Grid, OBJECT_TO_IDX, Key, Door, Goal, COLOR_TO_IDX
)
from gym import error, spaces, utils

import time


def main():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = OneHotPartialObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    env.render(mode='human')
    state, reward, done, _ = env.step(2)
    print(state['image'].shape, reward, done, _)
    time.sleep(15)


if __name__ == "__main__":
    main()