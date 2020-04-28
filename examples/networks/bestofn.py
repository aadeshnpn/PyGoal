"""Experiment for best-of-N problem with Graph.

Expreriments and results for best-of-N
problem with Graph
"""

import copy
import numpy as np
import pickle
# from joblib import Parallel, delayed

from graphenv import GraphBestofNEnvironment

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt
from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


def env_setup():
    num_agent = 10
    num_site = 2
    attach_mode = 'importance 2 linear' # choices for attach interface are' always', 'linear', 'exponential', 'importance linear', 'importance 2 linear', 'importance exponential' or \'importance 2 exponential\'
    detach_mode = 'power law' # choices are 'uniform', 'linear', 'exponential', 'power law', or 'perfect'
    seed = 1234
    env = GraphBestofNEnvironment(num_agent, num_site, attach_mode, detach_mode, seed=seed)
    # env.seed(seed)
    env.reset()
    return env


def main():
    env = env_setup()
    for epoch in range(100):
        env.step()
        env.showGraph()


if __name__ == "__main__":
    main()
