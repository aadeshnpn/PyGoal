"""Competency experiments with MultiGoalGrid for the IJCAIpaper"""

from examples.competent.multigoalgrid import (
    MultiGoalGridExp)
from pygoal.utils.distribution import plot_competency_variance
import numpy as np
from tqdm import tqdm


def save_pickle(data, name):
    import pathlib
    import os
    dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(dname, name + '.pkl')
    import pickle
    pickle.dump(data, open(fname, "wb"))


def run_experiments(fn, name, train=True, runs=50):
    datas = []
    for i in tqdm(range(runs)):
        datas += [fn(i, name, train)]

    # print(datas, np.array(datas).shape)
    # Plot the data
    save_pickle(datas, name)
    plot_competency_variance(datas, name)


def exp_find_key(expid, name='exp_find_key', train=False):
    goalspec = 'F P_[KE][1,none,==]'
    keys = [
        'LO', 'FW', 'KE']
    exp = MultiGoalGridExp(
        name+str(expid), goalspec, keys,
        actions=list(range(7)), seed=None, maxtracelen=50,
        epoch=100, trainc=train)
    exp.run()
    exp.draw_plot(['F(P_[KE][1,none,==])'], train=train)
    exp.save_data()
    if train:
        return np.mean(
            exp.blackboard.shared_content[
             'ctdata']['F(P_[KE][1,none,==])'], axis=0)
    else:
        return np.mean(
            exp.blackboard.shared_content[
             'cidata']['F(P_[KE][1,none,==])'], axis=0)


def exp_carry_key():
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    keys = [
        'LO', 'FW', 'KE', 'CK']
    exp = MultiGoalGridExp('exp_carry_key', goalspec, keys)
    exp.run()
    root = exp.blackboard.shared_content['curve']['U']
    exp.draw_plot(['F(P_[KE][1,none,==])', 'F(P_[CK][1,none,==])'], root)
    exp.save_data()


def main():
    # exp_find_key()
    # exp_carry_key()
    # Folder naming pattern
    # exp_name_train/test_runs_epochs_tracelen_actionsSize
    # Total Samples: 5 * 80(epoch) * 40 (states)
    # run_experiments(
    #     exp_find_key, name='exp_find_key_5_', runs=5)

    # # Total Samples: 10 * 80(epoch) * 40 (states)
    # run_experiments(
    #     exp_find_key, name='exp_find_key_10_', runs=10)

    # Total Samples: 50 * 80(epoch) * 80 (states)
    run_experiments(
        exp_find_key, name='exp_find_key_10_', runs=20, train=True)


if __name__ == "__main__":
    main()
