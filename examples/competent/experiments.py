"""Competency experiments with MultiGoalGrid for the IJCAIpaper"""

from examples.competent.multigoalgrid import (
    MultiGoalGridExp)
from pygoal.utils.distribution import plot_competency_variance, logistfunc
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


def run_experiments(
        fn, name, train=True, runs=50, seed=None,
        parallel=False):
    if parallel:
        from joblib import Parallel, delayed
        datas = Parallel(
            n_jobs=4)(
                delayed(fn)(
                    i,
                    name=name+str(runs)+'_'+str(i)+'_',
                    train=train, seed=seed
                    ) for i in tqdm(range(runs)))
    else:
        datas = []
        for i in tqdm(range(runs)):
            datas += [fn(i, name, train, seed)]

    # print(datas, np.array(datas).shape)
    # Plot the data
    save_pickle(datas, name)
    plot_competency_variance(datas, name)
    return np.mean(datas, axis=0)


def exp_find_key(expid, name='exp_find_key', train=False, seed=None):
    goalspec = 'F P_[KE][1,none,==]'
    keys = [
        'LO', 'FW', 'KE']
    exp = MultiGoalGridExp(
        name+str(expid), goalspec, keys,
        actions=list(range(3)), seed=seed, maxtracelen=50,
        epoch=100, trainc=train)
    exp.run()
    # exp.draw_plot(['F(P_[KE][1,none,==])'], train=train)
    # exp.save_data()
    if train:
        return np.mean(
            exp.blackboard.shared_content[
             'ctdata']['F(P_[KE][1,none,==])'], axis=0)
    else:
        return np.mean(
            exp.blackboard.shared_content[
             'cidata']['F(P_[KE][1,none,==])'], axis=0)


def exp_find_key_avoid_lava(
        expid, name='exp_find_key_avoid_lava', train=False, seed=None):
    goalspec = '(F(P_[KE][1,none,==])) & (G(P_[LV][0,none,==]))'
    keys = [
        'LO', 'FW', 'KE', 'LV']
    exp = MultiGoalGridExp(
        name+str(expid), goalspec, keys,
        actions=list(range(3)), seed=seed, maxtracelen=50,
        epoch=100, trainc=train)
    exp.run()
    # exp.draw_plot(['F(P_[KE][1,none,==])'], train=train)
    # exp.save_data()
    if train:
        return np.mean(
            exp.blackboard.shared_content[
             'ctdata']['F(P_[KE][1,none,==])'], axis=0)
    else:
        return np.mean(
            exp.blackboard.shared_content[
             'cidata']['F(P_[KE][1,none,==])'], axis=0)
    # if train:
    #     popt = exp.blackboard.shared_content[
    #             'curve']['&']
    # else:
    #     popt = exp.blackboard.shared_content[
    #             'curve']['&']
    # return np.array(logistfunc(range(50+4), *popt))


def exp_carry_key(expid, name='exp_carry_key', train=False, seed=None):
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    keys = [
        'LO', 'FW', 'KE', 'CK']

    exp = MultiGoalGridExp(
        name+str(expid), goalspec, keys,
        actions=list(range(4)), seed=seed, maxtracelen=50,
        epoch=100, trainc=train)
    exp.run()
    if train:
        popt = exp.blackboard.shared_content[
                'curve']['U']
    else:
        popt = exp.blackboard.shared_content[
                'curve']['U']
    # print(popt)
    # print(
    #     exp.blackboard.shared_content['curve']['F(P_[KE][1,none,==])'],
    #     exp.blackboard.shared_content['curve']['F(P_[CK][1,none,==])'],
    #     exp.blackboard.shared_content['curve']['U'],
    #     )
    return np.array(logistfunc(range(50+4), *popt))
    # exp = MultiGoalGridExp('exp_carry_key', goalspec, keys)
    # exp.run()
    # root = exp.blackboard.shared_content['curve']['U']
    # exp.draw_plot(['F(P_[KE][1,none,==])', 'F(P_[CK][1,none,==])'], root)
    # exp.save_data()


def run_experiments_seed_batch(
        seeds=[3, 5, 7, 9], threads=4, fn=exp_find_key,
        expname='exp_fine_key_', runs=50, train=False):
    # Use parallel in here to speed up
    from joblib import Parallel, delayed
    if seeds is None:
        randseeds = list(np.random.randint(0, 100, 4))
    else:
        randseeds = seeds
    datas = Parallel(
        n_jobs=threads)(
            delayed(run_experiments)(
                fn,
                name=expname+str(runs)+'_'+str(i)+'_',
                runs=runs, train=train, seed=int(i)
                ) for i in randseeds)

    print('plotting main competency')
    save_pickle(datas, expname + str(runs)+'_all')
    plot_competency_variance(datas, expname + str(runs)+'_all')


def main():
    # exp_find_key()
    # exp_carry_key()
    # Folder naming pattern
    # exp_name_train/test_runs_epochs_tracelen_actionsSize
    # Total Samples: 5 * 80(epoch) * 40 (states)
    # run_experiments(
    #     exp_find_key, name='exp_find_key_5_', runs=5)

    # # Total Samples: 10 * 80(epoch) * 40 (states)
    # Experiment exp_find_key
    # run_experiments(
    #     exp_find_key, name='exp_find_key_50_',
    #     runs=100, parallel=True, seed=7, train=True)

    # Total Samples: 50 * 80(epoch) * 80 (states)
    # run_experiments_seed_batch()
    # run_experiments_seed_batch()
    # print(exp_carry_key(1, seed=9, train=True))

    # Experiment exp_carry_key
    # run_experiments_seed_batch(
    #     seeds=[3, 9], threads=2, fn=exp_carry_key,
    #     expname='exp_carry_key', runs=30, train=True)

    # Experiment exp_find_key_avoid_lava
    # exp_find_key_avoid_lava(1, seed=7)
    # Experiment exp_find_key_avoid_lava
    run_experiments(
        exp_find_key_avoid_lava, name='exp_find_key_avoid_lava_50_',
        runs=100, parallel=True, seed=7, train=True)


if __name__ == "__main__":
    main()
