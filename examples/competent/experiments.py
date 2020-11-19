"""Competency experiments with MultiGoalGrid for the IJCAIpaper"""

from examples.competent.multigoalgrid import (
    MultiGoalGridExp)
import numpy as np
import os
import matplotlib
# If there is $DISPLAY, display the plot
if os.name == "posix" and "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# plt.style.use("fivethirtyeight")


def draw_plots(data, name):
    mean = np.mean(data, axis=1)
    # Using quartile instead of mean and std
    median = np.quantile(data, 0.5, axis=1)
    q1 = np.quantile(data, 0.25, axis=1)
    q3 = np.quantile(data, 0.75, axis=1)
    fig = plt.figure()
    # field_max = self.median + self.std
    # field_min = self.mean - self.std
    xvalues = range(1, len(median) + 1)
    ax1 = fig.add_subplot(1, 1, 1)
    # Plotting mean and standard deviation
    ax1.plot(
        xvalues,
        median,
        color="blue",
        label="Median",
        linestyle="dashdot",
        linewidth=1.0,
    )

    ax1.plot(xvalues, mean, color="mediumblue", label="Mean", linewidth=1.0)

    ax1.fill_between(xvalues, q1, q3, color="DodgerBlue", alpha=0.3)
    plt.xlim(0, len(median))
    ax1.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Probability")

    plt.tight_layout()
    import pathlib
    import os
    dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(dname, name + '.png')
    fig.savefig(fname)
    plt.close(fig)


def save_pickle(data, name):
    import pathlib
    import os
    dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(dname, name + '.pkl')
    import pickle
    pickle.dump(data, open(fname, "wb"))


def run_experiments(fn, name, runs=50):
    datas = []
    for i in range(runs):
        datas += [fn(i, name)]

    print(datas, np.array(datas).shape)
    # Plot the data
    save_pickle(datas, name)
    draw_plots(datas, name)


def exp_find_key(expid, name='exp_find_key'):
    goalspec = 'F P_[KE][1,none,==]'
    keys = [
        'LO', 'FW', 'KE']
    exp = MultiGoalGridExp(name+str(expid), goalspec, keys, seed=3)
    exp.run()
    exp.draw_plot(['F(P_[KE][1,none,==])'])
    exp.save_data()
    return np.mean(
        exp.blackboard.shared_content[
           'ctdata']['F(P_[KE][1,none,==])'], axis=0)


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
    run_experiments(
        exp_find_key, name='exp_find_key', runs=5)


if __name__ == "__main__":
    main()
