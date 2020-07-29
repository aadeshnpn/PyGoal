"""Learn policy for MDP world using GenRecProp algorithm."""
import os
from pathlib import Path
import numpy as np
from py_trees.trees import BehaviourTree
from py_trees.common import Status

from pygoal.lib.mdp import GridMDP
from pygoal.lib.genrecprop import GenRecPropMDP     # GenRecPropMDPNear
from pygoal.utils.bt import goalspec2BT, reset_env
from joblib import Parallel, delayed

import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
# matplotlib.rc('text', usetex=True)


def load_files(directory, fnames, fn):
    # Find all the files with matching fname
    path = Path(directory)
    files = path.glob(fnames + '*.npy')
    datas = [fn(directory, f) for f in files]
    datas = np.stack(datas)
    return datas


def load_file_mdp_prob(directory, fname):
    data = np.load(
        os.path.join(directory, fname)
    )
    return data


def init_mdp(sloc):
    """Initialized a simple MDP world."""
    grid = np.ones((4, 4)) * -0.04

    # Obstacles
    grid[3][0] = None
    grid[2][2] = None
    grid[1][1] = None

    # Cheese and trap
    grid[0][3] = None
    grid[1][3] = None

    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()

    # Terminal and obstacles defination
    grid[0][3] = +2
    grid[1][3] = -2

    mdp = GridMDP(
        grid, terminals=[(3, 3), (3, 2)], startloc=sloc)

    return mdp


def init_10x10mdp(sloc):
    """Initialized a 10x10 MDP world."""
    grid = np.ones((10, 10)) * -0.04
    grid[1:8, 2:6] = None
    grid = np.where(np.isnan(grid), None, grid)
    grid = grid.tolist()
    # Terminal and obstacles defination
    grid[0][9] = +2
    grid[1][9] = -2
    grid[0][7] = -2

    mdp = GridMDP(
        grid, terminals=[(9, 9), (9, 8), (7, 9)], startloc=sloc)

    return mdp


# def get_near_trap(seed):
#     # Define the environment for the experiment
#     goalspec = 'F P_[NT][True,none,==]'
#     startpoc = (3, 0)
#     # env1 = init_mdp(startpoc)
#     env2 = init_mdp(startpoc)
#     keys = ['L', 'IC', 'IT', 'NC', 'NT']
#     actions = [0, 1, 2, 3]
#     gmdp = GenRecPropMDPNear(env2, keys, goalspec,
#       dict(), 30, actions, False)
#     gmdp.gen_rec_prop(100)

#     policy = create_policy(gmdp.gtable)

#     print(gmdp.run_policy(policy))


# def find_cheese_return(seed):
#     # Define the environment for the experiment
#     goalspec = 'F P_[IC][True,none,==] U F P_[L][13,none,==]'
#     startpoc = (1, 3)
#     # env1 = init_mdp(startpoc)
#     env2 = init_mdp(startpoc)
#     keys = ['L', 'IC', 'IT', 'NC', 'NT']
#     actions = [0, 1, 2, 3]
#     gmdp = GenRecPropMDPNear(env2, keys, goalspec,
#       dict(), 30, actions, False)
#     gmdp.gen_rec_prop(100)

#     policy = create_policy(gmdp.gtable)

#     print(gmdp.run_policy(policy))


# def find_cheese_return(seed, max_trace_len=10):
#     # Define the environment for the experiment
#     goalspec = 'F P_[NC][True,none,==] U F P_[L][03,none,==]'
#     startpoc = (0, 3)
#     env = init_mdp(startpoc)
#     keys = ['L', 'NC']
#     actions = [0, 1, 2, 3]

#     root = goalspec2BT(goalspec, planner=None)
#     # print(root)
#     behaviour_tree = BehaviourTree(root)
#     # # Need to udpate the planner parameters
#     child = behaviour_tree.root
#     for child in behaviour_tree.root.children:
#         print(child, child.name, env.curr_loc)
#         planner = GenRecPropMDPNear(
#             env, keys, None, dict(),
#             actions=actions, max_trace=max_trace_len)
#         child.setup(0, planner, True, 10)

#     for i in range(10):
#         behaviour_tree.tick(
#             pre_tick_handler=reset_env(env)
#         )
#         # print(behaviour_tree.root.status)

#     for child in behaviour_tree.root.children:
#         child.setup(0, planner, True, 10)
#         child.train = False
#         print(child, child.name, child.train)
#     print('before inference start', env.curr_loc)
#     for i in range(1):
#         behaviour_tree.tick(
#             pre_tick_handler=reset_env(env)
#         )
#     print('inference', behaviour_tree.root.status)
#     print(env.curr_loc)


def check_bt_status(status):
    result = 0
    if status == Status.SUCCESS:
        result = 1
    else:
        result = 0
    return result


def find_cheese(seed, max_trace_len=10, epoch=10):
    # Define the environment for the experiment
    goalspec = 'F P_[IC][True,none,==]'
    # startpoc = (3, 0)
    startpoc = (9, 0)
    env = init_10x10mdp(startpoc)
    keys = ['L', 'IC']
    actions = [0, 1, 2, 3]

    root = goalspec2BT(goalspec, planner=None)
    # print(root)
    behaviour_tree = BehaviourTree(root)
    # display_bt(behaviour_tree, True)
    # print(dir(behaviour_tree))
    # # Need to udpate the planner parameters
    child = behaviour_tree.root
    # for child in behaviour_tree.root.children:
    # print(child, child.name, env.curr_loc)
    planner = GenRecPropMDP(
        env, keys, None, dict(), actions=actions,
        max_trace=max_trace_len, epoch=epoch)
    child.setup(0, planner, True, epoch=epoch)
    # Experiment data
    # print(planner.trace_len_data)
    data = np.zeros(epoch, dtype=np.uint8)
    for i in range(epoch):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
        # print(behaviour_tree.root.status)
        data[i] = check_bt_status(behaviour_tree.root.status)
    # print(planner.trace_len_data)
    # for child in behaviour_tree.root.children:
    child.setup(0, planner, True, max_trace_len)
    child.train = False
    # print(child, child.name, child.train)

    for i in range(1):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    # data[-1] = check_bt_status(behaviour_tree.root.status)
    # print('inference', behaviour_tree.root.status)
    # print(planner.trace_len_data)
    # print(data)
    # print(env.curr_loc)
    return (data, planner.trace_len_data)


def run_experiments():
    dname = os.path.join('/tmp', 'mdp', 'data', 'experiments')
    Path(dname).mkdir(parents=True, exist_ok=True)
    # trace = [10, 20, 30, 40, 50]
    # trace = [30, 40, 50, 60, 70]
    trace = [70, 80, 90, 100, 110]
    for k in range(len(trace)):
        for j in range(50):
            fname = 'mdp10_' + str(trace[k]) + '_' + str(j)
            tname = 'mdp10_t_' + str(trace[k]) + '_' + str(j)
            print(fname)
            datas = Parallel(
                n_jobs=16)(
                    delayed(find_cheese)(
                        None,
                        max_trace_len=trace[k],
                        epoch=100
                    ) for i in range(64))
            # data = [d[0] for d in data]
            # print(len(datas))
            probdata = []
            tdata = []
            for i in range(len(datas)):
                probdata.append(datas[i][0])
                tdata.append(datas[i][1])
            # For success prob
            fname = os.path.join(dname, fname)
            data = np.array(probdata)
            data = data.astype(np.float16)
            # print(k, j, data)
            data = np.mean(data, axis=0)

            np.save(fname, data)

            # For trace len data
            tname = os.path.join(dname, tname)
            tdata = np.array(tdata)
            tdata = tdata.astype(np.float16)
            # print(k, j, tdata)
            tdata = np.quantile(tdata, 0.5, axis=0)
            # print(k, j, tdata)
            np.save(tname, tdata)
        # find_cheese(None, max_trace_len=10, epoch=10)


def filter_data(data):
    mean = np.mean(data, axis=0)
    # median = np.quantile(data, 0.5, axis=0)
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)

    # return mean, std
    return mean, q1, q3


def draw_success_prob(data, tracelist, pname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    label = [str(tracelen) for tracelen in tracelist]
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        mean, field_min, field_max = filter_data(data[i])
        # mean = mean[:15]
        # field_max = field_max[:15]
        # field_min = field_min[:15]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Goal Success Probability \n0.2 action uncertainty')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Probability')

    plt.tight_layout()
    fig.savefig(
        '/tmp/mdp/data/experiments/' + pname + '.png')
    plt.close(fig)


def draw_trace_len(data, tracelist, pname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    label = [str(tracelen) for tracelen in tracelist]
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        mean, field_min, field_max = filter_data(data[i])
        # mean = mean[:15]
        # field_max = field_max[:15]
        # field_min = field_min[:15]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    plt.title('Average Trace length \n0.2 action uncertainty')
    ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('$\it{m}$')

    plt.tight_layout()
    fig.savefig(
        '/tmp/mdp/data/experiments/' + pname + '.png')
    plt.close(fig)


def plot_all():
    # tracelenlist = [10, 20, 30, 40, 50]
    # tracelenlist = [30, 40, 50, 60, 70]
    tracelenlist = [70, 80, 90, 100, 110]
    datas = []
    for trace in tracelenlist:
        expname = 'mdp10_' + str(trace)
        # After the experiments are done, draw plots
        directory = os.path.join('/tmp', 'mdp', 'data', 'experiments')
        data = load_files(
            directory, expname, load_file_mdp_prob)
        # print(trace, data.shape)
        datas.append(data)
    draw_success_prob(datas, tracelenlist, 'plot_'+expname)

    datas = []
    for trace in tracelenlist:
        expname = 'mdp10_t_' + str(trace)
        # After the experiments are done, draw plots
        directory = os.path.join('/tmp', 'mdp', 'data', 'experiments')

        data = load_files(
            directory, expname, load_file_mdp_prob)
        print(trace, data.shape)
        # print(np.max(data))
        data = np.clip(data, 0, np.floor(np.max(data)//10)*10)
        # print(data)
        datas.append(data)
    draw_trace_len(datas, tracelenlist, 'plot_'+expname)


def main():
    # find_cheese(None, 10, 2)
    # find_cheese_return(123)
    run_experiments()
    # plot_all()


main()
