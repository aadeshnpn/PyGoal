import os
import numpy as np
from pathlib import Path
# from keydoor_temporal import KeyDoorEnvironmentFactory
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402


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
    fnames = fnames+'.csv'
    # print(fnames)
    files = path.glob(fnames)
    # print(list(files))
    data = [load_file_all(directory, f) for f in files]
    # rindx = [i for i in range(len(data)) if data[i].shape[0] != 100]
    # print(rindx)
    # for indx in rindx:
    #     data.remove(indx)
    # for i in range(len(data)):
    #     if data[i].shape[0] != 100:
    #         print(i)
    #         data.remove(i)
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

    # plt.title('Trace Length')
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Trace Length')

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
    ax1.legend()
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


def draw_success_comp(data, pname, tl):
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # label = ['20', '30', '40', '50', '60']
    # label = ['30', '40', '50', '60', '70']
    # label = ['60', '70', '80', '90', '100']
    label = [str(t) for t in tl]
    idx = [0] * len(data)
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data[i], idx[i])
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

    plt.title('Goal Success Probability \n KeyDoor16x16')
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


def draw_trace_comp(data, pname, tl):
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
    # label = ['30', '40', '50', '60', '70']
    # label = ['60', '70', '80', '90', '100']
    label = [str(t) for t in tl]
    # idx = [4, 5, 6]
    idx = [4] * len(data)
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        mean, field_min, field_max = filter_data(data[i], idx[i])
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
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

    plt.title('Average Trace Length \n KeyDoor16x16')
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


def draw_time_comp(data, pname, tl):
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
    # label = ['30', '40', '50', '60', '70']
    # label = ['60', '70', '80', '90', '100']
    label = [str(t) for t in tl]
    # idx = [4, 5, 6]
    # print(len(data))
    idx = [7] * len(data)
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        # mean, std = filter_data(data[i], idx[i])
        # field_max = mean + std
        # field_min = mean - std
        mean, field_min, field_max = filter_data(data[i], idx[i])
        # print(mean.shape, field_min.shape, field_max.shape)
        # mean = mean[:25]
        # field_max = field_max[:25]
        # field_min = field_min[:25]
        xvalues = range(1, len(mean) + 1)

        # Plotting mean and standard deviation
        ax1.plot(
            xvalues, mean, color=color[i], label=label[i],
            linewidth=1.0)
        ax1.fill_between(
            xvalues, field_max, field_min,
            color=colorshade[i], alpha=0.3)

    # plt.title('Trace Length')
    ax1.legend()
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


# def draw_action_comp(data, pname):
#     plt.style.use('fivethirtyeight')
#     fig = plt.figure()
#     color = [
#         'forestgreen', 'indianred',
#         'gold', 'tomato', 'royalblue']
#     colorshade = [
#         'springgreen', 'lightcoral',
#         'khaki', 'lightsalmon', 'deepskyblue']
#     label = ['2', '4', '6', '8', '10']

#     idx = [0] * 5
#     ax1 = fig.add_subplot(1, 1, 1)
#     for i in range(5):
#         # mean, std = filter_data(data[i], idx[i])
#         # field_max = mean + std
#         # field_min = mean - std
#         mean, field_min, field_max = filter_data(data[i], idx[i])
#         # mean = mean[:25]
#         # field_max = field_max[:25]
#         #field_min = field_min[:25]
#         xvalues = range(1, len(mean) + 1)

#         # Plotting mean and standard deviation
#         ax1.plot(
#             xvalues, mean, color=color[i], label=label[i],
#             linewidth=1.0)
#         ax1.fill_between(
#             xvalues, field_max, field_min,
#             color=colorshade[i], alpha=0.3)

#     plt.title('Goal Success Probability\n with various action space')
#     ax1.legend()
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Probability')

#     # ax1.set_yticks(
#     #     np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
#     plt.tight_layout()

#     # fig.savefig(
#     #     '/tmp/goal/data/experiments/' + pname + '.pdf')
#     fig.savefig(
#         '/tmp/goal/data/experiments/' + pname + '.png')
#     plt.close(fig)

def draw_distance_dist(data, fname):
    # Find infected nodes in the peak
    # expname = fname + str(complex_id) + '_'
    # After the experiments are done, draw plots
    # directory = os.path.join('/tmp', 'covid', 'data', 'experiments')

    # Find total node infected so far
    # data = load_files_peak(directory, expname)
    # print(data.shape)
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'bisque',
        1: 'darkorange',
        2: 'orangered',
        3: 'seagreen'
    }

    labels = ['Agent-Key', 'Key-Door', 'Door-Goal', 'Total']
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper left")
    ax1.set_xticklabels(labels)
    # ax1.set_xlabel('Object ')
    ax1.set_ylabel('Block Distance')
    ax1.set_title('Block Distance for Agent-Key-Door in KeyDoor Environment')

    plt.tight_layout()

    fig.savefig(
        '/tmp' + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def sucess_comparasion(tl=[30, 40, 50, 60], a=6):
    datas = []
    for i in tl:
        name = '_' + str(a) + '_' + str(i)
        data = load_files_all(
            '/tmp/goal/data/experiments', 'keydoor'+name+'_*')
        datas.append(data)
    draw_success_comp(datas, 'success_' + str(a) + '__', tl)


def trace_comparasion(tl=[30, 40, 50, 60], a=6):
    datas = []
    for i in tl:
        name = '_' + str(a) + '_' + str(i)
        # print(name)
        data = load_files_all(
            '/tmp/goal/data/experiments', 'keydoor'+name+'_*')
        datas.append(data)
    draw_trace_comp(datas, 'trace_' + str(a) + '__', tl)


def time_comparasion(tl=[60, 70, 80, 90, 100]):
    datas = []
    for i in tl:
        name = '_6_' + str(i)
        data = load_files_all(
            '/tmp/goal/data/experiments', 'keydoor'+name+'_*')
        datas.append(data)
    draw_time_comp(datas, 'keydoor_6_ti', tl)


# def action_comparasion():
#     datas = []
#     for i in [2, 4, 6, 8, 10]:
#         name = '_' + str(i) + '_50'
#         # print(name)
#         data = load_files_all(
#             '/tmp/goal/data/experiments', 'keydoor'+name+'_*')
#         datas.append(data)
#     draw_action_comp(datas, 'keydoor_2_a')


def distance_dist():
    data8x8 = np.load('/tmp/keydoor-8x8-dist.npy')
    data8x8 = data8x8.reshape(512, 4)
    # print(data8x8, data8x8.shape)
    draw_distance_dist(data8x8, 'keydoor8x8')

    data16x16 = np.load('/tmp/keydoor-16x16-dist.npy')
    data16x16 = data16x16.reshape(512, 4)
    # print(data8x8, data8x8.shape)
    draw_distance_dist(data16x16, 'keydoor16x16')


def results():
    name = '_2_70'
    datas = load_files_all(
        '/tmp/goal/data/experiments', 'keydoor'+name+'_*')
    draw_trace_data(datas, 'traces'+name)
    draw_success_prob(datas, 'sucess'+name)


def main():
    time_comparasion([80, 90, 100, 120, 150])
    trace_comparasion([80, 90, 100, 120, 150])
    sucess_comparasion([80, 90, 100, 120, 150])
    # distance_dist()


# def find_min_path():
#     factory = KeyDoorEnvironmentFactory()
#     environments = [factory.new() for _ in range(4)]
#     environments[0].reset()
#     # print(environments[0].step(0))


if __name__ == '__main__':
    main()
    # find_min_path()
