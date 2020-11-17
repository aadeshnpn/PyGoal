import os
import numpy as np
import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402


def exp_norm(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def logistpdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
    # return exp_norm(
    #     (loc-x)/scale)/(scale*(1+exp_norm((loc-x)/scale))**2)


def logistcdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    return 1/(1+np.exp((loc-x)/scale))
    # return 1/(1+exp_norm((loc-x)/scale))


def logistfunc(x, L, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    return L/(1+np.exp((loc-x)/scale))
    # return L/(1+exp_norm((loc-x)/scale))


def logistfunc1(x, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    return 1/(1+np.exp((loc-x)/scale))
    # return 1/(1+exp_norm((loc-x)/scale))


def normalpdf(x, loc, std):
    return stats.norm.pdf(x, loc, std)


def normalcdf(x, loc, std):
    return stats.norm.cdf(x, loc, std)


def compare_curve(
        competency, datas, visdata=True, vispdf=False,
        viscdf=True, name='competency'):
    fig = plt.figure()
    colors = [
        'blue', 'orange', 'green', 'red', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan']
    linepat = ['^', '-.']
    labels = ['data'+str(i) for i in range(len(competency))]
    plt.plot(
        range(len(datas[0])), [0.8]*len(datas[0]), '*', label="$\\theta$",
        color='tomato', linewidth=1, alpha=0.9)
    for i in range(len(competency)):
        popt = competency[i]
        data = datas[i]
        x = range(len(datas[i]))
        plt.plot(
            x, data, linepat[0], label=labels[i],
            color=colors[i], linewidth=1, alpha=0.5)
        plt.plot(
            x, logistfunc(x, *popt), linepat[1], color=colors[i],
            label='Logistic : L=%5.2f, $\mu$=%5.2f, s=%5.2f'
            % tuple(popt), linewidth=2, alpha=0.8)

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Time')
    # plt.title('PDF')
    plt.tight_layout()
    # Create folder if not exists
    import pathlib
    dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(dname, name+'.png')
    fig.savefig(fname)  # pylint: disable = E1101
    plt.close(fig)
    # plt.show()