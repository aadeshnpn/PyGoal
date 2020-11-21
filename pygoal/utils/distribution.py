import os
import numpy as np
import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
from py_trees.composites import Sequence, Selector, Parallel
plt.style.use("fivethirtyeight")


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


def stdf(scale):
    return (scale * np.pi) / np.sqrt(3)


def scalef(std):
    return (np.sqrt(3) / np.pi) * std


def plot_competency_variance(data, name):
    # mean = np.mean(data, axis=1)
    # Using quartile instead of mean and std
    # median = np.quantile(data, 0.5, axis=0)
    median = np.mean(data, axis=0)
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)
    fig = plt.figure()
    xvalues = range(1, len(median) + 1)
    # print(median.shape, median)
    # Plotting mean and standard deviation using curve fitting
    fig = plt.figure()
    from scipy.optimize import curve_fit
    try:
        poptmed, _ = curve_fit(
            logistfunc, xvalues, median,
            maxfev=800)
        poptq1, _ = curve_fit(
            logistfunc, xvalues, q1,
            maxfev=800)
        poptq3, _ = curve_fit(
            logistfunc, xvalues, q3,
            maxfev=800)
    except RuntimeError:
        poptmed = np.array([0.99, 1.0, 1.0])
        poptq1 = np.array([0.99, 1.0, 1.0])
        poptq3 = np.array([0.99, 1.0, 1.0])

    plt.plot(
        range(len(median)), [0.8]*len(median), '*', label="$\\theta$",
        color='tomato', linewidth=1, alpha=0.9)

    plt.plot(
        xvalues,
        logistfunc(xvalues, *poptmed),
        color="blue",
        label='Mean=%5.2f, $\mu$=%5.2f, s=%5.2f' % tuple(
            poptmed), linewidth=1.0, alpha=0.8,
        linestyle="dashdot",
    )

    plt.fill_between(
        xvalues,
        logistfunc(xvalues, *poptq1),
        logistfunc(xvalues, *poptq3),
        color="DodgerBlue", alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.tight_layout()

    import pathlib
    import os
    dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(dname, name + '.png')
    fig.savefig(fname)
    plt.close(fig)


def compare_curve(
        competency, datas, visdata=True, vispdf=False,
        viscdf=True, name='competency', root=False):
    fig = plt.figure()
    colors = [
        'blue', 'orange', 'red', 'purple',
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
            label='Logistic : L=%5.2f, $\mu$=%5.2f, s=%5.2f'    # noqa: W605
            % tuple(popt), linewidth=2, alpha=0.8)
    if root is not False:
        plt.plot(
            x, logistfunc(x, *root), ':', color='green',
            label='Root : L=%5.2f, $\mu$=%5.2f, s=%5.2f'    # noqa: W605
            % tuple(root), linewidth=3, alpha=0.7)

    plt.ylim(-0.1, 1.1)
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


def sequence(nodes):
    weights = np.array(list(range(len(nodes), 0, -1)), dtype=np.float)
    m = weights.shape[0]
    M = (m * (m+1)) / 2
    weights = weights / M
    # Let X ~ N(mu, std), then Y = kX
    # Y(mu) = k * mu, Y(std) = |k| * std
    means = np.array([node[1] for node in nodes]) * weights
    stds = np.array([stdf(node[2]) for node in nodes]) * weights
    # Let X ~ N(mu1, std1) and Y ~ N(mu2, std2) then
    # Z = X + Y, Z ~ N(mu1+mu2, std1^2 + std2^2)
    mean = np.sum(means)
    scale = scalef(np.sum(np.power(stds, 2)))
    confidence = np.array([node[0] for node in nodes]) * weights
    confidence = np.sum(confidence)
    return [confidence, mean, scale]


def selector(nodes):
    weights = np.ones((len(nodes))) / len(nodes)
    # Let X ~ N(mu, std), then Y = kX
    # Y(mu) = k * mu, Y(std) = |k| * std
    means = np.array([node[1] for node in nodes]) * weights
    stds = np.array([stdf(node[2]) for node in nodes]) * weights
    # Let X ~ N(mu1, std1) and Y ~ N(mu2, std2) then
    # Z = X + Y, Z ~ N(mu1+mu2, std1^2 + std2^2)
    mean = np.sum(means)
    scale = scalef(np.sum(np.power(stds, 2)))
    confidence = np.array([node[0] for node in nodes]) * weights
    confidence = np.sum(confidence)
    return [confidence, mean, scale]


def parallel(nodes):
    # Let X ~ N(mu1, std1) and Y ~ N(mu2, std2) then
    # Z = X * Y,
    # Z(meu) = mu1/std1**2 + mu2/std2**2 / (1/std1**2 + 1/std2**2)
    # z(std) = std1**2 * std2**2 / (std1**2 + std2**2)
    node = nodes[0]
    for i in range(1, len(nodes)):
        std1sqred = node[2]**2
        std2sqred = nodes[i][2]**2
        invstd1 = 1 / std1sqred
        invstd2 = 1 / std2sqred
        std = (std1sqred * std2sqred) / (std1sqred + std2sqred)
        meannum = (node[1] * invstd1) + (nodes[i][1] * invstd2)
        meandem = invstd1 + invstd2
        mean = meannum / meandem
        L = node[0] * nodes[i][0]
        node = [L, mean, std]
    return node


def recursive_setup(node, fn_e, fn_c):
    if node.children:
        for c in node.children:
            if (
                isinstance(c, Sequence) or isinstance(c, Selector) or
                    isinstance(c, Parallel)):
                # Control nodes
                fn_c(c)
            recursive_setup(c, fn_e, fn_c)
    else:
        # Execution nodes
        return fn_e(node)


def recursive_com(node, blackboard):
    # Control Nodes
    if isinstance(node, Sequence):
        val = sequence(
            [recursive_com(child, blackboard) for child in node.children])
        blackboard.shared_content[
            'curve'][node.name] = val
        return val
    elif isinstance(node, Selector):
        val = selector(
            [recursive_com(child, blackboard) for child in node.children])
        blackboard.shared_content[
            'curve'][node.name] = val
        return val
    elif isinstance(node, Parallel):
        val = parallel(
            [recursive_com(child, blackboard) for child in node.children])
        blackboard.shared_content[
            'curve'][node.name] = val
        return val
    else:
        # Execution nodes
        return node.planner.blackboard.shared_content['curve'][node.name]
