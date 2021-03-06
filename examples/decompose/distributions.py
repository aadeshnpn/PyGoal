import os
import numpy as np
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import scipy.stats as stats


def logistpdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)


def logistcdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    return 1/(1+np.exp((loc-x)/scale))


def logistfunc(x, L, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    return L/(1+np.exp((loc-x)/scale))


def logistfunc1(x, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    return 1/(1+np.exp((loc-x)/scale))


def normalpdf(x, loc, std):
    return stats.norm.pdf(x, loc, std)


def normalcdf(x, loc, std):
    return stats.norm.cdf(x, loc, std)


def draw(j, meu, scale, xrang, labels, logist=logistpdf):
    plt.subplot(2, 2, j+1)
    for i in range(len(meu)):
        # s = np.random.logistic(meu[i], scale[i], 10000)
        # count, bins, ignored = plt.hist(s, bins=50)
        x = np.linspace(xrang[0], xrang[1], 1000)
        lgst_val = logist(x, meu[i], scale[i])
        # std = (scale[i] * np.pi) / np.sqrt(3)
        # plt.plot(bins, lgst_val * count.max() / lgst_val.max())
        lab = '$\mu$' + '=' + str(meu[i]) + ',' + '$\sigma=$' + str(scale[i])
        plt.plot(x, lgst_val, label=lab)
        # plt.plot(x, stats.norm.pdf(x, meu[i], std), label='N')
        # fig, ax = plt.subplots(1, 1)
    plt.title(labels[j])
    plt.legend()


def compare():
    fig = plt.figure()      # noqa: F841
    drawf = [logistcdf, logistpdf, normalcdf, normalpdf]
    labels = ['Logistic CDF', 'Logistic PDF', 'Normal CDF', 'Normal PDF']
    meu = [5, 9, 9, 6, 2]
    scale = [2, 3, 4, 2, 1]

    for i in range(len(drawf)):
        draw(i, meu, scale, (-5, 20), labels, drawf[i])

    plt.tight_layout()
    plt.show()


def linear_combine():
    fig = plt.figure()      # noqa: F841
    a, b = (1, 2)
    meu = [0, a*0 + b]
    scale = [2, np.abs(a) * 2]
    drawf = [normalpdf, logistpdf]
    labels = ['Normal', 'Logistic']
    for i in range(len(drawf)):
        draw(i, meu, scale, (-10, 10), labels, drawf[i])

    plt.tight_layout()
    # plt.title('Linear Combination (Y = aX + b)')
    plt.show()


def all_combine():
    fig = plt.figure()      # noqa: F841
    a1, b1, a2, b2 = (0, 2, 0, 3)
    a3, b3 = [a1+a2, np.sqrt(b1**2 + b2**2)]
    a4 = (a1 * (b2**2) + a2 * (b1**2)) / (b2**2 + b1**2)
    b4 = b2**2 * b1**2 / (b2**2 + b1**2)
    # print(a4, b4)
    meu = [a1, a2, a3, a4]
    scale = [b1, b2, b3, b4]
    drawf = [normalpdf, logistpdf, normalcdf, logistcdf]
    labels = ['Normal PDF', 'Logistic PDF', 'Normal CDF', 'Logistic CDF']
    for i in range(len(drawf)):
        draw(i, meu, scale, (-10, 10), labels, drawf[i])

    plt.tight_layout()
    # plt.title('Linear Combination (Y = aX + b)')
    plt.show()


def get_para():
    x = list(range(100))
    y = np.array(range(100), dtype=np.float)
    y[:30] = np.zeros((30,))
    y[30:40] = [0.35, 0.4, 0.6, 0.65, 0.7, 0.75, 0.75, 0.76, 0.78, 0.79]
    for i in range(40, 100, 5):
        y[i:i+5] = np.ones(
            (5)) * np.random.choice([0.8, 0.83, 0.85])
    # print(y.shape, y)
    from scipy.optimize import curve_fit
    # plt.subplot(1, 2, 1)
    popt, pcov = curve_fit(logistfunc, x, y)
    scale = popt[2]
    std = (scale * np.pi) / np.sqrt(3)
    plt.plot(x, y, 'b-', label='data')
    plt.plot(
        x, logistfunc(x, *popt), 'g--',
        label='3P Logistic: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # plt.plot(
    #     x, normalcdf(x, popt[1], std), 'y--',
    #     label='3P Normal: $\mu$=%5.3f, $\delta$=%5.3f'
    #       % tuple([popt[1], std]))
    plt.plot(
        x, logistpdf(x, popt[1], std), 'g--',
        label='3P PDF: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt[1], std]))

    popt, pcov = curve_fit(logistfunc1, x, y)
    scale = popt[1]
    std = (scale * np.pi) / np.sqrt(3)
    plt.plot(
        x, logistfunc1(x, *popt), 'r--',
        label='2P Logistic: a=%5.3f, b=%5.3f' % tuple(popt))
    # plt.plot(
    #     x, normalcdf(x, popt[0], std), 'c--',
    #     label='2P Normal: $\mu$=%5.3f, $\delta$=%5.3f'
    # % tuple([popt[0], std]))
    plt.plot(
        x, logistpdf(x, popt[0], std), 'r--',
        label='2P PDF: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt[1], std]))
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    # plt.subplot(1, 2, 2)

    plt.show()


def do_comp():
    x1 = list(range(100))
    y1 = np.array(range(100), dtype=np.float)
    y1[:30] = np.zeros((30,))
    y1[30:40] = [0.35, 0.4, 0.6, 0.65, 0.7, 0.75, 0.75, 0.76, 0.78, 0.79]
    for i in range(40, 100, 5):
        y1[i:i+5] = np.ones(
            (5)) * np.random.choice([0.8, 0.83, 0.85])

    x2 = list(range(100))
    y2 = np.array(range(100), dtype=np.float)
    y2[:40] = np.zeros((40,))
    y2[40:50] = [0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.68, 0.71, 0.75, 0.85]
    for i in range(50, 100, 5):
        y2[i:i+5] = np.ones(
            (5)) * np.random.choice([0.9, 0.92, 0.95])

    from scipy.optimize import curve_fit
    # popt1, pcov1 = curve_fit(logistfunc, x1, y1)
    # scale1 = popt1[2]
    # std1 = (scale1 * np.pi) / np.sqrt(3)
    # plt.subplot(1, 2, 1)
    # plt.plot(x1, y1, 'b-', label='data1')
    # plt.plot(
    #     x1, logistfunc(x1, *popt1), 'b-.',
    #     label='3P Logistic: L=%5.3f, $\mu$=%5.3f, s=%5.3f' % tuple(popt1))

    # popt2, pcov2 = curve_fit(logistfunc, x2, y2)
    # scale2 = popt2[2]
    # std2 = (scale2 * np.pi) / np.sqrt(3)
    # plt.plot(x2, y2, 'g-', label='data2')
    # plt.plot(
    #     x2, logistfunc(x2, *popt2), 'g-.',
    #     label='3P Logistic: L=%5.3f, $\mu$=%5.3f, s=%5.3f' % tuple(popt2))
    # plt.plot(x2, [0.8]*100, 'r.', label="$\\theta$", linewidth=1, alpha=0.3)
    # mpopt = (popt1 + popt2) / 2
    # plt.plot(
    #     x2, logistfunc(x2, *mpopt), 'c--',
    #     label='Merge: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(mpopt))
    # x = x1 + x2
    # print(y1.shape, y2.shape)
    # y = np.hstack((y1, y2))
    # popt3, pcov3 = curve_fit(logistfunc, x, y)
    # # print(x)
    # plt.plot(
    #     x1, logistfunc(x1, *popt3), 'c-.',
    #     label='All data: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3))
    # # plt.plot(
    # #     x, logistpdf(x, popt[0], std), 'r--',
    # #     label='2P PDF: $\mu$=%5.3f, $\delta$=%5.3f'
    # % tuple([popt[1], std]))
    # plt.ylabel('Probability')
    # plt.xlabel('Time')
    # plt.legend()
    # plt.tight_layout()
    # # plt.subplot(1, 2, 2)
    # plt.subplot(1, 2, 2)
    popt1, pcov1 = curve_fit(logistfunc, x1, y1)
    scale1 = popt1[2]
    std1 = (scale1 * np.pi) / np.sqrt(3)
    popt2, pcov2 = curve_fit(logistfunc, x2, y2)
    scale2 = popt2[2]
    std2 = (scale2 * np.pi) / np.sqrt(3)
    plt.subplot(1, 2, 1)
    plt.plot(x2, [0.8]*100, 'r.', label="$\\theta$", linewidth=1, alpha=0.3)
    plt.plot(
        x1, normalcdf(x1, popt1[1], std1), 'b--',
        label='Normal: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt1[1], std1]),
        linewidth=4, alpha=0.5)

    plt.plot(
        x1, logistcdf(x1, popt1[1], std1), 'b-.',
        label='Logistic: $\mu$=%5.3f, s=%5.3f' % tuple(popt1[1:]), alpha=0.5)
    plt.plot(
        x2, normalcdf(x2, popt2[1], std2), 'g--',
        label='Normal: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt2[1], std2]),
        linewidth=4, alpha=0.5)
    plt.plot(
        x1, logistcdf(x2, popt2[1], std2), 'g-.',
        label='Logistic: $\mu$=%5.3f, s=%5.3f' % tuple(popt2[1:]), alpha=0.5)
    plt.title('CDF')
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.subplot(1, 2, 2)

    plt.plot(
        x1, normalpdf(x1, popt1[1], std1), 'b--',
        label='Normal: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt1[1], std1]),
        linewidth=4, alpha=0.5)

    plt.plot(
        x1, logistpdf(x1, popt1[1], std1), 'b-.',
        label='Logistic: $\mu$=%5.3f, s=%5.3f' % tuple(popt1[1:]), alpha=0.5)
    plt.plot(
        x2, normalpdf(x2, popt2[1], std2), 'g--',
        label='Normal: $\mu$=%5.3f, $\delta$=%5.3f' % tuple([popt2[1], std2]),
        linewidth=4, alpha=0.5)
    plt.plot(
        x1, logistpdf(x2, popt2[1], std2), 'g-.',
        label='Logistic: $\mu$=%5.3f, s=%5.3f' % tuple(popt2[1:]), alpha=0.5)
    # plt.plot(
    #     x1, normalpdf(x1, popt1[0], std1), 'b-.',
    #     label='Data1 : $\mu$=%5.3f, s=%5.3f' % tuple(popt1))
    # plt.plot(
    #     x2, normalpdf(x2, popt2[0], std2), 'g-.',
    #     label='Data2 : $\mu$=%5.3f, s=%5.3f' % tuple(popt2))
    # popt, pcov = curve_fit(logistfunc1, x, y)
    # scale = popt[1]
    # std = (scale * np.pi) / np.sqrt(3)
    # plt.plot(
    #     x1, normalcdf(x1, popt[0], std), 'c--',
    #     label='CDF DataAll')
    # plt.plot(
    #     x1, normalpdf(x1, popt[0], std), 'c-.',
    #     label='PDF DataAll')
    # mean, std = [(popt1[0]+popt2[0])/2, np.sqrt(0.5*std1**2 + 0.5*std2**2)]
    # print(mean, popt1[0], popt2[0], std1, std2)
    # plt.plot(
    #     x1, normalcdf(x1, mean, std), 'g--',
    #     label='CDF Mixed')
    # plt.plot(
    #     x1, normalpdf(x1, mean, std), 'g-.',
    #     label='PDF Mixed')
    plt.tight_layout()
    plt.title('PDF')
    # plt.ylabel('Probability')
    # plt.xlabel('Time')
    plt.legend()
    plt.show()


def fig3():
    x2 = list(range(100))
    y2 = np.array(range(100), dtype=np.float)
    y2[:40] = np.zeros((40,))
    y2[40:50] = [0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.68, 0.71, 0.75, 0.85]
    for i in range(50, 100, 5):
        y2[i:i+5] = np.ones(
            (5)) * np.random.choice([0.9, 0.92, 0.95])

    from scipy.optimize import curve_fit
    popt2, pcov2 = curve_fit(logistfunc, x2, y2)
    # scale2 = popt2[2]
    # std2 = (scale2 * np.pi) / np.sqrt(3)
    # plt.subplot(1, 2, 1)
    popt3 = [popt2[0], popt2[1]-popt2[2], popt2[2]]
    popt4 = [popt2[0], popt2[1]+popt2[2], popt2[2]]
    plt.plot(x2, [0.8]*100, 'r--', label="$\\theta$", linewidth=2, alpha=0.3)
    plt.plot(x2, y2, 'g-', label='data2')
    plt.plot(
        x2, logistfunc(x2, *popt2), 'g-.',
        label='3P Logistic: L=%5.3f, $\mu$=%5.3f, s=%5.3f' % tuple(popt2))
    plt.fill_between(
        x2, logistfunc(x2, *popt3), logistfunc(x2, *popt4),
        color='green', alpha=0.2)
    # plt.plot(
    #     x2, normalpdf(x2, popt2[1], std2), 'g--',
    #     label='Normal: $\mu$=%5.3f, $\delta$=%5.3f'
    # % tuple([popt2[1], std2]),
    #     linewidth=2, alpha=0.5)
    # plt.fill_between(
    #     xvalues, field_max, field_min, color='DodgerBlue', alpha=0.3)
    # y3 = normalpdf(x3, popt2[0], 1-popt2[0]) * 15
    # print(x3, y3)
    # plt.plot(
    #     y3, x3, 'c--',
    #     label='Confidence: $L_2$=%5.3f, $1-L_2$=%5.3f'
    # % tuple([popt2[0], 1-popt2[0]]),
    #     linewidth=2, alpha=0.5)
    x3 = range(int(popt2[1]), 100)
    plt.plot(
        x3, [popt2[0]]*len(x3), 'm-.',
        label="$L_2$=%5.3f, $1-L_2$=%5.3f" % tuple([popt2[0], 1-popt2[0]]),
        linewidth=2, alpha=0.3)
    plt.fill_between(
        x3, [popt2[0]-(1-popt2[0])] * len(x3),
        [popt2[0]+(1-popt2[0])] * len(x3), color='orchid', alpha=0.3)
    plt.tight_layout()
    plt.title('Competency and Confidence')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def fig4():
    x3 = np.linspace(0.0, 1.0, 100)
    y3 = normalpdf(x3, 0.9, 0.1) * 100
    print(x3, y3)
    plt.plot(
        y3, x3, 'c--',
        label='Normal: $L_2$=%5.3f, $1-L_2$=%5.3f' % tuple([0.9, 0.1]),
        linewidth=5, alpha=0.5)
    plt.tight_layout()
    # plt.title('PDF')
    # plt.ylabel('Probability')
    # plt.xlabel('Time')
    plt.legend()
    plt.show()


def fig5():
    x1 = list(range(80))
    y1 = np.array(range(80), dtype=np.float)
    y1[:20] = np.zeros((20,))
    y1[20:30] = [0.35, 0.4, 0.6, 0.65, 0.7, 0.75, 0.75, 0.76, 0.78, 0.79]
    for i in range(30, 80, 5):
        y1[i:i+5] = np.ones(
            (5)) * np.random.choice([0.8, 0.83, 0.85])

    x2 = list(range(80))
    y2 = np.array(range(80), dtype=np.float)
    y2[:30] = np.zeros((30,))
    y2[30:40] = [0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.68, 0.71, 0.75, 0.85]
    for i in range(40, 80, 5):
        y2[i:i+5] = np.ones(
            (5)) * np.random.choice([0.9, 0.92, 0.95])

    x3 = list(range(81))
    y3 = np.array(range(81), dtype=np.float)
    y3[:40] = np.zeros((40,))
    y3[40:60] = [
        0.2, 0.24, 0.29, 0.34, 0.4, 0.45, 0.49, 0.53, 0.58, 0.65,
        0.69, 0.73, 0.75, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83]
    for i in range(60, 81, 3):
        y3[i:i+3] = np.ones(
            (3)) * np.random.choice([0.84, 0.87, 0.89])
    y3[80] = 0.89
    from scipy.optimize import curve_fit
    # Combine data
    # xall = x1 + x2 + x3
    # yall = np.concatenate((y1, y2, y3))

    x = [x1, x2, x3]
    y = [y1, y2, y3]
    style = ['g--', 'b--', 'c--']
    label = ['data1 fit', 'data2 fit', 'data3 fit']
    dist = []
    for i in range(len(x)):
        popt, pcov = curve_fit(logistfunc, x[i], y[i])
        # print(popt)
        dist.append(popt)
        # plt.plot(x[i], y[i], style[i], label=label[i])
        plt.plot(
            x[i], logistfunc(x[i], *popt), style[i],
            label=label[i]+' : L=%5.3f, $\mu$=%5.3f, s=%5.3f' % tuple(popt))
        # plt.plot(
        #     x[i], logistfunc1(x[i], *popt[1:]), style[i]+'-',
        #     label=label[i]+' : L=%5.3f, $\mu$=%5.3f, s=%5.3f'
        #     % tuple([1, popt[1], popt[2]]))

    # Final competency curve using sequence
    args = [sequence, selector, parallel]
    labels = ['Sequence', 'Selector', 'Parallel']
    color = ['orange', 'tomato', 'sienna']
    for i in range(3):
        paras = args[i](dist)
        # print(paras)
        fx = list(range(80))
        plt.plot(
            fx, logistfunc(fx, *paras), '-.', color=color[i],
            label=labels[i] + ': L=%5.3f, $\mu$=%5.3f, s=%5.3f'
            % tuple(paras))
        # plt.plot(
        #     fx, logistfunc1(fx, *paras[1:]),'--', color=color[i],
        #     label=labels[i]+': L=%5.3f, $\mu$=%5.3f, s=%5.3f'
        #     % tuple([1, paras[1], paras[2]]))

    # popt, pcov = curve_fit(logistfunc, xall, yall)
    # plt.plot(
    #     fx, logistfunc(fx, *popt), 'y-.',
    #     label='Real: L=%5.3f, $\mu$=%5.3f, s=%5.3f' % tuple(popt))
    # plt.plot(
    #     fx, logistfunc1(fx, *popt[1:]), 'y--',
    #     label='Real: L=%5.3f, $\mu$=%5.3f, s=%5.3f'
    #     % tuple([1, popt[1], popt[2]]))

    plt.tight_layout()
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.show()


def stdf(scale):
    return (scale * np.pi) / np.sqrt(3)


def scalef(std):
    return (np.sqrt(3) / np.pi) * std


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


def main():
    # compare()
    # linear_combine()
    # all_combine()
    # get_para()
    # do_comp()
    # fig4()
    # fig3()
    fig5()


if __name__ == '__main__':
    main()
