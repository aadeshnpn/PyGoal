import numpy as np
import matplotlib.pyplot as plt
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
    print(a4, b4)
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
    popt, pcov = curve_fit(logistfunc, x, y)
    # print(popt, pcov)
    plt.plot(x, y, 'b-', label='data')
    plt.plot(
        x, logistfunc(x, *popt), 'g--',
        label='3P Logistic: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    popt, pcov = curve_fit(logistfunc1, x, y)
    plt.plot(
        x, logistfunc1(x, *popt), 'r--',
        label='2P Logistic: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # compare()
    # linear_combine()
    # all_combine()
    get_para()


if __name__ == '__main__':
    main()
