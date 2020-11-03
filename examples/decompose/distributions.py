import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def logistpdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)


def logistcdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
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
    drawf = [normalpdf, logistpdf]
    labels = ['Normal', 'Logistic']
    for i in range(len(drawf)):
        draw(i, meu, scale, (-10, 10), labels, drawf[i])

    plt.tight_layout()
    # plt.title('Linear Combination (Y = aX + b)')
    plt.show()


def main():
    compare()
    # linear_combine()
    # all_combine()


if __name__ == '__main__':
    main()
