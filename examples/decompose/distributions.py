import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


meu = [5, 9, 9, 6, 2]
scale = [2, 3, 4, 2, 1]


def logistpdf(x, loc, scale):
    return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)


def logistcdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))


def normalpdf(x, loc, std):
    return stats.norm.pdf(x, loc, std)


def normalcdf(x, loc, std):
    return stats.norm.cdf(x, loc, std)


def draw(j, labels, logist=logistpdf):
    plt.subplot(2, 2, j+1)
    for i in range(len(meu)):
        # s = np.random.logistic(meu[i], scale[i], 10000)
        # count, bins, ignored = plt.hist(s, bins=50)
        x = np.linspace(-5, 20, 1000)
        lgst_val = logist(x, meu[i], scale[i])
        # std = (scale[i] * np.pi) / np.sqrt(3)
        # plt.plot(bins, lgst_val * count.max() / lgst_val.max())
        plt.plot(x, lgst_val)
        # plt.plot(x, stats.norm.pdf(x, meu[i], std), label='N')
        # fig, ax = plt.subplots(1, 1)
    plt.title(labels[j])


fig = plt.figure()
drawf = [logistcdf, logistpdf, normalcdf, normalpdf]
labels = ['Logistic CDF', 'Logistic PDF', 'Normal CDF', 'Normal PDF']
for i in range(len(drawf)):
    draw(i, labels, drawf[i])

plt.show()