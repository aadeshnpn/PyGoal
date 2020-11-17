import numpy as np
import scipy.stats as stats


def exp_norm(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def logistpdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    # return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
    return exp_norm(
        (loc-x)/scale)/(scale*(1+exp_norm((loc-x)/scale))**2)


def logistcdf(x, loc, scale):
    scale = (np.sqrt(3) / np.pi) * scale
    # return 1/(1+np.exp((loc-x)/scale))
    return 1/(1+exp_norm((loc-x)/scale))


def logistfunc(x, L, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    # return L/(1+np.exp((loc-x)/scale))
    return L/(1+exp_norm((loc-x)/scale))


def logistfunc1(x, loc, scale):
    # scale = (np.sqrt(3) / np.pi) * scale
    # return 1/(1+np.exp((loc-x)/scale))
    return 1/(1+exp_norm((loc-x)/scale))


def normalpdf(x, loc, std):
    return stats.norm.pdf(x, loc, std)


def normalcdf(x, loc, std):
    return stats.norm.cdf(x, loc, std)
