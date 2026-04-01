#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

''' Script for contour plots in EM analysis'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import json


def log_lh(X, mu):
    s = 0
    for i in X:
        s += np.log(mixture(i, [0.5, 0.5], mu))
    return s


def main():
    X = np.loadtxt('hw5.data')

    mu_1 = np.arange(-1, 4, 0.25)
    mu_2 = np.arange(-1, 4, 0.25)

    axisz = [[log_lh(X, [x, y]) for x in mu_1] for y in mu_2]

    CS = plt.contour(mu_1, mu_2, axisz)
    plt.colorbar(CS, shrink=0.6, extend='both')

    plt.axis('tight')
    plt.title("Log-likelihood vs. Means for Mixture Model")
    plt.xlabel(r'$\mu_1$')
    plt.ylabel(r'$\mu2$')
    plt.savefig('cont_em.png')
    

if __name__ == '__main__':
    main()