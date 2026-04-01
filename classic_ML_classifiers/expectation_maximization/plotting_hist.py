#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

''' This script plots the historgam and the mixture density for em fit '''

import matplotlib.pyplot as plt
import numpy as np
import json


def gaussian(x, theta, mu):
    return theta * np.exp(-1.0 * np.square(x - mu)/ 2.0) / np.sqrt(2.0 * np.pi)


def main():
    X = np.loadtxt('hw5.data')

    mu = [0.03360538, 2.55387958]
    theta = [0.50216929, 0.49783071]

    xpoints = np.linspace(-5, 7, 100)

    y = np.vectorize(gaussian)
    y1 = y(xpoints, theta[0], mu[0])
    y2 = y(xpoints, theta[1], mu[1])
    y3 = y1 + y2



    with open("conf.json") as json_file:
        s = json.load(json_file)
 
    plt.rcParams.update(s)
 
    

    plt.plot(xpoints, y1, '*', label='Gaussian 1')
    plt.plot(xpoints, y2, '*', label='Gaussian 2')
    plt.plot(xpoints, y3, label='Gaussian Mixture', linewidth=2)
    plt.hist(X, bins=30, label="Histogram of hw5.data", normed=True)

    plt.annotate('Mean: 0.03', xy=(0.03,0.21), xytext=(-1, 0.24),
            arrowprops=dict(facecolor='black'),)

    plt.annotate('Mean: 2.55', xy=(2.5,0.21), xytext=(3, 0.23),
            arrowprops=dict(facecolor='black'),)


    plt.legend(loc=2, prop={'size':8})
    plt.title('Example of two-class Gaussian Mixture Model');
    plt.savefig('hist_em.png')

    

if __name__ == '__main__':
    main()
