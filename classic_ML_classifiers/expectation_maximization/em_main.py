#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

''' This program implements the EM algorithm for a two-class
    Gaussian mixture model '''

import numpy as np


def load_data(datafile_name):
    data = np.mat(np.loadtxt(datafile_name))
    dataT = data.T
    return dataT


def perform_em(X, mu, theta, epsilon):
    stop = 1.0
    n = np.shape(X)[0]
    P = np.mat(np.zeros((n, len(mu))))

    while epsilon - stop < 0:
        aux_gauss = np.mat(np.zeros((n, len(mu))))
        P = e_step(aux_gauss, P, X, n, mu, theta)
        mu_n, theta_n = m_step(mu, theta, P, X, n)
        stop = sum(abs(mu_n-mu) + abs(theta_n - theta))/2.0
        mu, theta = mu_n, theta_n

    return mu, theta


def e_step(aux_gauss, P, X, n, mu, theta):
    for i in range(n):
        aux_gauss[i, :] = np.exp(-1.0 * np.square((X[i, :] - mu)) / 2.0)
    for j in range(len(mu)):
        P[:, j] = np.divide(aux_gauss[:, j]*theta[j], aux_gauss[:, 0]*theta[0] + aux_gauss[:, 1]*theta[1])
    return  P


def m_step(mu, theta, P, X, n):
    for i in range(len(mu)):
        theta[i] = sum(P[:, i])/n
        mu[i] = sum(np.multiply(X, P[:, i]))/sum(P[:, i])
    return mu, theta


def main():
    X = load_data('hw5.data')
    #print(X)
    epsilon = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9] # we want to test convergence 
    mu = np.array([1.0, 2.0])
    theta = np.array([0.33, 0.67])

    for e in epsilon:
        mu_n, theta_n = perform_em(X, mu, theta, e)
        print '\nFor epsilon = ', e
        print 'mu = ', mu_n
        print 'theta = ', theta_n


if __name__ == '__main__':
    main()
