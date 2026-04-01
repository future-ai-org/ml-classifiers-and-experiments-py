'''
Calculates the k-nearest neighbor (kNN) algorithm

'''

import math
import numpy as np
import scipy.io


from kNN import calculate_knn, loading_data
from nkNN import divide_data_folds, return_data_nminusone


__author__ = """Marina von Steinkirch"""


if __name__ == '__main__':
    import sys

    num_k = sys.argv[1] if len(sys.argv) == 2 else 100
    num_n = sys.argv[2] if len(sys.argv) == 3 else 10
    datafile = sys.argv[3] if len(sys.argv) == 4 else 'cvdataset.mat'

    trainingdata, traininglabels, testdata, evaldata, testlabels = loading_data(datafile)

    trainingdata_folded, traininglabel_folded = divide_data_folds(trainingdata, traininglabels, num_n)

    data_n = []

    for n in range(num_n):

        data_k = []
        data_here, label_here = return_data_nminusone(trainingdata_folded, traininglabel_folded, n)  
        print('# n_here = %d' %n)
        for k in range(num_k):  
            a = calculate_knn(data_here, label_here, k)
            data_k.append(a)
            print('%d	%.3f' %(k, a))
        data_n.append(data_k)
 

    with open('output.dat', 'w') as f:
	f.write(data_n)



