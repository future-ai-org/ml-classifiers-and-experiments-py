'''
Calculates the cosine distance for an input data

'''

import math
import numpy as np
import scipy.io
from kNN import loading_data


__author__ = """Marina von Steinkirch"""


def divide_data_folds(traindata, trainlabels, n):
    '''
       Divide the data in n-folds and return each fold/set of training data and
       training labels as an element in an array.
    '''
    trainingdata_folded = []
    traininglabel_folded = []

    len_data = len(traindata)
    num_elements_folds = len_data/n

    aux_rest_of_data = traindata[:]
    aux_rest_of_label = trainlabels[:]

    for i in range(n):
	trainingdata_folded.append(aux_rest_of_data[0:num_elements_folds])
        aux_rest_of_data = aux_rest_of_data[num_elements_folds:]

  	traininglabel_folded.append(aux_rest_of_label[0:num_elements_folds])
        aux_rest_of_label = aux_rest_of_label[num_elements_folds:] 


    return trainingdata_folded, traininglabel_folded


#######################################################

def return_data_nminusone(trainingdata_folded, traininglabel_folded, n_here):
    '''
       Join the data together minus one fold.
    '''
    data_here = trainingdata_folded[0:n_here] + trainingdata_folded[n_here+1:]
    label_here = traininglabel_folded[0:n_here] + traininglabel_folded[n_here+1:]
    
    data_here_together = [] 
    label_here_together = []
    
    for i in data_here:
        data_here_together.extend(i) 

    for i in label_here:
        label_here_together.extend(i) 

    return  data_here_together, label_here_together 



#######################################################

if __name__ == '__main__':
    import sys

    n = sys.argv[1] if len(sys.argv) == 2 else  10
    datafile = sys.argv[2] if len(sys.argv) == 3 else 'cvdataset.mat'



    trainingdata, traininglabels, testdata, evaldata, testlabels = loading_data(datafile)
    
    # arrays with subarrays with data in n folds
    trainingdata_folded, traininglabel_folded = divide_data_folds(trainingdata, traininglabels, n)

    # get an array similar to trainingdata but without one of the folders
    n_here = 0 # of range(n)
    ntrainingdata, ntraininglabel = return_data_nminusone(trainingdata_folded, traininglabel_folded, n_here)

    print(len(traininglabels))
    print(len(ntraininglabel))

    

    

 

    

