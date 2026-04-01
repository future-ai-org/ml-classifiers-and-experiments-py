'''
Calculates the k-nearest neighbor (kNN) algorithm

'''

import math
import numpy as np
import scipy.io


from calculate_cosine_distance import cosineDistance


__author__ = """Marina von Steinkirch"""



###########################################


def loading_data(filename):
    ''' 
       This function load a MATLAB file and get the dict variables 
    '''
    f = scipy.io.loadmat(filename)
    traindata = f['traindata']
    trainlabels = f['trainlabels']
    testdata = f['testdata']
    evaldata = f['evaldata']
    testlabels = f['testlabels']
    return traindata, trainlabels, testdata, evaldata, testlabels

###########################################


def calculate_distances(trainlabels, traindata):
    '''
       This function calculate the distances for all the input examples
    '''
    distances = []
    for i in range(len(trainlabels)):
        first_train_example_class1 = traindata[i]
        aux = []
        for j in range (len(trainlabels)):
             if i != j:
                 first_train_example_class2 = traindata[j]
                 d = cosineDistance(first_train_example_class1, first_train_example_class2)
                 aux.append(d)
        distances.append(aux) 
    return distances


###########################################

def get_closest_k_points(D, k):
    '''
       Get the k closest points
    '''
    return sorted(D)[:k+1] 


###########################################

def get_index_vec(points_list, D):
    '''
       Return the index of the k closest points
    '''
    index_vec = []
    while points_list:
        point  = points_list.pop()
	for j, point_D in enumerate(D):
            if  np.isclose(point_D, point, rtol=1e-8, atol=1e-08, equal_nan=False) and j not in set(index_vec):	
		index_vec.append(j)
                break
        
    return index_vec


###########################################

def knn_search(x, D, k):
    ''' 
       Find the K nearest neighbours of data among distance D 
    '''
    points_list = get_closest_k_points(D, k)
    return get_index_vec(points_list, D)



############################################

def knn_classify(index_vec, trainlabels):
    '''
       Return the classification of the indices
    '''
    label1 = trainlabels[0]
    label2 = None
    neg, pos = 0,0

    for i in index_vec[1:]:
      if trainlabels[i] == label1: 
 	pos += 1
      else: 
	neg += 1
	label2 = trainlabels[i]
    if pos >= neg: return label1
    else: return label2



#########################################

def calculate_knn(traindata, trainlabels, k):
    '''
       Return knn classification
    '''
    distances = calculate_distances(trainlabels, traindata)

    
    # calculate knn for the entire data
    correct = 0.0
    total = len(traindata)
    for x in range(len(traindata)):
	    index_vec = knn_search(x, distances[x], k)
	    classification = knn_classify(index_vec, trainlabels)
            if trainlabels[x] == classification[0]:
		correct += 1
    return correct/total




###########################################



if __name__ == '__main__':
    import sys

    k = sys.argv[1] if len(sys.argv) == 2 else 5
    datafile = sys.argv[2] if len(sys.argv) == 3 else 'cvdataset.mat'

    traindata, trainlabels, testdata, evaldata, testlabels = loading_data(datafile)
    
    print(calculate_knn(traindata, trainlabels, k))
    

