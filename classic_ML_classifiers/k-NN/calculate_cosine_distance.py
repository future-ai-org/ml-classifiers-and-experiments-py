'''
Calculates the cosine distance for an input data

'''

import math
import numpy as np
import scipy.io


__author__ = """Marina von Steinkirch"""


def cosineDistance(x, y):
    ''' This function computes the cosine distance between feature vectors 
         x and y. This distance is frequently used for text classification. 
        It varies between 0 and 1. The distance is 0 if x==y. 
    '''

    denom = math.sqrt(sum(x**2)*sum(y**2))
    dist = 1.0-(np.dot(x, y.conj().transpose()))/denom
    return round(dist, 6)

def print_to_file(distances):
   with open('cos_distances.dat', 'w') as f:
       for i, col in enumerate(distances):
           f.write('# distance for example %d to others\n' %(i+1))
           for item in col:
               f.write(str(item) + ' ')
           f.write('\n')

def main():
    f = scipy.io.loadmat('cvdataset.mat')
    traindata = f['traindata']
    trainlabels = f['trainlabels']
    testdata = f['testdata']
    evaldata = f['evaldata']
    testlabels = f['testlabels']

    distances = []

    for i in range(len(trainlabels)):
        first_train_example_class1 = traindata[i]
        aux = []
        for j in range (len(trainlabels)):
    	     first_train_example_class2 = traindata[j]
    	     d = cosineDistance(first_train_example_class1, first_train_example_class2)
    	     aux.append(d)
        distances.append(aux)
    
    print_to_file(distances)



if __name__ == '__main__':
    main()
