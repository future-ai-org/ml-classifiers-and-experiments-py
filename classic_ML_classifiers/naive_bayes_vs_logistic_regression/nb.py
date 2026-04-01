#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix
np.set_printoptions(threshold='nan')


def naive_bayes(features, labels):
    ''' compute the naive bayes learner '''
    # count number each label for prior
    neginds = (labels == 0.0).nonzero()[0]
    posinds = (labels == 1.0).nonzero()[0]

    a1 = len(neginds)/len(labels)
    a2 = len(posinds)/len(labels)
    label_counts = np.array([a1, a2]).flatten()
    label_prob = np.log(label_counts)
    
    # dividing the feature data into pos and neg 
    aux_neg = features[neginds, :]
    aux_pos = features[posinds, :]
    
    # summing over for each document (first field)
    # adding smoothing, log to prevent overflow
    param_neg = np.log((1 + aux_neg.sum(0)) / (1+aux_neg.sum(0)).sum())
    param_pos = np.log((1 + aux_pos.sum(0)) / (1+aux_pos.sum(0)).sum())
    pn = np.squeeze(np.asarray(param_neg))
    pp = np.squeeze(np.asarray(param_pos))
    params = [pn, pp]
    
    return label_prob, params


     
def classify_naive_bayes(params, label_probs, features):
    ''' compute the naive bayes classifier '''
    #create an array with zeros in the size of test (200)
    laux = np.zeros((np.size(features, axis=0), 1))
    labels = np.squeeze(np.asarray(laux)) 

    # find conditional proba. for each class
    # find most likely label for each instance
    for i in range(np.size(features, axis=0)):
    	cp1 = features[i,:]*params[0]+label_probs[0]
  	cp2 = features[i,:]*params[1]+label_probs[1]
        if cp1 > cp2:
        	j = 0
	else:
		j = 1
 
	labels[i]=j
    return labels
    
