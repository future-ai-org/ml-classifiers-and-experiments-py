#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix


def train_logistic_regression(features, labels, l_rate, target_delta, reg_constant):
    ''' gradient ascent algorithm to train logistic regression params,
        where offset and weights are the parameters '''
    offset = 0
    w_delta = 100000

    # old_ws and ws are matrices 1x 10770 with zeros
    old_ws = np.zeros((1, np.size(features, axis=1)-1))
    ws =  np.zeros((1, np.size(features, axis=1)))
    
    # regularizer is a matrix 1x10770 multiplied by reg_constant
    regularizer = np.zeros((1, np.size(features, axis=1)))*reg_constant
  
    ''' calculate the probability that each instance is classified as 1 '''
    ''' first calculate the weight sums '''
    # create a 200x1 array with the value of offset
    # same as repmat(offset, size(features,1),1) in MATLAB
    w_1 = np.ones((np.size(features, axis=0),1))* offset

    # create a 200x10770 aarray with the value of ws
    # same as repmat(ws, size(features,1),1) in MATLAB    
    w_2 =   np.ones((np.size(features, axis=0), 1)) * ws

    # multiply these by features (which is 200x10770)
    w_3 =  (np.multiply(w_2, features.todense())).sum(1)
    
    # finally calculates the weight sums 200x1
    w_sums = w_1 + w_3
    
    ''' now comput the probabilities '''
    # creating logistic
    den = np.exp(w_sums) + 1
    num = np.exp(w_sums)
    probs = num/den
    
    ''' calculating current ll '''
    # expand 1 dim in labels to 200x1
    l_2d = np.expand_dims(labels,1)
    c_aux = np.multiply(w_sums[:np.size(w_sums, axis=0)-1], l_2d)
    c_aux2 = np.log(1 + np.exp(w_sums[:np.size(w_sums, axis=0)-1]))
    c_aux3 = (c_aux - c_aux2).sum()
    c_aux4 = np.multiply(np.multiply(ws,ws), (regularizer//2))
    c_aux5 = c_aux4.sum()
    current_ll = c_aux3 - c_aux5

    print("Training logistic regression. Initial: ", current_ll)
	
    '''starting iterations '''
    iter_n = 0
    probss = probs[:np.size(probs)-1]
    featuress =  features [:np.size(features, axis=0)-1,:np.size(features, axis=1)-1]
    wss= ws[:, :np.size(ws, axis=1)-1]
    regularizers = regularizer[:,:np.size(regularizer, axis=1)-1]

    while (w_delta > target_delta):
	old_ws[:] = wss[:]

        # calculating the gradient 
        grad_aux0 = (l_2d  - probss)
        grad_aux00 = np.ones((1, np.size(features, axis=1)-1))
        grad_aux = grad_aux0*grad_aux00
        grad_aux1 = np.multiply(grad_aux, featuress.todense())
        grad_aux2 = csc_matrix(grad_aux1.sum(0))
        grad_aux3 = np.multiply(regularizers, wss)
        grad_aux4 = grad_aux3[:, :np.size(grad_aux3, axis=1)-1]
        grad = grad_aux2 - grad_aux3	

	# magnitude limit gradient
        grad = l_rate*grad
        iter_n += 1

        # update ws with previous labe prob
        offset = offset +  l_rate*grad_aux0.sum()
        wss = wss + grad

	# using the current weights, calculate the proba for instance 
    	w_1 = np.ones((np.size(featuress, axis=0),1))* offset  
   	w_2 =   np.ones((np.size(featuress, axis=0), 1)) * wss
    	w_3 =  (np.multiply(w_2, featuress.todense())).sum(1)
   	w_sums = w_1 + w_3

        # iterating logist
	den = np.exp(w_sums) + 1
    	num = np.exp(w_sums)
    	probss = num/den

	# update likelihood
    	l_2d = np.expand_dims(labels,1)
    	c_aux = np.multiply(w_sums, l_2d)
    	c_aux2 = np.log(1 + np.exp(w_sums))
    	c_aux3 = (c_aux - c_aux2).sum()
    	c_aux4 = np.multiply(np.multiply(wss,wss), (regularizers//2))
    	c_aux5 = c_aux4.sum()
    	current_ll = c_aux3 - c_aux5
	
	# update weight delta
	w_delta = np.sqrt((  np.multiply((old_ws - wss),(old_ws - wss)) ).sum() )

	# print?
  	if (np.mod(iter_n, 100) == 0):
		print('Log-likelihood, weight delta: ', current_ll, w_delta)

    print('Final ll:', current_ll)
    return  offset, wss 



def run_logistic_regression(offset, wss, features):
	featuress =  features [:np.size(features, axis=0)-1,:np.size(features, axis=1)-1]
    	# using the current weights, calculate the proba for instance 
    	w_1 = np.ones((np.size(featuress, axis=0),1))* offset  
   	w_2 =   np.ones((np.size(featuress, axis=0), 1)) * wss
    	w_3 =  (np.multiply(w_2, featuress.todense())).sum(1)
   	w_sums = w_1 + w_3

    	posinds = (w_sums > 0).nonzero()[0]

	laux = np.zeros((np.size(features, axis=0), 1))
    	labels = np.squeeze(np.asarray(laux)) 

	labels[posinds] = 1
    	return labels 



















