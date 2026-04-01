#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

''' TO RUN: 

$ python run_problem_final.py

(in the same folder as the data files)
'''


from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix
from lr import run_logistic_regression, train_logistic_regression
from nb import naive_bayes, classify_naive_bayes


def load_data(filename):
    ''' load the data and the label files for
        either training or test data '''   
    c1, c2, c3 = np.loadtxt(filename + ".data", unpack=True)
    # sames as training_data = spconvert(loaded_file); in MATLAB
    counts = csc_matrix((c3, (c1, c2)))
    labels = np.loadtxt(filename + ".label", unpack=True)
    return counts, labels



def print_results(string, result_train, train_labels, result_test, test_labels):
    ''' return the accuracy for test and training '''
    accuracy_train = len(result_train)/len(train_labels)
    print(string +  'Train accuracy: ' , accuracy_train)
    
    accuracy_test = len(result_test)/len(test_labels)
    print(string +  'Test accuracy: ' , accuracy_test)




def main():
    #load training and test data
    train, train_labels = load_data("train")
    test, test_labels = load_data("test")

    # run naive bayes
    nb_label_probs, nb_params = naive_bayes(train, train_labels)
    nb_train_labels = classify_naive_bayes(nb_params, nb_label_probs, train)
    nb_labels = classify_naive_bayes(nb_params, nb_label_probs, test)
 
    # print results
    result_test_nb = []
    for i in range(len(test_labels)):
	if nb_labels[i] == test_labels[i]:
		result_test_nb.append(i)
  
    result_train_nb = []
    for i in range(len(train_labels)):
	if nb_train_labels[i] == train_labels[i]:
		result_train_nb.append(i)
    print_results('Naive Bayes ',result_train_nb,train_labels,result_test_nb, test_labels)
    

    # run logistic regression
    l_rate = 0.0001
    target_delta = 0.001
    reg_constant = 0
    lr_offset, lr_w = train_logistic_regression(train, train_labels,l_rate, target_delta, reg_constant)   
    
    lr_train_labels = run_logistic_regression(lr_offset, lr_w, train)
    lr_labels = run_logistic_regression(lr_offset, lr_w, test)

    
    # print results
    result_test_lr = []
    for i in range(len(test_labels)):
	if lr_labels[i] == test_labels[i]:
		result_test_lr.append(i)
  
    result_train_lr = []
    for i in range(len(train_labels)):
	if lr_train_labels[i] == train_labels[i]:
		result_train_lr.append(i)
    print_results('Logistic Regression ',  result_train_lr,  train_labels, result_test_lr, test_labels)

   
    print("Done!")



if __name__ == '__main__':
    main()
