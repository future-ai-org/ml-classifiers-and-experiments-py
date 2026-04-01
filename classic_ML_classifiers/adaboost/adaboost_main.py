#!/usr/bin/python
# marina von steinkirch @2014
# steinkirch at gmail

''' This program shows the application of boosting for weak
    classifiers given by decision stumps '''

import numpy as np
import random
from adaboost import AdaBoost
from decision_stump import DecisionStump



def save_result_split(final, name):
    ''' Save in a file the results of spliting'''

    with open('./data/' + name + ".txt", "w") as f:
        for i in range(len(final)):
            f.write(str(final[i]) + "\n")



def save_result_final(final, name):
    ''' Save in a file the final result to plot'''

    with open('./data/' + name + ".txt", "w") as f:
        for i in range(len(final)):
            f.write(str(final[i]) + "," + str(i+1) + "\n")



def split_data(percentage, num_sets):
    ''' split data for training and test sets '''

    with open('./data/' + "bupa.data", "rb") as f:
        data = f.read().split('\n')
 	border = int(percentage*len(data))
	
        for i in range(num_sets):        
	    random.shuffle(data)
            train_data = data[:border][:]
            test_data = data[border:][:]
            save_result_split(train_data, "bupa_train" + str(i))
            save_result_split(test_data, "bupa_test" + str(i))




def load_data(datafile_name):
     ''' Load the data and separate it by feature
         and labels '''
     data = np.loadtxt(datafile_name, delimiter = ',')

     # features
     ''' X will be an array with each of the 6 features
         in 6 columns, and each row a data entry '''
     X = data[:,:-1] 

     # label
     ''' The last column of the data is the "selector"  
         field used to split data into two sets, and we
         use it as a label, setting 2 -> -1 '''
     Y = data[:,-1]
     Y[Y==2] = -1   
 
     return X, Y




def calculate_error(T, score, Y):
    ''' Calculate error '''
    final = []
    for j in range(T):
        right, wrong = 0, 0
	dataset_for_this_T = score[j]	
	for i in range(len(dataset_for_this_T)):
	    if dataset_for_this_T[i] == Y[i]:
	       right += 1.0
            else:
               wrong += 1.0
	final.append(wrong/(right+wrong))
    
    return final



def main():
    ''' Load data, split data, creates adaboost algorithm 
        with decision stump, calculates errors, save final file'''
  
    classifier = AdaBoost(DecisionStump)

    num_sets = 50
    T = 100  
    percentage = 0.9 

    all_errors_train = []
    all_errors_test = []    
    aver_error_train = []
    aver_error_test = []


    # split data in the # of datasets
    split_data(percentage, num_sets)


    # run  for all datasets, for boosting interations = T 
    for i in range(num_sets):
        data_split_train = './data/bupa_train' + str(i) + ".txt"
        data_split_test = './data/' + "bupa_test" + str(i) + ".txt"
        X_train, Y_train = load_data(data_split_train)
        X_test, Y_test = load_data(data_split_test)

        score_train, score_test = classifier.run_adaboost(X_train, Y_train, T, X_test)

	    error_train = calculate_error(T, score_train, Y_train)
        error_test = calculate_error(T, score_test, Y_test)
	
	   all_errors_train.append(error_train)
	   all_errors_test.append(error_test)
   

    # calculates the average errors
    for j in range(T):
        a_e_train = 0
	    a_e_test = 0
	    for i in range(num_sets):
		   a_e_train += all_errors_train[i][j]
		   a_e_test += all_errors_test[i][j]
        
        aver_error_train.append(a_e_train/num_sets)
	    aver_error_test.append(a_e_test/num_sets)
  

    # save to file to plot
    save_result_final(aver_error_train, 'train')
    save_result_final(aver_error_test, 'test')



    # run optional prints for the homework
    dataset_here = "./data/bupa.data" 
    X_all, Y_all = load_data(dataset_here)
    score_optional = classifier.run_adaboost(X_all, Y_all, T, None, True)
    save_result_final(score_optional, 'empirical')



if __name__ == '__main__':
    main()
