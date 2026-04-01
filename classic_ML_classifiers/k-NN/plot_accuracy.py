'''
Normalization plots for each feature and each type of graophs.
The normalization shows the feature in tersm of the order of the network.

mari wahl @ 2014
'''


import pylab
import os
import numpy as np

__author__ = """Marina von Steinkirch"""

# plot config
pylab.rcParams.update({'font.size': 12})
color = ['#FF4848', '#800080', '#5757FF', '#1FCB4A', '#D9C400', '#F70000', '#0000CE', '#FF800D', '#23819C']
marker = ['o', 'v', 's', '*', 'D', '>', '<', 'p', '8']


   

def main():  

   FILE1 = "./TO_PLOT.dat" 
   FILE2 = "./test.dat" 
   FILE3 = "./train.dat" 
   PATH_TO_OUTPUT = "./plot.png"


   f1 = pylab.loadtxt(FILE1,dtype = str, unpack=True)
   f2 = pylab.loadtxt(FILE2,dtype = str, unpack=True)
   f3 = pylab.loadtxt(FILE3,dtype = str, unpack=True)

   pylab.clf()
   pylab.cla()

   pylab.scatter(f1[0], f1[1], c=color[1], label='10-fold KNN')
   pylab.plot(f2[0], f2[1],  c=color[2], label='Test data')
   pylab.scatter(f3[0], f3[1],  c=color[3], label='Train data')

   pylab.ylabel('Acccuracy', fontsize=14)  
   pylab.xlabel('k', fontsize=14)       
   pylab.legend(loc=1, prop={'size':10})
   pylab.xlim(0.1,100)

		    
   pylab.savefig(PATH_TO_OUTPUT)



if __name__ == '__main__':
    main()

