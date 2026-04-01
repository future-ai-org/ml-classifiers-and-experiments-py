'''
   based on: http://nipunbatra.github.io/2013/05/simulating-a-discrete-hidden-markov-model/
   Unfair cassino problem: there may be two die, one fair and other biased. 
   The biased die is much more likely to produce a 6 than others (p=0.5).
   The observer is only able to observe the values of die being thrown without
   having a knowledge whether a fair or biased die were used:

   observed states: 1 to 6 on the die faces
   hidden states: fair or biased die
   prior: probability that the first thworn is made from a fair or biased die
   transition matrix A: matrix enconding the prob of the 4 possible transition 
       between fair and biased die
   emission matrix B: matrix enconding the prob of an obsevation given the hidden
       state
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

# plot setup
matplotlib.rcParams.update({'font.size': 11})


''' 
    setting the components of HMM
    prior: a fair die is twice as likely as biased die
    A : 
    	1. Fair -> Fair: .95
    	2. Fair -> Biased: 1-.95 =.05
    	3. Biased -> Biased: .90
    	4. Biased -> Biased: 1-.90 =.10

    B: 
    	Pr(6)= 0.5 
        Pr(1)= Pr(2)= Pr(3)= Pr(4)= Pr(5)= 0.1
'''

prior = np.array([2.0/3,1.0/3])
A = np.array([[.95,.05],[.1,.9]])
B = np.array([[1.0/6 for i in range(6)],[.1,.1,.1,.1,.1,.5]])



# return next state to the weighted probability array
def next_state(weights):
    choice = random.random() * sum(weights)
    for i, w in enumerate(weights):
        choice -= w
        if choice < 0:
            return i


def create_hidden_sequence(prior, A, length):
    out=[None]*length
    out[0]=next_state(prior)
    for i in range(1,length):
        out[i]=next_state(A[out[i-1]])
    return out


def create_observation_sequence(hidden_sequence, B):
    length=len(hidden_sequence)
    out=[None]*length
    for i in range(length):
        out[i]=next_state(B[hidden_sequence[i]])
    return out


# group all contiguous values in tuple
def group(L):
    first = last = L[0]
    for n in L[1:]:
        if n - 1 == last: 
            last = n
        else: 
            yield first, last
            first = last = n
    yield first, last 


# create tuples of the form (start, number_of_continuous values)
def create_tuple(x):
    return [(a,b-a+1) for (a,b) in x]




if __name__ == '__main__':
	count = 0
	num_calls = 500

	for i in range(num_calls):
	    count += next_state(prior)

	print("Expected number of Fair states:", num_calls-count)
	print("Expected number of Biased states:", count)

    # Create the sequences
	hidden = np.array(create_hidden_sequence(prior, A, num_calls))
	observed = np.array(create_observation_sequence(hidden, B))
	print('Observed: ', observed)
	print('Hidden: ', hidden)

	# Tuples of form index value, number of continuous values corresponding to Fair State
	indices_hidden_fair = np.where(hidden==0)[0]
	tuples_contiguous_values_fair = list(group(indices_hidden_fair))
	tuples_start_break_fair = create_tuple(tuples_contiguous_values_fair)

	# Tuples of form index value, number of continuous values corresponding to Biased State
	indices_hidden_biased = np.where(hidden==1)[0]
	tuples_contiguous_values_biased = list(group(indices_hidden_biased))
	tuples_start_break_biased = create_tuple(tuples_contiguous_values_biased)

	# Tuples for observations
	observation_tuples=[]
	for i in range(6):
	    observation_tuples.append(create_tuple(group(list(np.where(observed==i)[0]))))

    # Make plots
	plt.subplot(2,1,1)
	plt.xlim((0, num_calls));
	plt.title('Observations');
	for i in range(6):
	    plt.broken_barh(observation_tuples[i],(i+0.5,1),facecolor='k');
	plt.subplot(2,1,2);
	plt.xlim((0, num_calls));
	plt.title('Hidden States Blue:Fair, Red: Biased');
	plt.broken_barh(tuples_start_break_fair,(0,1),facecolor='b');
	plt.broken_barh(tuples_start_break_biased,(0,1),facecolor='r');
	plt.savefig('hmm.png')