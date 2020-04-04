#!/usr/bin/python

from pylab import *
from random import random
import sys


from replacement_model import xmax, next_state_and_reward


gamma = 0.8 # discount factor


##########################        
### Generation of samples

# generate x samples

N = 150 #  number of samples points
M = 10 # number of next_state_and_reward samples
K=30 # number of updating alpha

d = 30

step = xmax/(N-1)
x_samples = arange(0.0, xmax+step, step)
#x_samples = array( sorted ([ random()*xmax for i in xrange(N) ]) ) # random


# generate y0,r0, y1,r1 samples

print("Generation of samples...")

y0_samples, r0_samples, y1_samples, r1_samples = [], [], [], []
for x in x_samples:
    y0_l,r0_l,y1_l,r1_l=[],[],[],[]
    for j in range(M):
        y0,r0=next_state_and_reward(x,0)
        y0_l.append(y0)
        r0_l.append(r0)
        y1,r1=next_state_and_reward(x,1)
        y1_l.append(y1)
        r1_l.append(r1)
    y0_samples.append(y0_l)
    r0_samples.append(r0_l)
    y1_samples.append(y1_l)
    r1_samples.append(r1_l)




 

################################
### Discretized Value Iteration

def index_position(y): # returns the index of the closest x in x_samples
    i=int(N/2)
    step=int(N/4)
    while True:
        if x_samples[i]>y:
            i=i-step
            if i<0:
                return 0
        elif x_samples[i+1]<y:
            i=i+step
            if i>N-2:
                return N-2
        else:
            return i
        step=max(1,int(step/2))

def value_iterate(alpha):
    alpha2,pol=[],[]
    for i in range(N):
        print((i+1),"/",N,"   \r", end=' ')
        sys.stdout.flush()
        Tv0,Tv1=0,0
        for j in range(M):
            Tv0 += r0_samples[i][j] + gamma * alpha[ index_position(y0_samples[i][j]) ]
            Tv1 += r1_samples[i][j] + gamma * alpha[ index_position(y1_samples[i][j]) ]
        if Tv0<=Tv1:
            Tv=Tv0/M
            pol.append(0)
        else:
            Tv=Tv1/M
            pol.append(1)
        alpha2.append(Tv)
    return alpha2,pol


def value_iteration(K):
    alpha=zeros(N)
    for i in range(K):
        alpha, pol = value_iterate(alpha)
        print(i,"             ")#,pol
    return alpha, pol
    

###########################
### Fitted Value Iteration

def return_cos(j, x, x_max=xmax):
    return np.cos(j*np.pi*x/x_max)

def compute_F(x):
    F = []
    for j in range(d):
        F.append(return_cos(j, x))
    F = np.array(F)
    return F


def return_value(alpha, x):
    res = 0
    for j in range(len(alpha)):
        res += alpha[j] * return_cos(j, x)
    return res

def initiate_alpha():
    return zeros(d)

Fi = compute_F(x_samples).T

def fitted_value_iterate(alpha):
    alpha2,pol=[],[]
    Tvs = []
    for i in range(N):
        print((i+1),"/",N,"   \r", end=' ')
        sys.stdout.flush()
        Tv0,Tv1=0,0
        for j in range(M):
            Tv0 += r0_samples[i][j] + gamma * return_value(alpha, y0_samples[i][j])
            Tv1 += r1_samples[i][j] + gamma * return_value(alpha, y1_samples[i][j])
        if Tv0<=Tv1:
            Tv=Tv0/M
            pol.append(0)
        else:
            Tv=Tv1/M
            pol.append(1)
        Tvs.append(Tv)
    Tvs = np.array(Tvs)
    alpha2 = np.linalg.inv((np.transpose(Fi) @ Fi)) @ (np.transpose(Fi) @ Tvs)
    return alpha2,pol

def fitted_value_iteration(K):
    alpha=initiate_alpha()
    for i in range(K):
        alpha, pol = fitted_value_iterate(alpha)
        print(i,"             ")#,pol
    values = return_value(alpha, x_samples)
    return values, pol


###########################
### Fitted Q Iteration


def fitted_q_iterate(alpha0, alpha1):
    pol=[]
    Tvs0, Tvs1  = [],[]
    for i in range(N):
        #print((i+1),"/",N,"   \r", end=' ')
        sys.stdout.flush()
        Tv0,Tv1=0,0
        for j in range(M):
            min_q0 = np.min(np.array([return_value(alpha0, y0_samples[i][j]), return_value(alpha1, y0_samples[i][j])]))
            min_q1 = np.min(np.array([return_value(alpha0, y1_samples[i][j]), return_value(alpha1, y1_samples[i][j])]))
            Tv0 += r0_samples[i][j] + gamma * min_q0
            Tv1 += r1_samples[i][j] + gamma * min_q1
        Tvs0.append(Tv0/M)
        Tvs1.append(Tv1/M)
    Tvs0 = np.array(Tvs0).reshape(N, 1)
    Tvs1 = np.array(Tvs1).reshape(N, 1)
    alpha0_2 = np.linalg.inv((np.transpose(Fi) @ Fi)) @ (np.transpose(Fi) @ Tvs0)
    alpha1_2 = np.linalg.inv((np.transpose(Fi) @ Fi)) @ (np.transpose(Fi) @ Tvs1)
    return alpha0_2, alpha1_2, pol

def fitted_q_iteration(K):
    alpha0, alpha1=initiate_alpha(), initiate_alpha()
    for i in range(K):
        alpha0, alpha1, pol = fitted_q_iterate(alpha0, alpha1)
        print(i,"             ")#,pol

    qa0 = [return_value(alpha0, x_sample) for x_sample in x_samples]
    qa1 = [return_value(alpha1, x_sample) for x_sample in x_samples]
    values = np.min(np.stack((qa0, qa1)), axis=0)
    pol = np.argmin(np.stack((qa0, qa1)), axis=0)
    return values, pol

#################################################################################################
### MAIN 
#################################################################################################

    
#"""
i=0
for f in [value_iteration,fitted_q_iteration,fitted_value_iteration]:
    v,pol = f(K)
    figure(0)
    plot (x_samples,v, label=f.__name__)
    print(f.__name__)

    xlabel(('level of use'))
    ylabel(('value'))
    legend()
    figure(1)
    plot (x_samples,i+array(pol),linewidth=2, label=f.__name__)

    xlabel(('level of use'))
    ylabel(('action'))
    
    i=i+2
legend()
show()
#"""