#!/usr/bin/python

from pylab import *
from random import random



####################################
### Replacement model (original) ###


xmax = 10 # maximum level of use
xdead = 9 # level of death
Creplace = 40 # cost of replacement
Cdead = 10; # cost of death


beta = 2.0 # exponential distribution parameter



def maintenance_cost(x,y):

    c=y-x
    if x < 2.5 < y:
        c=c+10
    if x < 5 < y:
        c=c+20
    if x < 7.5 < y:
        c=c+30
    return c



def sample_exp( beta ): # samples from the exponential distribution of parameter beta
    return -1.0/beta*log( random() )
            
        
def next_state_and_reward( x, a):
    c=0
    if a==0: # replace
        y=sample_exp(beta)
        c+=Creplace + maintenance_cost(0,y)
    else: # keep
        y = x + sample_exp(beta)
        c += maintenance_cost(x, y)
    if y>xdead:
        y, c2 = next_state_and_reward ( y, 0 )
        c += Cdead + c2
    return y,c


 
