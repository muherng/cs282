#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:39:10 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 

print("Hello")


#size of paintbox
F = 2 #features
D = 2**F #discretization
T = F #length of datapoint

#data size 
N = 10

#initalize features
#this is a hack, W is F by T matrix 
W = np.eye(F)

#noise parameter
sig = 0.1

#generate data
#set paintbox to be anything

def pb_partition(D,F):
    pb = np.zeros((F,D))
    for i in range(F):
        print(D/F)
        try:
            pb[i,i*D/F:(i+1)*D/F] = [1]*(D/F)
        except ValueError: 
            print("Value Error")
    return pb
    
def pb_init(D,F):
    pb = np.zeros((F,D))
    for i in range(F):
        pb[i,:] = np.tile([1]*(D/(2**(i+1))) + [0]*(D/(2**(i+1))),2**i)
    return pb
    
def draw_Z(pb,D,F,N,T):
    draws = np.random.multinomial(N, [1.0/D]*D, size=1)
    print(draws)
    Z = np.zeros((N,F))
    cum_draws = np.cumsum(draws)
    #generate Z
    for i in range(D):
        density = draws[0,i]
        row = pb[:,i]
        data_chunk = np.tile(row,(density,1))
        if i == 0:
             Z[0:cum_draws[i],:] = data_chunk
        else:    
            Z[cum_draws[i-1]:cum_draws[i],:] = data_chunk
    return Z

def generate_data(D,F,N,T,sig):
    pb = pb_partition(D,F)
    print(pb)
    #This line is problematic, does not adapt to T
    W = np.eye(F)
    Y = np.zeros((N,T))
    E = np.reshape(np.random.normal(0,sig,N*T),(N,T))
    
    Z = draw_Z(pb,D,F,N,T)
    #consider ignoring the noise
    #Y = np.dot(Z,W) + E
    Y = np.dot(Z,W) + E
    return Y    
    
Y = generate_data(D,F,N,T,sig)
print(Y)



