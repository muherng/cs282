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
from fractions import gcd

#size of paintbox
res = 2
F = 2 #features
D = res**F #discretization
T = F #length of datapoint

#data size 
N = 10

#initalize features
#this is a hack, W is F by T matrix 
W = np.eye(F)

#noise parameter
sig = 0.1

def lcm(*numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)

def scale(pb):
    F,D = pb.shape
    lcm = (F * D) // gcd(F, D)
    v_scale = lcm/F
    h_scale = lcm/D
    pb_scale = np.zeros((lcm,lcm))
    for i in range(F):
        for j in range(D):
            try:
                pb_scale[i*v_scale:(i+1)*v_scale,j*h_scale:(j+1)*h_scale] = pb[i,j] * np.ones((v_scale,h_scale))
            except IndexError:
                print("IndexError")
    return pb_scale

#partition paintbox
def pb_partition(D,F):
    pb = np.zeros((F,D))
    for i in range(F):
        print(D/F)
        try:
            pb[i,i*D/F:(i+1)*D/F] = [1]*(D/F)
        except ValueError: 
            print("Value Error")
    return pb

#uniform draw from random prior
#arbitrary paintbox is easy, legal paintbox 
def pb_random(res,D,F):
    pb = np.zeros((F,D))
    row_count = [D]
    for i in range(F):
        new_row_count = []
        #iterate over nodes j in tree layer i 
        #has to be better way of doing this
#        if i != 0:
#            ind = pb[i-1,0]
#            count = 0
#            for k in range(D):
#                if ind == pb[i-1,k]:   
#                    count+=1
#                else:
#                    row_count.append(count)
#                    ind = 1-ind
#                    count = 1
#            row_count.append(count)
#        else:
#            row_count.append(D)
        cum_count = np.cumsum(row_count)
        cum_count = np.insert(cum_count,0,0)
        for j in range(2**i):
            try:
                unit = row_count[j]/res
            except IndexError:
                print("IndexError")
            draw = int(np.where(np.random.multinomial(1,[1./(res+1)]*(res+1)) == 1)[0])
            try:
                pb[i,cum_count[j]:cum_count[j+1]] = [1]*unit*draw + [0]*unit*(res-draw)
            except IndexError:
                print("IndexError")
            new_row_count = new_row_count + [unit*draw,unit*(res-draw)]
        row_count = new_row_count
    return pb
#1/2 tree intialization  
#note that you can't do this for res not divisible by 2  
def pb_init(D,F):
    pb = np.zeros((F,D))
    for i in range(F):
        pb[i,:] = np.tile([1]*(D/(2**(i+1))) + [0]*(D/(2**(i+1))),2**i)
    return pb
    
def draw_Z(pb,D,F,N,T):
    draws = np.random.multinomial(N, [1.0/D]*D, size=1)
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

def generate_data(res,D,F,N,T,sig):
    #pb = pb_partition(D,F)
    pb = pb_random(res,D,F)
    #print("PAINTBOX GENERATING")
    #print(pb)
    #This line is problematic, does not adapt to T
    W = np.eye(F)
    Y = np.zeros((N,T))
    E = np.reshape(np.random.normal(0,sig,N*T),(N,T))
    
    Z = draw_Z(pb,D,F,N,T)
    #consider ignoring the noise
    #Y = np.dot(Z,W) + E
    Y = np.dot(Z,W) + E
    return (Y,pb)    
    
#Y,pb = generate_data(res,D,F,N,T,sig)



