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
import matplotlib.pyplot as plt
from tree_paintbox import gen_tree,update,add,get_vec,get_FD
import math

#size of paintbox
res = 2
F = 4 #features
D = res**F #discretization
T = 36 #length of datapoint

#data size 
N = 100

#initalize features
#this is a hack, W is F by T matrix 
W = np.eye(F)

#noise parameter
sig = 0.1

def lcm(*numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)
        

def generate_gg_blocks(): 
    W = np.zeros( [ 4 , 36 ] ) 
    W[0, [ 1 , 6 , 7 , 8 , 13 ] ] = 1
    W[1, [ 3 , 4 , 5 , 9 , 11 , 15 , 16 , 17  ] ] = 1 
    W[2, [ 18 , 24 , 25 , 30 , 31 , 32 ] ] = 1
    W[3, [ 21 , 22 , 23 , 28 , 34 ] ] = 1 
    #W = W - 0.5*np.ones([4,36])
    return W 

def generate_blocks(small_x,small_y,big_x,big_y):
    features = big_x*big_y
    block = small_x*small_y
    signal_length = features*block
    W = np.zeros([features,signal_length])
    density = [5,6,7,8]
    roulette = [1./len(density) for r in range(len(density))]
    for i in range(features):
        choice = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
        fill = density[choice]
        new_feature = np.random.permutation(np.array([1]*fill + [0]*(block-fill)))
        W[i,block*i:block*(i+1)] = new_feature
    display_W(W)
    return W

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
    
def draw_Z_tree(tree,N):
    F,D = get_FD(tree)
    vec = get_vec(tree)
    normal_vec = 1./np.sum(vec) * vec
    draws = np.random.multinomial(N, normal_vec)
    ctree,ptree = tree
    Z = np.zeros((N,F))
    cum_draws = np.cumsum(draws)
    #generate Z
    for i in range(D):
        density = draws[i]
        binary = list(map(int,"{0:b}".format(i)))
        row = [0]*(F-len(binary)) + binary
        data_chunk = np.tile(row,(density,1))
        if i == 0:
             Z[0:cum_draws[i],:] = data_chunk
        else:    
            Z[cum_draws[i-1]:cum_draws[i],:] = data_chunk
    #np.random.shuffle(Z)
    return Z
    

#this is for toy image W    
def display_W(W):
    if len(W.shape) == 1:
        T, = W.shape
        dim = int(np.sqrt(T))
        image_format = np.reshape(W,(dim,dim))
        plt.imshow(image_format,interpolation='nearest')
        plt.gray()
        plt.show()
    else:
        F,T = W.shape
        image_format = []
        for i in range(F):
            dim = int(np.sqrt(T))
            image_format.append(np.reshape(W[i,:],(dim,dim)))
            plt.imshow(image_format[i],interpolation='nearest')
            plt.gray()
            plt.show()
    return 
 

def log_data_zw(Y,Z,W,sig):
    NK = 1
    if len(Y.shape) == 1:
        NK = Y.shape[0]
    else:
        NK = Y.shape[0]*Y.shape[1]
    delta = Y - np.dot(Z,W)
    if len(Y.shape) == 1:
        delta_sum = np.dot(delta,delta)   
    else:
        delta_sum = np.trace(np.dot(delta.T,delta))
    #print(NK)
    ll =  -1./(2*sig**2) * delta_sum - NK*(0.5*math.log(2*np.pi) + math.log(sig))
    return ll       

def generate_data(F,N,T,sig,data_type):
    #pb = pb_partition(D,F)
    #pb = pb_random(res,D,F)
    #print("PAINTBOX GENERATING")
    #print(pb)
    #This line is problematic, does not adapt to T
    #W = np.hstack((np.eye(F),np.eye(F)))
    W = generate_gg_blocks()
    Y = np.zeros((N,T))
    E = np.reshape(np.random.normal(0,sig,N*T),(N,T))
    
    #Z = draw_Z(pb,D,F,N,T)
    Z = np.zeros([N,F])
    if data_type == 'random':
        Z = np.random.binomial( 1 , .25 , [ N , F ] )
    if data_type == 'anti':
        roulette = [1./F for i in range(F)]
        indices = np.eye(F)
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = indices[index,:]
    if data_type == 'corr':
        
        roulette = [1./6 for i in range(6)]
        indices = np.array([[0,0,1,1],[0,1,0,1],[1,0,0,1],[0,1,1,0],[1,0,1,0],[1,1,0,0]])
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = indices[index,:]        
    Z = np.random.permutation(Z)
    
    #for i in range(N):
    #    Z[i,:] = 
    
    #for debugging the uncollapsed IBP we're going to fix the Z
    #Z = np.zeros([N,F])
    #Z[:,0] = np.random.binomial(1,0.5,N)
    
    
    #consider ignoring the noise
    #Y = np.dot(Z,W) + E
    #Y = np.dot(Z,W) + E
    Y = np.dot(Z,W) + E
    return (Y,Z)    

