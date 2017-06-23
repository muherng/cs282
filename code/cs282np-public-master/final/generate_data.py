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
import itertools

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
    W = 500*1./np.sqrt(2*np.pi)*W
    return W 

def generate_blocks(data_dim):
    small_x,small_y,big_x,big_y = data_dim
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
    #W = W - 0.5*np.ones(W.shape)
    display_W(W,data_dim,'nine')
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
    
def super_format(W,small_x,small_y,big_x,big_y):
    image_format = np.zeros([small_y*big_y,small_x*big_x])
    w_dim = 0
    for i in range(big_y):
        for j in range(big_x):
            for k in range(small_y):
                for l in range(small_x):
                    x_dim = l + j*small_x
                    y_dim = k + i*small_y
                    #w_dim = l + k*small_x + j*small_x*small_y + i*small_x*small_y*big_x
                    image_format[y_dim,x_dim] = W[w_dim]
                    w_dim = w_dim + 1
    return image_format
#this is for toy image W    
#x goes across
#y goes down
#we fill across and then down
def display_W(W,data_dim,flag = 'four'):
    small_x,small_y,big_x,big_y = data_dim
    if flag == 'four':
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
    else:
        if len(W.shape) == 1:
            T, = W.shape
            image_format = super_format(W,small_x,small_y,big_x,big_y)
            plt.imshow(image_format,interpolation='nearest')
            plt.gray()
            plt.show()
        else:
            F,T = W.shape
            image_set = []
            for im in range(F):
                image_format = super_format(W[im,:],small_x,small_y,big_x,big_y)
                image_set.append(image_format)
                plt.imshow(image_set[im],interpolation='nearest')
                plt.gray()
                plt.show()     
    return 0 
 

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
    #ll =  -1./(2*sig**2) * delta_sum - NK*(0.5*math.log(2*np.pi) + math.log(sig))
    ll =  -1./(2*sig**2) * delta_sum
    return ll 


def construct_data(data_dim,N,sig,data_type,corr_value=2):
    small_x,small_y,big_x,big_y = data_dim
    #W = generate_blocks(data_dim) #features
    W = generate_gg_blocks()
    F = big_x*big_y #number of features
    T = small_x*small_y*big_x*big_y #dimension of feature
    Y = np.zeros((N,T)) #init data
    E = np.reshape(np.random.normal(0,sig,N*T),(N,T)) #noise
    Z = np.zeros([N,F]) #init feature matrix
    if data_type == 'random':
        Z = np.random.binomial( 1 , .25 , [ N , F ] )
    if data_type == 'anti':
        roulette = [1./F for i in range(F)]
        indices = np.eye(F)
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = indices[index,:]
    if data_type == 'corr': 
        base = [1]*corr_value + [0]*(F-corr_value)
        indices = np.array(list(itertools.permutations([j for j in range(F)],F)))
        roulette = [1./len(indices) for i in range(len(indices))]
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = [base[indices[index,k]] for k in range(F)]  
    if data_type == 'special':
        #this will change with 
        #pattern = np.array([[0,1,0,1],[1,1,0,0],[0,0,1,1],[1,0,1,0])
        pattern = np.array([[1,0,1,0],[1,1,0,0],[0,1,0,1],[0,0,1,1]])
        combo = pattern.shape[0]
        #we keep two copies so that 8 features is enough
        pattern = np.concatenate((pattern,pattern,pattern,pattern))
        
        hold = 0
        #roulette = 1./combo * np.ones(combo)
        roulette = np.array([0.5,0.25,0.125,0.0625])
        #adding together 4 times
        #choose = [1]*4 + [0]*12
        for overlap in range(4):
            for i in range(N):
                index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
                #val = np.where(np.random.permutation(choose) == 1)[0]
                #for index in val:
                Z[i,:] = Z[i,:] + pattern[index,:]
    if data_type == 'debug':
        pattern = np.array([[1,0,1,0],[1,0,0,0],[0,0,1,0],[0,0,0,0]])
        roulette = np.array([0.55,0.05,0.05,0.35])
        combo = pattern.shape[0]
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = pattern[index,:]
    if data_type == 'debug-four':
        pattern = np.array([[1,0,1,0],[1,0,0,0],
                            [0,0,1,0],[0,0,0,0],
                            [0,1,0,1],[0,1,0,0],
                            [0,0,0,1],[0,0,0,0]])
        roulette = np.array([0.16,0.02,
                             0.02,0.1,
                             0.16,0.02,
                             0.02,0.1])
        combo = pattern.shape[0]
        for i in range(N):
            index = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            Z[i,:] = pattern[index,:]
        
    #Z = np.random.permutation(Z)
    #Y = np.dot(Z,W) + E
    Y = np.dot(Z,W) 
    display_W(Y[1:10],'four')
    return (Y,Z)  

      

def generate_data(F,N,T,sig,data_type):
    W = generate_gg_blocks()
    Y = np.zeros((N,T))
    E = np.reshape(np.random.normal(0,sig,N*T),(N,T))
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
    Y = np.dot(Z,W) + E
    return (Y,Z)  

  

