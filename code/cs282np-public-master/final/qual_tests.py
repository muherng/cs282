#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:18 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import pdb
import matplotlib.pyplot as plt
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree,log_data_zw,construct_data 
import profile
from fractions import gcd
import math 
from tree_paintbox import gen_tree,update,add,get_vec,access,get_FD,drop_tree,conditional_draw
import time
from varIBP import run_vi
#from IBP import ugibbs_sampler
import sys
from qual_paintbox import upaintbox_sample,print_paintbox,recover_paintbox
from qual_IBP import ugibbs_sampler,print_posterior,truncate,recover_IBP
from load_data import load
from itertools import combinations,permutations


args = sys.argv
#choose the algorithm
#filename = args[1]
#filename = 'SVD_reconUrban_F210.npz'
#sig = float(args[1])
#obs = float(args[2]) #fraction observed
display = False
sig = .001
obs = 0.7
init_iter = 10
iterate = 100
limit = 10000
full_data = np.loadtxt('UN_long.txt')
full_data = np.loadtxt('UN.txt')
full_data = np.loadtxt('UN_noscale.txt')
full_data = np.loadtxt('UN_longnoscale.txt')
train_count = 180
test_count = 16
total_data =  train_count + test_count
#full_data = full_data[:total_data,:30]
maxarray = np.amax(full_data)
scale = 1.0/maxarray
full_data = scale * full_data
datapoints,dimension = full_data.shape
sig_w = 0.1
sig_test = sig
#full_data,Z_gen = construct_data(data_dim,data_count + held_out,sig,data_type)
if train_count + test_count > datapoints:
    data_count = datapoints
    
#indices = list(combinations([i for i in range(datapoints)],total_data))
index = np.zeros(datapoints)
index[:total_data] = np.ones(total_data)
#indices = np.where(np.random.permutation(index) == 1)[0]
indices = np.loadtxt('indices1.txt')
indices = indices.astype(int)
select_data = full_data[indices,:]
#train_indices = list(combinations([i for i in range(total_data)],train_coabs
index = np.zeros(total_data)
index[:train_count] = np.ones(train_count)
#train_indices = np.where(np.random.permutation(index) == 1)[0]
train_indices = np.loadtxt('train_indices1.txt')
train_indices = train_indices.astype(int)
train = select_data[train_indices,:]
all_indices = [i for i in range(total_data)]
test_indices = [item for item in all_indices if item not in train_indices]
test = select_data[test_indices,:]
#we observe half the signal and recover the other half
#obs = 0.7 #fraction observed
dim_indices = [i for i in range(dimension)]
#obs_indices = np.random.choice(dimension,int(dimension*obs))
#console 51
#startign console 53 
obs_indices = np.loadtxt('obs1.txt')
obs_indices = obs_indices.astype(int)
#obs_indices = [i for i in range(int(dimension*obs))]
#print(obs_indices)
observe = test[:,obs_indices]
trunc = 12 #truncate active features
#print('sig:' + str(sig))
#print('obs:' + str(obs))
#print('trunc:' + str(trunc))
#print('data count:' + str(train_count))
#print('obs_indices:')
#print(obs_indices)
#print('tran_indices')
#print(train_indices)

log_res = 10 #log of res
hold = 100 #hold resolution for # iterations
initialize = 1
#print('getting here')
if initialize:
    alpha = 2.0
    #was working for 10
    pre_trunc = 12
    #dummy variable
    select = 10
    Z_init,W_init,_,_,rec_ll,iter_time = ugibbs_sampler(train,test,alpha,
                                                sig_test,sig_w,init_iter,
                                                select,pre_trunc,observe,
                                                obs_indices,limit,init=initialize,display=display)
    zip_list = zip(iter_time,rec_ll)
    print('IBP')
    for z in zip_list:
        print(z)
    K = Z_init.shape[1]
    #print("SWITCHING TO PAINTBOX")
else:
    K = 1 #start with K features
ext = 1 #draw one new feature per iteration
outputs = upaintbox_sample(log_res,hold,train,test,ext,
                           sig_test,sig_w,iterate,K,trunc,
                           obs_indices,limit,Z_init=Z_init,W_init=W_init,init=initialize,display=display)
ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,rec_ll,tree = outputs
zip_list = zip(iter_time,rec_ll)
print('paintbox')
for z in zip_list:
    print(z)
