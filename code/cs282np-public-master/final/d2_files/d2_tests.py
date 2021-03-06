#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:18 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
#import scipy.special as sps
#import scipy.stats as SPST
import pdb
import matplotlib.pyplot as plt
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree,log_data_zw,construct_data 
import profile
from fractions import gcd
#from scipy.stats import norm
import math 
from tree_paintbox import gen_tree,update,add,get_vec,access,get_FD,drop_tree,conditional_draw
import time
from varIBP import run_vi
from IBP import ugibbs_sampler
import sys
from d2_paintbox import upaintbox_sample,print_paintbox,recover_paintbox
from d2_IBP import ugibbs_sampler,print_posterior,truncate,recover_IBP
from load_data import load
from itertools import combinations,permutations

args = sys.argv
#print(args)
#choose the algorithm
#algorithm = args[1]
sig = float(args[1])
obs = float(args[2]) #fraction observed

#good setting: .01 or .005, obs 0.7, first 30
#algorithm = 'paintbox'
#sig = 0.001
#obs = 0.7
init_iter = 10
iterate = 10
display= False
filename = 'SVD_reconUrban.npz'
limit = 10000
full_data = load(filename)

train_count = 500
test_count = 100
total_data =  train_count + test_count
full_data = full_data[:total_data,:30]
full_data = full_data*.001
datapoints,dimension = full_data.shape

sig_w = 0.1
sig_test = sig
if train_count + test_count > datapoints:
    data_count = datapoints

index = np.zeros(datapoints)
index[:total_data] = np.ones(total_data)
indices = np.where(np.random.permutation(index) == 1)[0]
select_data = full_data[indices,:]
index = np.zeros(total_data)
index[:train_count] = np.ones(train_count)
train_indices = np.where(np.random.permutation(index) == 1)[0]
train = select_data[train_indices,:]
all_indices = [i for i in range(total_data)]
test_indices = [item for item in all_indices if item not in train_indices]
test = select_data[test_indices,:]
dim_indices = [i for i in range(dimension)]
obs_indices = np.random.choice(dimension,int(dimension*obs))
#obs_indices = [i for i in range(int(dimension*obs))]
#print(obs_indices)
observe = test[:,obs_indices]

trunc = 12 #truncate active features
log_res = 10 #log of res
hold = 100 #hold resolution for # iterations
initialize = True
if initialize:
    alpha = 2.0
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
else:
    K = 1 #start with K features
ext = 0 #draw one new feature per iteration
outputs = upaintbox_sample(log_res,hold,train,test,ext,
                           sig_test,sig_w,iterate,K,trunc,
                           obs_indices,limit,Z_init=Z_init,W_init=W_init,init=initialize,display=display)
ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,rec_ll,tree = outputs
zip_list = zip(iter_time,rec_ll)
print('paintbox')
for z in zip_list:
    print(z)
