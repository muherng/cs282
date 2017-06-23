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
from adaptive_real import upaintbox_sample,print_paintbox,recover_paintbox
from IBP import ugibbs_sampler,print_posterior,truncate,recover_IBP
from load_data import load

#choose the algorithm
#algorithm = 'paintbox'
algorithm = 'IBP'

filename = 'SVD_reconBreast_Cancer.npz'
full_data = load(filename)
datapoints,dimension = full_data.shape
print(full_data.shape)
train_count = 500
test_count = 69
total_data =  train_count + test_count
#sig = 1./np.sqrt(2*np.pi) #noise
#sig_w = sig*150 #feature deviation
sig = 10
sig_w = 150
sig_test = sig
#full_data,Z_gen = construct_data(data_dim,data_count + held_out,sig,data_type)
if train_count + test_count > datapoints:
    data_count = datapoints
    
indices = np.random.choice(datapoints,total_data)
select_data = full_data[indices,:]
train_indices = np.random.choice(total_data,train_count)
train = select_data[train_indices,:]
all_indices = [i for i in range(total_data)]
test_indices = [item for item in all_indices if item not in train_indices]
test = select_data[test_indices,:]
#we observe half the signal and recover the other half
obs = 0.9 #fraction observed
dim_indices = [i for i in range(dimension)]
obs_indices = np.random.choice(dimension,int(dimension*obs))
observe = test[:,obs_indices]
trunc = 12 #truncate active features

if algorithm == 'IBP':
    iterate = 200
    alpha = 2.0
    select = 12
    Z,W,ll_set,pred_ll = ugibbs_sampler(train,test,alpha,sig_test,sig_w,iterate,select,trunc,observe,obs_indices)
    Z_trunc,W_trunc = truncate(Z,W,select)
    #print_posterior(Z_trunc,W_trunc,data_dim)
    recover_ll = recover_IBP(test,observe,Z,W,sig_test,obs_indices)
    print("Log Recovery")
    print(recover_ll)
    print("End Uncollapsed IBP")

if algorithm == 'paintbox':
    log_res = 10 #log of res
    hold = 300 #hold resolution for # iterations
    iterate = 3000
    K = 1 #start with K features
    ext = 1 #draw one new feature per iteration
    outputs = upaintbox_sample(log_res,hold,train,test,ext,sig_test,sig_w,iterate,K,trunc,obs_indices)
    ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,tree = outputs
    #print_paintbox(tree,W,data_dim,flag)
    recover_ll = recover_paintbox(test,observe,W,tree,sig_test,obs_indices)
    print("Log Recovery")
    print(recover_ll)
    print("End Paintbox Inference")