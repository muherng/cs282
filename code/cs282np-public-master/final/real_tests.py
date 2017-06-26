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
from itertools import combinations,permutations

args = sys.argv
#print(args)
#choose the algorithm
algorithm = args[1]
sig = float(args[2])
#algorithm = 'IBP'
filename = 'SVD_reconBreast_Cancer.npz'
limit = 10000
full_data = load(filename)
datapoints,dimension = full_data.shape
#print(full_data.shape)
train_count = 500
test_count = 69
total_data =  train_count + test_count
#sig = 1./np.sqrt(2*np.pi) #noise
#sig_w = sig*150 #feature deviation
#sig = 3
sig_w = 150
sig_test = sig
#full_data,Z_gen = construct_data(data_dim,data_count + held_out,sig,data_type)
if train_count + test_count > datapoints:
    data_count = datapoints
    
#indices = list(combinations([i for i in range(datapoints)],total_data))
index = np.zeros(datapoints)
index[:total_data] = np.ones(total_data)
indices = np.where(np.random.permutation(index) == 1)[0]
select_data = full_data[indices,:]
#train_indices = list(combinations([i for i in range(total_data)],train_coabs
index = np.zeros(total_data)
index[:train_count] = np.ones(train_count)
train_indices = np.where(np.random.permutation(index) == 1)[0]
train = select_data[train_indices,:]
all_indices = [i for i in range(total_data)]
test_indices = [item for item in all_indices if item not in train_indices]
test = select_data[test_indices,:]
#we observe half the signal and recover the other half
obs = 0.7 #fraction observed
dim_indices = [i for i in range(dimension)]
#obs_indices = np.random.choice(dimension,int(dimension*obs))
obs_indices = [i for i in range(int(dimension*obs))]
#print(obs_indices)
observe = test[:,obs_indices]
#print(observe.shape)

if algorithm == 'IBP':
    trunc = 12
    iterate = 800
    alpha = 2.0
    #dummy variable 
    #kept around because we may have to truncate at future date
    select = 10
    #print("test shape")
    #print(test.shape)
    Z,W,ll_set,pred_ll,rec_ll,iter_time = ugibbs_sampler(train,test,alpha,sig_test,
                                                         sig_w,iterate,select,trunc,
                                                         observe,obs_indices,limit)
    zip_list = zip(iter_time,rec_ll)
    for z in zip_list:
        print(z)
    #Z_trunc,W_trunc = truncate(Z,W,select)
    #print_posterior(Z_trunc,W_trunc,data_dim)
    #recover_ll = recover_IBP(test,observe,Z,W,sig_test,obs_indices)
    #print("Log Recovery")
    #print(recover_ll)
    #print("End Uncollapsed IBP")
    #np.savetxt("rec" + process + ".txt", rec_ll)
    #np.savetxt("time" + process + ".txt", iter_time)

if algorithm == 'paintbox':
    trunc = 12 #truncate active features
    log_res = 10 #log of res
    hold = 100 #hold resolution for # iterations
    iterate = 10
    initialize = True
    if initialize:
        alpha = 2.0
        pre_trunc = 10
        init_iter = 10
        #dummy variable
        select = 10
        Z_init,W_init,_,_,_,_ = ugibbs_sampler(train,test,alpha,
                                                    sig_test,sig_w,init_iter,
                                                    select,pre_trunc,observe,
                                                    obs_indices,limit,init=initialize)
        print('initial recover log likelihood:' + str(recover_IBP(test,observe,Z_init,W_init,sig,obs_indices)))
        K = Z_init.shape[1]
    else:
        K = 1 #start with K features
    ext = 0 #draw one new feature per iteration
    outputs = upaintbox_sample(log_res,hold,train,test,ext,
                               sig_test,sig_w,iterate,K,trunc,
                               obs_indices,limit,Z_init=Z_init,W_init=W_init,init=initialize)
    ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,rec_ll,tree = outputs
    zip_list = zip(iter_time,rec_ll)
    for z in zip_list:
        print(z)
    #print_paintbox(tree,W,data_dim,flag)
    #recover_ll = recover_paintbox(test,observe,W,tree,sig_test,obs_indices)
    #print("Log Recovery")
    #print(recover_ll)
    #print("End Paintbox Inference")
    #np.savetxt("rec" + process + ".txt", rec_ll)
    #np.savetxt("time" + process + ".txt", iter_time)
