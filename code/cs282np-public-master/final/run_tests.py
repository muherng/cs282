#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:18 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb
import matplotlib.pyplot as plt
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree,log_data_zw,construct_data 
import profile
from fractions import gcd
from scipy.stats import norm
import math 
from tree_paintbox import gen_tree,update,add,get_vec,access,get_FD,drop_tree,conditional_draw
import time
from varIBP import run_vi
from IBP import ugibbs_sampler
import sys
from adaptive_inference import upaintbox_sample,print_paintbox,recover_paintbox
from IBP import ugibbs_sampler,print_posterior,truncate,recover_IBP


#algorithm = 'paintbox'
algorithm = 'IBP'

data_count = 300
held_out = 100
sig = 0.1 #noise
small_x = 3
small_y = 3
big_x = 2
big_y = 2
data_dim = (small_x,small_y,big_x,big_y)
feature_count = big_x*big_y
T = small_x*small_y*big_x*big_y
data_type = 'special'
#data_type = 'random'
flag = 'four'
#full_data,Z_gen = generate_data(feature_count,data_count + held_out,T,sig,data_type)
full_data,Z_gen = construct_data(data_dim,data_count + held_out,sig,data_type)
Y = full_data[:data_count,:]
held_out = full_data[data_count:,:]
#we observe half the signal and recover the other half
observe = held_out[:,:T/2]
sig_test = 0.1
trunc = 12 #truncate active features

if algorithm == 'IBP':
    iterate = 200
    alpha = 2.0
    sig_w = 0.5 #feature deviation
    select = 12
    Z,W,ll_set,pred_ll = ugibbs_sampler(Y,held_out,alpha,sig_test,sig_w,iterate,select,trunc,data_dim)
    Z_trunc,W_trunc = truncate(Z,W,select)
    print_posterior(Z_trunc,W_trunc,data_dim)
    recover_ll = recover_IBP(held_out,observe,Z,W,sig_test)
    print("Log Recovery")
    print(recover_ll)
    print("End Uncollapsed IBP")

if algorithm == 'paintbox':
    log_res = 10 #log of res
    hold = 500 #hold resolution for # iterations
    #feature_count = 4 #features
    sig = 0.1 #noise
    sig_w = 0.5 #feature deviation #it was 0.3 for random #0.2 for corr
    iterate = 5000
    K = 1 #start with K features
    ext = 1 #draw one new feature per iteration
    outputs = upaintbox_sample(data_dim,log_res,hold,Y,held_out,ext,sig_test,sig_w,iterate,K,trunc)
    ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,tree = outputs
    print_paintbox(tree,W,data_dim,flag)
    recover_ll = recover_paintbox(held_out,observe,W,tree,sig_test)
    print("Log Recovery")
    print(recover_ll)
    print("End Paintbox Inference")
    
    
    


