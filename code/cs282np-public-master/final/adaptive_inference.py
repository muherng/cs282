#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 22:50:22 2017
@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb
import matplotlib.pyplot as plt
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree,log_data_zw,construct_data,generate_gg_blocks 
import profile
from fractions import gcd
from scipy.stats import norm
import math 
from tree_paintbox import gen_tree,update,add,get_vec,access,get_FD,drop_tree,conditional_draw
import time
from varIBP import run_vi
from IBP import ugibbs_sampler
import sys

#vector representation of paintbox
def vectorize(pb):
    F,D = pb.shape
    vec = np.zeros(D)
    for i in range(D):
        #not proud of this at all
        try:
            num = int(''.join(map(str, pb[:,i])).replace(".0",""),2)           
        except ValueError:
            print("ValueError")
        vec[num] += 1
    return vec

#paintbox representation of vector
def devectorize(vec):
    vec = map(int,vec)
    total = int(np.sum(vec))
    unroll = np.zeros(total)
    pb = np.zeros((F,D))
    flag = 0
    for i in range(len(vec)):
        unroll[flag:flag+vec[i]] = i
        flag = flag + vec[i]
    lof_unroll = unroll[::-1] 
    for j in range(total):
        binary = map(int,"{0:b}".format(int(lof_unroll[j])))
        pad_binary = [0]*(F-len(binary)) + binary
        pb[:,j] = pad_binary
        #print(pb)
    return pb


def log_w_sig(W,sig):
    F,T = W.shape
    like = 0
    for i in range(F):
        for j in range(T):
            like = like + -0.5 * (1./(sig**2)) * W[i,j]**2
    return like
    

#probability of Z given the vectorized paintbox 
#henceforth we will be working over vectorized paintboxes.
#we unvectorize only for visual debugging. 
#LOG LIKELIHOODS HENCEFORTH 
def Z_vec(Z,vec):
    N,F = Z.shape
    zp = 1.0
    broken = False
    for i in range(N):
        index = 0
        for j in range(F):
            index = int(index + Z[i,j]*(2**(F-j-1)))
        if vec[index] == 0:
            broken = True
            #print("Z given Vector Invariant Break")
            #sys.exit()
        zp = zp+np.log(float(vec[index])) 
    return (zp,broken)
    
def debug(Z,vec,D):
    N,F = Z.shape
    zp = 1.0
    for i in range(N):
        z_index = int(''.join(map(str, Z[i,:])).replace(".0",""),2)
        zp = zp*float(vec[z_index])
        if float(vec[z_index])/D == 0.0:
            return z_index
    return -1

#performs Z_vec for a single row
def Z_paintbox(z_row,vec):
    z_index = int(''.join(map(str, z_row)).replace(".0",""),2)
    zp = float(vec[z_index])
    return zp
    
#def log_data_zw(Y,Z,W,sig):
#    delta = Y - np.dot(Z,W)
#    delta_sum = np.trace(np.dot(delta.T,delta))
#    ll =  -1./(2*sig**2) * delta_sum
#    return ll

#def log_data_zw(Y,Z,W,sig):
#    delta = Y - np.dot(Z,W)
#    if len(Y.shape) == 1:
#        delta_sum = np.dot(delta,delta)   
#    else:
#        delta_sum = np.trace(np.dot(delta.T,delta))
#    ll =  -1./(2*sig**2) * delta_sum
#    return ll
    
def log_collapsed(Y,Z,sig,sig_w):
    N,D = Y.shape
    K = Z.shape[1]
    try: 
        tmp = np.linalg.inv(np.dot(Z.T, Z) + (float(sig)/sig_w)**2*np.eye(K))
    except np.linalg.linalg.LinAlgError:
        print('singular matrix')
        
    try:
        tmp = np.eye(N)-np.dot(np.dot(Z, tmp), Z.T)
    except ValueError:
        print('Value Error')
    tmp = -1/(2*sig**2)*np.trace(np.dot(np.dot(Y.T, tmp), Y))
    try:
        ll = tmp - (float(N*D)/2*np.log(2*np.pi)+(N-K)*D*np.log(sig)+K*D*np.log(sig_w)+float(D)/2*np.log(np.linalg.det(np.dot(Z.T,Z)+(float(sig)/sig_w)**2*np.eye(K))))
    except np.linalg.linalg.LinAlgError:
        print('LinAlgError')
    
    return ll 

# Uncollapsed Likelihood 
def log_uncollapsed(Y,Z,W,sig):
    shape = Y.shape
    if len(shape) == 1:
        N = shape[0]
        T = 1
        YZW = Y - np.dot(Z,W)
        trYZW = np.dot(YZW.T,YZW)
    else:
        N,T = shape
        YZW = Y - np.dot(Z,W)
        trYZW = np.trace(np.dot(YZW.T,YZW))
    ll = (-1.0*float(N*T)/2) * np.log(2*np.pi*sig**2) - 1.0/(2*sig**2) * trYZW
    return ll
    
def add_feature(i,Y,Z,W,tree,vec,prior,sig,sig_w):
    N,T = Y.shape
    N,K = Z.shape
    old = log_data_zw(Y,Z,W,sig)
    col = np.zeros((N,1))
    col[i,0] = 1
    Z_new = np.hstack((Z,col))
    W_new = np.vstack((W,np.random.normal(0,sig_w,(1,T))))
    #W_new = sample_W(Y,Z_new,sig,sig_w)
    new = log_data_zw(Y,Z_new,W_new,sig)
    new = new - old
    old = 0
    roulette = [np.exp(old)*prior[0],np.exp(new)*prior[1]]
    normal_roulette = [float(r)/np.sum(roulette) for r in roulette]
    chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
    if chosen:
        Z = Z_new
        W = W_new
        tree = add(tree,res)
        vec = get_vec(tree)
    return (Z,W,tree,vec)

def sample_Z(Y,Z,W,sig,sig_w,tree):
    N,T = Y.shape
    N,K = Z.shape
    prob_matrix = np.zeros([N,K])
    vec = get_vec(tree)
    for i in range(N):
        for j in range(K):
            Z_one = np.copy(Z)
            Z_zero = np.copy(Z)
            Z_one[i,j] = 1
            Z_zero[i,j] = 0
            zp_one = Z_paintbox(Z_one[i,:],vec)
            zp_zero = Z_paintbox(Z_zero[i,:],vec)
            if zp_one == 0 or zp_zero == 0:
                if zp_one == 0:
                    Z[i,j] == 0
                    prob_matrix[i,j] = 0
                if zp_zero == 0:
                    if np.sum(Z,axis=0)[j] == 0:
                        print("Feature Kick: Undesirable Behavior")
                    Z[i,j] == 1
                    prob_matrix[i,j] = 1
            else:
                #numerical adjustment
                yz_one = log_uncollapsed(Y[i,:],Z_one[i,:],W,sig)
                yz_zero = log_uncollapsed(Y[i,:],Z_zero[i,:],W,sig)
                yz_one = yz_one - yz_zero
                yz_zero = 0
                p_one = 0
                if math.isinf(np.exp(yz_one)):
                    p_one = 1
                else:
                    p_one = float(np.exp(yz_one)*zp_one)/(np.exp(yz_one)*zp_one + np.exp(yz_zero)*zp_zero)
                Z[i,j] = np.random.binomial(1,p_one)
                prob_matrix[i,j] = p_one
        #There is an indent here
        K = Z.shape[1]
    return (Z,prob_matrix)

def excise1(Z,vec,start,mark):
    N,F = Z.shape
    zp = 0.0
    for i in range(N):
        index = 0
        ignore = 0
        for j in range(F):
            if j < mark:
                if Z[i,j] != start[j]:
                    ignore = 1
                    break
            index = int(index + Z[i,j]*(2**(F-j-1)))
        if ignore == 0:
            zp = zp+np.log(float(vec[index])) 
    return zp

#computes log probability of the rows starting
#from start zero and ending with end one given
#probabilities in vec
#we should never return 0 because excise2 is
#only called when there is a meaningful paintbox
#update
#returns -1 when vec is invalid
def excise2(compact,vec,start_zero,end_one):
    zp = 0        
    for i in range(start_zero,end_one+1):
        if compact[i] != 0:
            if vec[i] == 0:
                zp = -1
                break
            else:
                zp = zp + compact[i]*np.log(float(vec[i]))
    return zp
    
def Z_compact(Z):
    N,F = Z.shape
    sort = np.zeros(N)
    for i in range(N):
        sort[i] = int(''.join(map(str, Z[i,:])).replace(".0",""),2)
    sort = np.sort(sort)
    compact = np.zeros(2**F)
    for i in range(N):
        compact[int(sort[i])] = compact[int(sort[i])] + 1 
    return compact
#Algorithm: Perform tree updates on vectorized paintbox
def sample_pb(Z,tree,res):
    bound = 2 #the exponent
    F,D = get_FD(tree)
    ctree,ptree = tree
    vec = get_vec(tree)
    compact = Z_compact(Z)
    #iterate over features (row of paintbox)
    start_pb = time.time()
    count = 0
    for i in range(F):
        #iterate over nodes j in tree layer i 
        for j in range(2**i):
            start_zero = j*2**(F-i)
            end_one = (j+1)*2**(F-i) - 1
            if np.sum(compact[start_zero:end_one+1]) == 0:
                continue
            count = count + 1
            end_zero = j*2**(F-i) + 2**(F-i-1) - 1
            #start_one =  j*2**(F-i) + 2**(F-i-1)
            start_one = end_zero + 1
            tot = np.sum(vec[start_zero:end_one+1])
            if tot == 0:
                continue
            else:
                binary = map(int,"{0:b}".format(int(start_zero)))
                #start = np.concatenate((np.zeros(F-len(binary)), binary))
                old_prob = float(np.sum(vec[start_one:end_one+1]))/tot 
                unit = float(np.sum(vec[start_zero:end_one+1]))/res
                log_roulette = []
                center = int(round(res*old_prob))
                if center == res:
                    lbound = res - 1
                    ubound = res
                elif center == 0:
                    lbound = 0
                    ubound = 1
                else:
                    lbound = center - 1
                    ubound = center + 1
                #lbound = 0
                #ubound = res
                mat_vec = np.tile(vec,(ubound-lbound+1,1))
                wheel = [w for w in range(ubound-lbound+1)]
                for k in range(lbound,ubound+1):
                    mat_pos = k - lbound
                    new_prob = float(k)/res
                    if old_prob != new_prob:
                        if old_prob == 0:
                            ratio_zero = float((1 - new_prob))/(1 - old_prob) 
                            mat_vec[mat_pos,start_zero:end_zero+1] = ratio_zero*mat_vec[mat_pos,start_zero:end_zero+1]                       
                            mat_vec[mat_pos,start_one] = unit*k
                        elif old_prob == 1:
                            ratio_one = float(new_prob)/old_prob
                            mat_vec[mat_pos,start_one:end_one+1] = ratio_one*mat_vec[mat_pos,start_one:end_one+1]
                            mat_vec[mat_pos,end_zero] = unit*(res-k)
                        else:        
                            ratio_one = float(new_prob)/old_prob
                            ratio_zero = float((1 - new_prob))/(1 - old_prob)
                            mat_vec[mat_pos,start_one:end_one+1] = ratio_one*mat_vec[mat_pos,start_one:end_one+1]
                            mat_vec[mat_pos,start_zero:end_zero+1] = ratio_zero*mat_vec[mat_pos,start_zero:end_zero+1]
                    #bottleneck line  
                    #val = excise(Z,mat_vec[mat_pos,:],start,i)
                    val = excise2(compact,mat_vec[mat_pos,:],start_zero,end_one)
                    if math.isinf(val) or math.isnan(val) or val == -1:
                        wheel.remove(mat_pos)
                    else:
                        log_roulette.append(val)
                if len(log_roulette) == 0:
                    print("paintbox update broken")
                    sys.exit()
                shift = max(log_roulette)
                roulette = [np.exp(lr - shift) for lr in log_roulette] 
                normal_roulette = [r/np.sum(roulette) for r in roulette]
                #Hacked Solution Beware
                try:
                    bucket = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
                    chosen = wheel[bucket]
                except TypeError:
                    #BEWARE, THIS CHANGES IF YOU ADJUST THE EXPONENT OF RES
                    chosen = 1
                    print("INVARIANT BROKEN")
                vec = mat_vec[chosen,:]
            ctree[i,j] = 0
            ctree[i,j] = float(chosen+lbound)/res
    end_pb = time.time()
    lapse = end_pb-start_pb
    tree = update((ctree,ptree))
    return tree,lapse

#find posterior mean of W 
def mean_w(Y,Z):
    N,K = Z.shape
    W = np.dot(np.linalg.inv(np.dot(Z.T,Z) + (sig/sig_w)**2*np.eye(K)), np.dot(Z.T,Y))
    return W
    
def sample_W(Y,Z,sig,sig_w):
    N,K = Z.shape
    T = Y.shape[1]
    W = np.zeros((K,T))
    ZTZ = np.dot(Z.T,Z)
    novera = float(sig)/sig_w
    I = np.eye(K)
    ZTY = np.dot(Z.T,Y)
    iZTZI = np.linalg.inv(ZTZ + novera**2*I)
    mean = np.dot(iZTZI,ZTY)
    cov = sig**2*iZTZI
    #Note, that the covariance is defined for each column is just depressing
    for col in range(T):
        try:
            W[:,col] = SPST.multivariate_normal.rvs(np.squeeze(np.asarray(mean[:,col])),cov)
        except (ValueError,IndexError):
            print('ValueError')
        
    return W

def draw_feature(Z,tree,res,ext):
    N,F = Z.shape
    Z_new = np.zeros((N,F+ext))
    for i in range(N):
        Z_new[i,:] = conditional_draw(tree,Z[i,:],ext,F+ext)
    return Z_new  

def drop_feature(Z,W,tree):
    zeros = np.where(np.sum(Z,axis=0) == 0)[0]
    ones = np.where(np.sum(Z,axis=0) != 0)[0]
    Z = Z[:,ones]
    W = W[ones,:]
    tree = drop_tree(tree,zeros)
    ctree,ptree = tree
    return (Z,W,tree)
    

def new_feature(Y,Z,W,tree,ext,K,res,sig,sig_w,drop,truncate):
    ctree,ptree = tree
    N,T = Y.shape
    #debugging invariant
    vec = get_vec(tree)
    Z,W,tree = drop_feature(Z,W,tree)
    F,D = get_FD(tree)
    if F >= truncate or drop:
        return (Z,W,tree)
    else:
        if F + ext < K:
            more = K - F
        else:
            more = ext
        tree = add(tree,more,res)
        Z = draw_feature(Z,tree,res,more)
        W = np.vstack((W,np.random.normal(0,sig_w,(more,T))))
    #W = sample_W(Y,Z,sig,sig_w)
    return (Z,W,tree)
    
def pred_ll_paintbox(held,W,tree,sig):
    #should you be comparing the predictive log likelihood?  I think you should
    R,T = held.shape
    K,T = W.shape
    log_pred = 0
    vec = get_vec(tree)
    for i in range(R):
        pred_row = 0
        for j in range(2**K):
            binary = list(map(int,"{0:b}".format(j)))
            pad_binary = [0]*(K-len(binary)) + binary
            log_z_post = np.log(Z_paintbox(pad_binary,vec))
            total_z = np.array(pad_binary)
            pred_row = pred_row + np.exp(log_data_zw(held[i,:],total_z,W,sig) + log_z_post)
        log_pred = log_pred + np.log(pred_row)
    return log_pred

def print_paintbox(tree,W,data_dim,flag='four'):
    small_x,small_y,big_x,big_y = data_dim
    vec = get_vec(tree)
    F,D = get_FD(tree)
    K = int(math.log(D,2))
    print("outputting paintbox")
    for j in range(D):
        binary = list(map(int,"{0:b}".format(j)))
        pad_binary = [0]*(K-len(binary)) + binary
        prob = Z_paintbox(pad_binary,vec)
        if prob > 0.01:
            pad_binary = np.array(pad_binary)
            reconstruct = np.dot(pad_binary,W)
            print("pad binary, reconstruct, probability")
            print(pad_binary)
            print(prob)
            display_W(reconstruct,data_dim,flag)
    return 0
      
def cgibbs_sample(Y,sig,sig_w,iterate,D,F,N,T):
    pb = pb_init(D,F)
    Z = draw_Z(pb,D,F,N,T)
        
    ll_list = []
    print("gibbs sample")
    for it in range(iterate):
        #sample Z
        Z = sample_Z(Y,Z,sig,sig_w,pb,D,F,N,T)
        #sample paintbox
        pb = sample_pb(Z,pb,D,F,N,T,res)        
        W = mean_w(Y,Z)
        vec = vectorize(pb)
        ll_list.append(log_data_zw(Y,Z,W,sig) + Z_vec(Z,vec,D) + log_w_sig(W,sig))
    
    return (ll_list,Z,W,pb)
    
def recover_paintbox(held,observe,W,tree,sig):
    N,half = observe.shape 
    R,T = held.shape
    K,T = W.shape
    log_recover = 0
    vec = get_vec(tree)
    for i in range(R):
        full_ll = 0
        observe_ll = 0
        for j in range(2**K):
            binary = list(map(int,"{0:b}".format(j)))
            pad_binary = [0]*(K-len(binary)) + binary
            log_z_post = np.log(Z_paintbox(pad_binary,vec))
            total_z = np.array(pad_binary)
            full_ll = full_ll + np.exp(log_data_zw(held[i,:],total_z,W,sig) + log_z_post)
            observe_ll = observe_ll + np.exp(log_data_zw(observe[i,:],total_z,W[:,:half],sig) + log_z_post)
        log_recover = log_recover + np.log(full_ll) - np.log(observe_ll)
    return log_recover
    
def upaintbox_sample(data_dim,log_res,hold,Y,held_out,ext,sig,sig_w,iterate,K,truncate):
    small_x,small_y,big_x,big_y = data_dim
    N,T = Y.shape
    #technically this is a bug
    #generating tree with res = 1 is wrong.  
    #but you fixed it in tree paintbox hard coded 0.5 
    res = 1
    tree = gen_tree(K,res)
    ctree,ptree = tree
    Z = draw_Z_tree(tree,N)
    #Z = np.loadtxt('assignments.txt')
    print(Z)
    #W = sample_W(Y,Z,sig,sig_w)
    W = np.reshape(np.random.normal(0,sig_w,K*T),(K,T))
    #W = np.loadtxt('features.txt')
#    full = generate_gg_blocks()
#    W = np.zeros((3,T))
#    W[0,:] = full[0,:]
#    W[1,:] = full[2,:]
#    W[2,:] = full[0,:] + full[2,:]
    display_W(W,'four')
    ll_list = [] 
    iter_time = [] 
    f_count = [] 
    lapse_data = [] 
    pred_ll = []
    pred = 0
    rec = 0
    for redo in range(2):
        if redo == 1:
            res = 1
            N,K = Z.shape
            tree = gen_tree(K,res)
            ctree,ptree = tree
        for it in range(iterate):
            if it%hold == 0:
                if res < 2**log_res:
                    res = res*2
            
            start = time.time()
            N,K = Z.shape
            #sample Z
            Z,prob_matrix = sample_Z(Y,Z,W,sig,sig_w,tree)
            if it%10 == 0:
                print("iteration: " + str(it))
                print("Sparsity: " + str(np.sum(Z,axis=0)))
                print('predictive log likelihood: ' + str(pred))
                print('recover log likelihood: ' + str(rec))
            #sample paintbox
            tree,lapse = sample_pb(Z,tree,res)
            #sample W        
            W = sample_W(Y,Z,sig,sig_w)
            #add new features
            ll_list.append(log_data_zw(Y,Z,W,sig))
            F,D = get_FD(tree)
            f_count.append(F)
            #predictive log likelihood
            if it%100 == 0:
                pred = pred_ll_paintbox(held_out, W, tree, sig)
                pred_ll.append(pred)
                rec = recover_paintbox(held_out,held_out[:,:T/2],W,tree,sig)
            if it%500 == 0 and it > 0:
                print_paintbox(tree,W,data_dim,'four')
            #if it%200 == 0 and it > 0:
            #    display_W(W,data_dim,'nine')
            #handling last iteration edge case
            drop = 0
            if it == iterate - 1:
                drop = 1
            Z,W,tree = new_feature(Y,Z,W,tree,ext,K,res,sig,sig_w,drop,truncate)
            end = time.time()
            iter_time.append(end - start)
            lapse_data.append(lapse)
        #iter_time = np.cumsum(iter_time)
    return (ll_list,iter_time,f_count,lapse_data,Z,W,prob_matrix,pred_ll,tree)

def plot(title,x_axis,y_axis,data_x,data_y):
    plt.plot(data_x,data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    #for now res is multiple of 2 because of pb_init (not fundamental problem )
    #res = 128 #all conditionals will be multiples of 1/res 
    log_res = 10 #log of res
    hold = 500 #hold resolution for # iterations
    feature_count = 4 #features
    #T = 36 #length of datapoint
    data_count = 200
    held_out = 50
    sig = 0.1 #noise
    sig_w = 0.2 #feature deviation #it was 0.3 for full operation
    #data_type = 'random'
    #full_data,Z_gen = generate_data(feature_count,data_count + held_out,T,sig,data_type)
    #Y = full_data[:data_count,:]
    #held_out = full_data[data_count:,:]
    iterate = 5000
    small_x = 3
    small_y = 3
    big_x = 3
    big_y = 3
    data_dim = (small_x,small_y,big_x,big_y)
    feature_count = big_x*big_y
    T = small_x*small_y*big_x*big_y
    data_type = 'corr'
    flag = 'nine'
    #full_data,Z_gen = generate_data(feature_count,data_count + held_out,T,sig,data_type)
    full_data,Z_gen = construct_data(small_x,small_y,big_x,big_y,data_count + held_out,sig,data_type,corr_value=2)
    Y = full_data[:data_count,:]
    held_out = full_data[data_count:,:]
    
    
    K = 1 #start with K features
    ext = 1 #draw one new feature per iteration
#    profile.run('ugibbs_sample(log_res,hold,Y,ext,sig,sig_w,iterate,K,valid)') 
    
    runs = 1
    ll_data = np.zeros((runs,iterate))
    ll_time = np.zeros((runs,iterate))
    feature = np.zeros((runs,iterate))
    lapse_data = np.zeros((runs,iterate))
    #pred_data = np.zeros((runs,iterate))
    valid = 0
    algorithm = 'paintbox'
    truncate = 12
    #algorithm = 'var'
    #algorithm = 'uncollapsed'
    #you'll come back to this
    while valid < runs:
        print("Trial Number: " + str(valid))
        if algorithm == 'paintbox': 
            #profile.run('upaintbox_sample(log_res,hold,Y,held_out,ext,sig_alg,sig_w,iterate,K,valid)')
            ll_list,iter_time,f_count,lapse,Z,W,prob_matrix,pred_ll,tree = upaintbox_sample(data_dim,log_res,hold,Y,held_out,ext,sig,sig_w,iterate,K,truncate)
        if algorithm == 'uncollapsed':
            Z,W,ll_set = ugibbs_sampler(Y,alpha,sig,sig_w,iterate,Z_gen)
        if algorithm == 'var':
            nu_set,phi_set,Phi_set,tau_set,pred_ll = run_vi(Y,held_out,alpha,sig_w,sig,iterate,feature_count)
        if len(ll_list) == 0:
            continue
        ll_data[valid,:] = ll_list
        ll_time[valid,:] = iter_time
        feature[valid,:] = f_count
        lapse_data[valid,:] = lapse
        #pred_data[valid,:] = pred_ll
        valid += 1
     
#    plot_feature = [num for num in range(1,int(np.max(feature[:,iterate-1])+1))]    
#    plot_lapse = [0 for i in range(len(plot_feature))]
#    all_sums = [0 for i in range(len(plot_feature))]
#    for i in range(len(plot_feature)):
#            for row in range(runs):
#                indices = [x for x in range(iterate) if feature[row,x] == i]
#                all_sums[i] = all_sums[i] + len(indices)
#                plot_lapse[i] = np.sum(lapse_data[row,indices])
#            if plot_lapse[i] != 0:
#                plot_lapse[i] = float(plot_lapse[i])/all_sums[i]
#    plt.plot(plot_feature, plot_lapse, 'ro')
#    plt.show()
    
    np.savetxt("log_likelihood.csv", ll_data, delimiter=",")
    np.savetxt("time.csv", ll_time, delimiter=",")
    np.savetxt("feature.csv", feature, delimiter=",")
    np.savetxt("lapse.csv", lapse_data, delimiter=",")
    
    ll_avg = 1./runs*np.sum(ll_data,axis=0)
    time_avg = 1./runs*np.sum(ll_time,axis=0)
    f_avg = 1./runs*np.sum(feature,axis=0)
    
    title = 'Nonparametric Paintbox: log likelihood vs. iterations'
    x_axis = 'iterations'
    y_axis = 'log likelihood'
    data_x = [i for i in range(1,iterate+1)]
    data_y = ll_avg
    plot(title,x_axis,y_axis,data_x,data_y)
    
    title = 'Nonparametric Paintbox: log likelihood vs. time'
    x_axis = 'time'
    y_axis = 'log likelihood'
    data_x = time_avg
    data_y = ll_avg
    plot(title,x_axis,y_axis,data_x,data_y)
    
    title = 'Nonparametric Paintbox: feature vs. iterations'
    x_axis = 'iterations'
    y_axis = 'log likelihood'
    data_x = [i for i in range(1,iterate+1)]
    data_y = f_avg
    plot(title,x_axis,y_axis,data_x,data_y)
    
    title = 'Nonparametric Paintbox: feature vs. time'
    x_axis = 'time'
    y_axis = 'features'
    data_x = time_avg
    data_y = f_avg
    plot(title,x_axis,y_axis,data_x,data_y)
    
    #visual verification of data reconstruction
    approx = np.dot(Z,W)
    for i in range(10):
        print("sample: " + str(i))
        print("features probability")
        print(prob_matrix[i,:])
        print("features selected")
        print(Z[i,:])
        display_W(approx[i:i+1,:],data_dim,flag)
        print("data: " + str(i))
        display_W(Y[i:i+1,:],data_dim,flag)
    
    #printing paintbox (only makes sense for single run)
    print_paintbox(tree,W,data_dim,flag)    
    #display_W(W,small_x,small_y,big_x,big_y,'nine')    
        
        
#    pb_scale = scale(pb)
#    plt.imshow(pb_scale,interpolation='nearest')
#    plt.show()   
    #print(ll_list)
    
    #plt.imshow(W,interpolation='nearest')
    #plt.show()
    #display_W(W)
    
    #plt.plot(pf1,pl1,'bs',pf2,pl2,'ro')
    #plt.title("Paintbox Sampling Time vs. Features")
    #plt.xlabel("Active Features")
    #plt.ylabel("time")
    #plt.show()
    
    #pylab.plot(f1,l1,'bs')
    #pylab.plot(f2,l2,'ro')
    #pylab.legend(loc='upper left')
    #pylab.title("Paintbox Sampling Time vs. Features")
    #pylab.xlabel("Active Features")
    #pylab.ylabel("time")
    #plt.show()
    
#    pylab.plot(time_avg2,ll_avg2,label="nonadaptive")
#    pylab.plot(time_avg1,ll_avg1,label="adaptive")
#    pylab.title("Nonparametric Paintbox: Log Likelihood vs. Time")
#    pylab.xlabel("Time")
#    pylab.ylabel("Log Likelihood")
#    pylab.legend(loc='upper left')