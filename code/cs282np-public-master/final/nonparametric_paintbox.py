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
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree 
import profile
from fractions import gcd
from scipy.stats import norm
import math 
from tree_paintbox import gen_tree,update,add,get_vec,access,get_FD,drop_tree,conditional_draw
import time
#There are a variety of edge case considerations when it comes to vectorized 
#form of the paintboxes, this leads me to believe trees are the right idea
#it turns out you can query trees very quickly 
#update for trees basically same speed, and much easier to understand
#Next time write down a pro-con functionality list and think it through
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
    for i in range(N):
        index = 0
        for j in range(F):
            index = int(index + Z[i,j]*(2**(F-j-1)))
        #if vec[index] == 0:
            #print("Z given Vector Invariant Break")
        zp = zp+np.log(float(vec[index])) 
    return zp
    
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
def Z_paintbox(z_row,vec,sig,D):
    z_index = int(''.join(map(str, z_row)).replace(".0",""),2)
    zp = float(vec[z_index])
    return zp
    
def log_data_zw(Y,Z,W,sig):
    delta = Y - np.dot(Z,W)
    delta_sum = np.trace(np.dot(delta.T,delta))
    ll =  -1./(2*sig**2) * delta_sum
    return ll
    
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
    vec = get_vec(tree)
    for i in range(N):
        for j in range(K):
            Z_one = np.copy(Z)
            Z_zero = np.copy(Z)
            Z_one[i,j] = 1
            Z_zero[i,j] = 0
            zp_one = Z_paintbox(Z_one[i,:],vec,sig,D)
            zp_zero = Z_paintbox(Z_zero[i,:],vec,sig,D)
            if zp_one == 0 or zp_zero == 0:
                if zp_one == 0:
                    Z[i,j] == 0
                if zp_zero == 0:
                    if np.sum(Z,axis=0)[j] == 0:
                        print("Feature Kick: Undesirable Behavior")
                    Z[i,j] == 1
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
                #if math.isnan(p_one):
                #    print("P IS NAN")
                Z[i,j] = np.random.binomial(1,p_one)
            #if math.isinf(Z_vec(Z,vec)):
            #    print("TOTALLY ILLEGAL") 
        K = Z.shape[1]
    return Z

def excise(Z,vec,start,mark):
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
    
#Algorithm: Perform tree updates on vectorized paintbox
def sample_pb(Z,tree,res):
    F,D = get_FD(tree)
    ctree,ptree = tree
    vec = get_vec(tree)
    #iterate over features (row of paintbox)
    for i in range(F):
        #iterate over nodes j in tree layer i 
        for j in range(2**i):
            mat_vec = np.tile(vec,(res+1,1))
            start_zero = j*2**(F-i)
            end_zero = j*2**(F-i) + 2**(F-i-1) - 1
            start_one =  j*2**(F-i) + 2**(F-i-1)
            end_one = (j+1)*2**(F-i) - 1
            tot = np.sum(vec[start_zero:end_one+1])
            binary = map(int,"{0:b}".format(int(start_zero)))
            start = np.concatenate((np.zeros(F-len(binary)), binary))
            if tot == 0:
                continue
            else:
                old_prob = float(np.sum(vec[start_one:end_one+1]))/tot 
                unit = float(np.sum(vec[start_zero:end_one+1]))/res
                roulette = []
                for k in range(res+1):
                    new_prob = float(k)/res
                    if old_prob != new_prob:
                        if old_prob == 0:
                            ratio_zero = float((1 - new_prob))/(1 - old_prob) 
                            mat_vec[k,start_zero:end_zero+1] = ratio_zero*mat_vec[k,start_zero:end_zero+1]                       
                            mat_vec[k,start_one] = unit*k
                        elif old_prob == 1:
                            ratio_one = float(new_prob)/old_prob
                            mat_vec[k,start_one:end_one+1] = ratio_one*mat_vec[k,start_one:end_one+1]
                            mat_vec[k,end_zero] = unit*(res-k)
                        else:        
                            ratio_one = float(new_prob)/old_prob
                            ratio_zero = float((1 - new_prob))/(1 - old_prob)
                            mat_vec[k,start_one:end_one+1] = ratio_one*mat_vec[k,start_one:end_one+1]
                            mat_vec[k,start_zero:end_zero+1] = ratio_zero*mat_vec[k,start_zero:end_zero+1]
                    #bottleneck line  
                    val = excise(Z,mat_vec[k,:],start,i)
                    #val = Z_vec(Z,mat_vec[k,:])
                    if math.isinf(val) or math.isnan(val) or val == 0:
                        roulette.append(0.0)
                    else:
                        roulette.append(np.exp(val))
                if np.sum(roulette) == 0:
                    roulette = 1./res * np.ones(res+1)
                normal_roulette = [r/np.sum(roulette) for r in roulette]
                #Hacked Solution Beware
                try:
                    chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
                except TypeError:
                    chosen = int(round(res*old_prob))
                    #print("INVARIANT BROKEN")
                #print("Before Paintbox Update")
                #run_line = Z_vec(Z,vec)
                vec = mat_vec[chosen,:]
                ctree[i,j] = float(chosen)/res
                #print(roulette)
                #print("After Paintbox Update")
                #run_line = Z_vec(Z,vec)
                #print("ILLEGAL PAINTBOX UPDATE")
    tree = update((ctree,ptree))
    return tree

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
    

def new_feature(Y,Z,W,tree,ext,K,res,sig,sig_w):
    ctree,ptree = tree
    Z,W,tree = drop_feature(Z,W,tree)
    F,D = get_FD(tree)
    if F > 8:
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
    
def ugibbs_sample(Y,ext,sig,sig_w,iterate,K,data_run):
    print("Trial Number: " + str(data_run))
    N,T = Y.shape
    tree = gen_tree(K,res)
    ctree,ptree = tree
    Z = draw_Z_tree(tree,N)
    #W = sample_W(Y,Z,sig,sig_w)
    W = np.reshape(np.random.normal(0,sig_w,K*T),(K,T))
    ll_list = [] 
    iter_time = [] 
    f_count = []  
    for it in range(iterate):
        
        start = time.time()
        N,K = Z.shape
        #sample Z
        Z = sample_Z(Y,Z,W,sig,sig_w,tree)
        if it%10 == 0:
            print("iteration: " + str(it))
            print("Sparsity: " + str(np.sum(Z,axis=0)))
        #sample paintbox
        tree = sample_pb(Z,tree,res)
        #ctree,ptree = tree
        #sample W        
        W = sample_W(Y,Z,sig,sig_w)
        #add new features
        vec = get_vec(tree)
        #ll_list.append(log_data_zw(Y,Z,W,sig) + Z_vec(Z,vec) + log_w_sig(W,sig))
        ll_list.append(log_data_zw(Y,Z,W,sig))
        F,D = get_FD(tree)
        f_count.append(F)
        Z,W,tree = new_feature(Y,Z,W,tree,ext,K,res,sig,sig_w)
        end = time.time()
        iter_time.append(end - start)
    iter_time = np.cumsum(iter_time)
    return (ll_list,iter_time,f_count,Z,W)

def plot(title,x_axis,y_axis,data_x,data_y):
    plt.plot(data_x,data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    #for now res is multiple of 2 because of pb_init (not fundamental problem )
    res = 4 #all conditionals will be multiples of 1/res 
    F = 4 #features
    D = res**F #discretization
    T = 36 #length of datapoint
    N = 100 #data size
    sig = 0.1
    sig_w = 0.7
    Y,Z_gen = generate_data(F,N,T,sig)
    iterate = 1000
    #active features
    K = 1
    ext = 1
    #profile.run() 
    runs = 10
    ll_data = np.zeros((runs,iterate))
    ll_time = np.zeros((runs,iterate))
    feature = np.zeros((runs,iterate))
    valid = 0
    while valid < runs:
        ll_list,iter_time,f_count,Z,W = ugibbs_sample(Y,ext,sig,sig_w,iterate,K,valid)
        if len(ll_list) == 0:
            continue
        ll_data[valid,:] = ll_list
        ll_time[valid,:] = iter_time
        feature[valid,:] = f_count
        valid += 1
    
    np.savetxt("log_likelihood.csv", ll_data, delimiter=",")
    np.savetxt("time.csv", ll_time, delimiter=",")
    np.savetxt("feature.csv", feature, delimiter=",")
    
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
    
    #pb_scale = scale(pb)
    #plt.imshow(pb_scale,interpolation='nearest')
    #plt.show()   
    #print(ll_list)
    
    #plt.imshow(W,interpolation='nearest')
    #plt.show()
    #display_W(W)
