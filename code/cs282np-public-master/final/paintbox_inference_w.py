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
from generate_data import generate_data,pb_init,draw_Z,scale 
import profile
from fractions import gcd

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

#probability of Z given the vectorized paintbox 
#henceforth we will be working over vectorized paintboxes.
#we unvectorize only for visual debugging.  
def Z_vec(Z,vec,D):
    N,F = Z.shape
    zp = 1
    for i in range(N):
        z_index = int(''.join(map(str, Z[i,:])).replace(".0",""),2)
        zp = zp*float(vec[z_index])/D
    return zp

#performs Z_vec for a single row
def Z_paintbox(z_row,vec,sig,D):
    z_index = int(''.join(map(str, z_row)).replace(".0",""),2)
    zp = float(vec[z_index])/D
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
    #ll = np.exp(tmp)
    try:
        ll = tmp - (float(N*D)/2*np.log(2*np.pi)+(N-K)*D*np.log(sig)+K*D*np.log(sig_w)+float(D)/2*np.log(np.linalg.det(np.dot(Z.T,Z)+(float(sig)/sig_w)**2*np.eye(K))))
    except np.linalg.linalg.LinAlgError:
        print('LinAlgError')
    
    return ll 
    
    
#adjust this to accomodate W
def sample_Z(Y,Z,sig,sig_w,pb,D,F,N,T):
    vec = vectorize(pb)
    for i in range(N):
        for j in range(F):
            Z_one = np.copy(Z)
            Z_zero = np.copy(Z)
            Z_one[i,j] = 1
            Z_zero[i,j] = 0
            yz_one = log_collapsed(Y,Z_one,sig,sig_w)
            yz_zero = log_collapsed(Y,Z_zero,sig,sig_w)
            zp_one = Z_paintbox(Z_one[i,:],vec,sig,D)
            zp_zero = Z_paintbox(Z_zero[i,:],vec,sig,D)
            #numerical adjustment
            yz_one = yz_one - yz_zero
            yz_zero = 0
            p_one = float(np.exp(yz_one)*zp_one)/(np.exp(yz_one)*zp_one + np.exp(yz_zero)*zp_zero)
            try:
                Z[i,j] = np.random.binomial(1,p_one)
            except ValueError:
                print("ValueError Forever")
    return Z

#Algorithm: Perform tree updates on vectorized paintbox
def sample_pb(Y,Z,pb,D,F,N,T,res):
    vec = vectorize(pb)
    updates = 2**F-1
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
                    roulette.append(Z_vec(Z,mat_vec[k,:],D))
                    normal_roulette = [1.0/np.sum(roulette) * r for r in roulette]
                chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
                vec = mat_vec[chosen,:]
    pb = devectorize(vec)
    return pb

#find posterior mean of W 
def mean_w(Y,Z):
    N,K = Z.shape
    W = np.dot(np.linalg.inv(np.dot(Z.T,Z) + (sig/sig_w)**2*np.eye(K)), np.dot(Z.T,Y))
    return W
      
def gibbs_sample(Y,sig,sig_w,iterate,D,F,N,T):
    W = np.eye(F)
    pb = pb_init(D,F)
    Z = draw_Z(pb,D,F,N,T)
    ll_list = []
    W_list = []
    for it in range(iterate):
        #sample Z
        print("SAMPLE Z")
        Z = sample_Z(Y,Z,sig,sig_w,pb,D,F,N,T)
        #sample paintbox
        print("SAMPLE PB")
        pb = sample_pb(Y,Z,pb,D,F,N,T,res)        
        W = mean_w(Y,Z)
        vec = vectorize(pb)
        ll_list.append(log_data_zw(Y,Z,W,sig) + np.log(Z_vec(Z,vec,D)))
    
    return (ll_list,Z,W,pb)
    

if __name__ == "__main__":
    #for now res is multiple of 2 because of pb_init (not fundamental problem )
    res = 16 #all conditionals will be multiples of 1/res 
    F = 2 #features
    D = res**F #discretization
    T = F #length of datapoint
    N = 100 #data size
    sig = 0.1
    sig_w = 1.0
    Y,gen_pb = generate_data(res,D,F,N,T,sig)
    gen_pb = scale(gen_pb)
    plt.imshow(gen_pb,interpolation='nearest')
    plt.show()
    iterate = 100
    #print("DATA")
    #print(Y)
    #init variables
    #profile.run('gibbs_sample(Y,sig,iterate,D,F,N,T)') 
    ll_list,Z,W,pb = gibbs_sample(Y,sig,sig_w,iterate,D,F,N,T)
    approx = np.dot(Z,W)
    #print(Z)
    pb_scale = scale(pb)
    plt.imshow(pb_scale,interpolation='nearest')
    plt.show()   
    print(ll_list)
