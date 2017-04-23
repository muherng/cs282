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
from tree_paintbox import gen_tree,update,drop,add,get_vec

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
def Z_vec(Z,vec,D):
    N,F = Z.shape
    zp = 1.0
    zp_list = []
    for i in range(N):
        z_index = int(''.join(map(str, Z[i,:])).replace(".0",""),2)
        zp = zp*float(vec[z_index])
        zp_list.append(zp)
        #if float(vec[z_index])/D == 0.0:
            #print("Z Invariant Broken")
            #print("Breaks at Index:")
            #print(z_index)
        #if zp == 0:
        #    print("Did Invariant Break?")
        
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
def log_uncollapsed( Y , Z , W , sig):
    N,T = Y.shape
    YZW = Y - np.dot(Z,W)
    trYZW = np.trace(np.dot(YZW.T,YZW))
    ll = (-1.0*float(N*T)/2) * np.log(2*np.pi*sig**2) - 1.0/(2*sig**2) * trYZW
    return ll
    
    
#adjust this to accomodate W
def sample_Z(Y,Z,W,sig,sig_w,tree,D,F,N,T):
    vec = get_vec(tree)
    for i in range(N):
        for j in range(F):
            Z_one = np.copy(Z)
            Z_zero = np.copy(Z)
            Z_one[i,j] = 1
            Z_zero[i,j] = 0
            yz_one = log_uncollapsed(Y,Z_one,W,sig)
            yz_zero = log_uncollapsed(Y,Z_zero,W,sig)
            zp_one = Z_paintbox(Z_one[i,:],vec,sig,D)
            zp_zero = Z_paintbox(Z_zero[i,:],vec,sig,D)
            if zp_one == 0 or zp_zero == 0:
                if zp_one == 0:
                    Z[i,j] == 0
                if zp_zero == 0:
                    Z[i,j] == 1
            else:
                #numerical adjustment
                yz_one = yz_one - yz_zero
                yz_zero = 0
                p_one = 0
                if math.isinf(np.exp(yz_one)):
                    p_one = 1
                else:
                    p_one = float(np.exp(yz_one)*zp_one)/(np.exp(yz_one)*zp_one + np.exp(yz_zero)*zp_zero)
                if math.isnan(p_one):
                    print("P IS NAN")
                Z[i,j] = np.random.binomial(1,p_one)
            if Z_vec(Z,vec,D) == 0:
                print("TOTALLY ILLEGAL")
            
    return Z

#Algorithm: Perform tree updates on vectorized paintbox
def sample_pb(Z,tree,D,F,N,T,res):
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
                    normal_roulette = [r/np.sum(roulette) for r in roulette]
                try:
                    chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
                except TypeError:
                    print("TypeError")
                vec = mat_vec[chosen,:]
                ctree[i,j] = float(chosen)/res
                if Z_vec(Z,vec,D) == 0:
                    print("ILLEGAL PAINTBOX UPDATE")
    tree = update((ctree,ptree))
#    for i in range(len(vec)):
#        if vec[i] != ptree[F-1,i]:
#            print("UPDATE PROBAILITY INVARIANT")
#            print(vec[i])
#            print(ptree[F-1,i])
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
      
def cgibbs_sample(Y,sig,sig_w,iterate,D,F,N,T):
    pb = pb_init(D,F)
    Z = draw_Z(pb,D,F,N,T)
    if Z_vec(Z,vectorize(pb),D) == 0:
        print("draw Z is wrong")
        
    ll_list = []
    print("gibbs sample")
    for it in range(iterate):
        print(it)
        #sample Z
        print("SAMPLE Z")
        Z = sample_Z(Y,Z,sig,sig_w,pb,D,F,N,T)
        #sample paintbox
        print("SAMPLE PB")
        pb = sample_pb(Z,pb,D,F,N,T,res)        
        W = mean_w(Y,Z)
        vec = vectorize(pb)
        ll_list.append(log_data_zw(Y,Z,W,sig) + np.log(Z_vec(Z,vec,D)) + log_w_sig(W,sig))
    
    return (ll_list,Z,W,pb)
    
def ugibbs_sample(Y,sig,sig_w,iterate,D,F,N,T):
    #pb = pb_init(D,F)
    tree = gen_tree(F,res)
    print("Tree")
    ctree,ptree = tree
    print(ctree)
    print(ptree)
    Z = draw_Z_tree(tree,N)
    print("LEGAL?")
    print(Z_vec(Z,get_vec(tree),D))
    W = sample_W(Y,Z,sig,sig_w)
    #W = np.reshape(np.random.normal(0,sig_w,F*T),(F,T))
    ll_list = []
    for it in range(iterate):
        print("iteration: " + str(it))
        #sample Z
        Z = sample_Z(Y,Z,W,sig,sig_w,tree,D,F,N,T)
        #sample paintbox
        tree = sample_pb(Z,tree,D,F,N,T,res)
        #sample W        
        W = sample_W(Y,Z,sig,sig_w)
        #add new features
        
        vec = get_vec(tree)
        ll_list.append(log_data_zw(Y,Z,W,sig) + np.log(Z_vec(Z,vec,D)) + log_w_sig(W,sig))
    return (ll_list,Z,W,devectorize(vec))

if __name__ == "__main__":
    #for now res is multiple of 2 because of pb_init (not fundamental problem )
    res = 16 #all conditionals will be multiples of 1/res 
    F = 4 #features
    D = res**F #discretization
    T = 36 #length of datapoint
    N = 100 #data size
    sig = 0.1
    sig_w = 5.0
    print("GENERATE DATA")
    Y,Z_gen = generate_data(res,D,F,N,T,sig)
    print('FINISH GENERATE')
    #gen_pb = scale(gen_pb)
    #plt.imshow(gen_pb,interpolation='nearest')
    #plt.show()
    iterate = 100
    #print("DATA")
    #print(Y)
    #init variables
    #profile.run('gibbs_sample(Y,sig,iterate,D,F,N,T)') 
    ll_list,Z,W,pb = ugibbs_sample(Y,sig,sig_w,iterate,D,F,N,T)
    #approx = np.dot(Z,W)
    #print(Z)
    #pb_scale = scale(pb)
    #plt.imshow(pb_scale,interpolation='nearest')
    #plt.show()   
    #print(ll_list)
    
    #plt.imshow(W,interpolation='nearest')
    #plt.show()
    display_W(W)
    
    plt.plot(ll_list)
    plt.show()
