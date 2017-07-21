#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:33:02 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 
from fractions import gcd
import matplotlib.pyplot as plt
import sys

#generate random tree with depth F
def gen_tree(F,res):
    ctree = np.zeros((F,2**(F-1)))
    ptree = np.zeros((F,2**F))
    for i in range(F):
        for j in range(2**i):
            #ctree[i,j] = float(int(np.where(np.random.multinomial(1,[1./(res+1)]*(res+1)) == 1)[0]))/res
            ctree[i,j] = 0.5
    tree = update((ctree,ptree))
    return tree

def update(tree):
    ctree,ptree = tree
    F,D = ctree.shape
    if F == 0:
        ptree = np.zeros((0,0))
    else:
        ptree = np.zeros((F,2**F))
        ptree[0][0] = 1 - ctree[0][0]
        ptree[0][1] = ctree[0][0]
        for i in range(1,F):
            for j in range(2**(i+1)):
                if j%2 == 0:
                    ptree[i,j] = ptree[i-1,int(j/2)]*(1 - ctree[i,int(j/2)])
                else: 
                    ptree[i,j] = ptree[i-1,int(j/2)]*ctree[i,int(j/2)]
    return (ctree,ptree)
    
#feature is a vector of features to be dropped
#for now assume that we never drop more than one feature
# and all dropped features are final features (Bold)
#def drop_feature(tree,feature):
#    feature = np.sort(feature)
#    feature = feature[::-1]
#    ctree,ptree = tree
#    F,D = ctree.shape
#    for f in feature:
#        ctree = drop(ctree,f)
#    ptree = np.zeros(ctree.shape)
#    return update((ctree,ptree))
            
#drop last feature
def drop_tree(tree,zeros):
    ctree,ptree = tree
    zeros = zeros[::-1]
    for z in zeros:
        F,D = ctree.shape
        if F == 1:
            ctree = np.zeros((0,0))
        else:
            copy = np.copy(ctree)
            for i in range(z,F-1): 
                for j in range(2**z):
                    ctree[i,j*2**(i-z):(j+1)*2**(i-z)] = copy[i+1,j*2**(i-z+1):j*2**(i-z+1)+2**(i-z)] 
            ctree = ctree[:F-1,:2**(F-2)]
        
    tree = update((ctree,ptree))
    ctree,ptree = tree
    return tree

#draw from paintbox conditioned on row    
def conditional_draw(tree,row,ext,tot):
    vec = get_vec(tree)
    if len(row) == 0:
        z_index = 0
    else:
        try:
            z_index = int(''.join(map(str, row)).replace(".0",""),2)*(2**ext)
        except ValueError:
            print("ValueError")
            
    roulette = vec[z_index:z_index + 2**ext]
    normal_roulette = [float(r)/np.sum(roulette) for r in roulette]
                       
    try:
        chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0]) + z_index
    except TypeError:
        print('vec')
        print(vec)
        print('roulette')
        print(roulette)
        print('z_index')
        print(z_index)
        roulette = vec
        normal_roulette = [float(r)/np.sum(roulette) for r in roulette]
        chosen = int(np.where(np.random.multinomial(1,normal_roulette) == 1)[0])
        print("TypeError") 
        print("Conditional Error")
        sys.exit()
    binary = list(map(int,"{0:b}".format(chosen)))
    pad_binary = np.concatenate((np.zeros(tot - len(binary)),binary))
    return pad_binary
    
# add a feature to the end
def add(tree,ext,res):
    ctree,ptree = tree
    F,D = ctree.shape
    new_ctree = np.zeros((F+ext,2**(F+ext-1)))
    new_ctree[:F,:D] = ctree
    new_ptree = np.zeros((F+ext,2**(F+ext)))
    for i in range(F,F+ext):
        for j in range(2**i):
            #res/2
            new_ctree[i,j] = 0.5
            #new_ctree[i,j] = float(np.where(np.random.multinomial(1,[1./res]*res) == 1)[0])/res
    return update((new_ctree,new_ptree))
    
def get_vec(tree):
    ctree,ptree = tree
    F,D = ptree.shape
    return ptree[F-1,:]

def get_FD(tree):
    ctree,ptree = tree
    return ptree.shape 

def access(ctree,z_row):
    depth = len(z_row)
    z_index = int(''.join(map(str, z_row)).replace(".0",""),2)
    return ctree[depth,z_index]
     
    
    

    
    