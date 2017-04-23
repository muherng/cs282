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

#generate random tree with depth F
def gen_tree(F,res):
    ctree = np.zeros((F,2**(F-1)))
    ptree = np.zeros((F,2**F))
    for i in range(F):
        for j in range(2**i):
            ctree[i,j] = float(int(np.where(np.random.multinomial(1,[1./(res+1)]*(res+1)) == 1)[0]))/res
    tree = update((ctree,ptree))
    return tree

def update(tree):
    ctree,ptree = tree
    F,D = ctree.shape
    ptree[0][0] = 1 - ctree[0][0]
    ptree[0][1] = ctree[0][0]
    for i in range(1,F):
        for j in range(2**(i+1)):
            if j%2 == 0:
                ptree[i,j] = ptree[i-1,j/2]*(1 - ctree[i,j/2])
            else: 
                ptree[i,j] = ptree[i-1,j/2]*ctree[i,j/2]
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
def drop(tree):
    ctree,ptree = tree
    F,D = ctree.shape
    ctree = ctree[:F-1,:D/2]
    ptree = ptree[:F-1,:D/2]
    return (ctree,ptree)
    
    
# add a feature to the end
def add(tree,res):
    ctree,ptree = tree
    F,D = ctree.shape
    new_ctree = np.zeros((F+1,2**(F+1)))
    new_ctree[:F,:D] = ctree
    new_ptree = np.zeros((F+1,2**(F+1)))
    for j in range(2**(F+1)):
        new_ctree[F+1,j] = int(np.where(np.random.multinomial(1,[1./res]*res) == 1)[0])
    return update((new_ctree,new_ptree))
    
def get_vec(tree):
    ctree,ptree = tree
    F,D = ptree.shape
    return ptree[F-1,:]

def get_FD(tree):
    ctree,ptree = tree
    return ptree.shape 
    

    
    