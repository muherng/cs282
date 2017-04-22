#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:33:02 2017

@author: morrisyau
"""

#generate random tree with depth F
def gen_tree(F,res):
    ctree = np.zeros((F,2**F))
    ptree = np.zeros((F,2**F))
    for i in range(F):
        for j in range(2**i):
            ctree[i,j] = int(np.where(np.random.multinomial(1,[1./res]*res) == 1)[0])
    tree = update((ctree,ptree))
    return tree

def update(tree):
    ctree,ptree = tree
    ptree[0,0] = ctree[0,0]
    for i in range(1,F):
        for j in range(2**i):
            if j%2 == 0:
                ptree[i,j] = ptree[i-1,j/2]*ctree[i,j]
            else: 
                ptree[i,j] = ptree[i-1,j/2+1]*ctree[i,j]
    return (ctree,ptree)
    
#feature is a vector of features to be dropped
def drop_feature(tree,feature):
    feature = np.sort(feature)
    feature = feature[::-1]
    ctree,ptree = tree
    F,D = ctree.shape
    for f in feature:
        ctree = drop(ctree,f)
    ptree = np.zeros(ctree.shape)
    return update((ctree,ptree))
            
#drop a single feature
def drop(ctree,f):
    copy_ctree = np.copy(ctree)
    for i in range(f,F):
        for j in range(2**i): 
            ctree[i,j] = copy_ctree[i,j]*copy_ctree[i+1,2*j+1] + (1-copy_ctree[i,j])*copy_ctree[i+1,2*j] 
    
# add a feature to the end
def add_feature(tree,res):
    ctree,ptree = tree
    F,D = ctree.shape
    new_ctree = np.zeros((F+1,2**(F+1)))
    new_ctree[:F,:D] = ctree
    for j in range(2**(F+1)):
        new_ctree[F+1,j] = int(np.where(np.random.multinomial(1,[1./res]*res) == 1)[0])
    return update((new_ctree,ptree))
    
def get_vec(tree):
    ctree,ptree = tree
    F,D = ptree.shape
    return ptree[F,:]
    

    
    