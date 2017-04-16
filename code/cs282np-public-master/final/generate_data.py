#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:39:10 2017

@author: morrisyau
"""

import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 

print("Hello")


#size of paintbox
f = 2 #features
d = 2 #discretization

#data size 

#paintbox 
pb = np.zeros(f,d)

#initalize features
W = np.eye(f)

#noise parameter
sig = 0.1

#generate data

