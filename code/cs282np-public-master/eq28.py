#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.special as sps
import scipy.stats as SPST

def sample(m , alpha, N,  iterations):

    K_dagger = len(m)
    pts = [0 for i in range(iterations)]
    x = 0
    for j  in range(iterations):
        x +=  float(m[K_dagger-1]) / iterations
        tmp = 0
        for i in range(N):
            tmp = tmp + 1/float(i+1) * (1-x)**i
    
        tmp = np.exp(alpha *tmp)
        tmp = tmp * (1-x)**(alpha-1) * (1-x)**N
        pts[j] = x


    summer = [0 for i in range(iterations)]
    area = [0 for i in range(iterations)]
    
    summer[0] = pts[0]
    for j in range(0,iterations):
        summer[j] = summer[j-1] + pts[j]
    for j in range(0,iterations-1):
        area[j] = (summer[j+1] - summer[j])* float(m[K_dagger-1])/ iterations

    soum = sum(area)
    for i in range(iterations):
        area[i] = area[i] / float(soum)    
    num_new, = np.where(np.random.multinomial(1,area) == 1)[0]
    
    #print num_new
    #print pts[num_new]
    return pts[num_new]

def eq28(mu , Z,  alpha, K_dagger, N, iterations):

    sumcolumns = [ 0 for x in range(K_dagger)]
    for k in range(0,Z.shape[1]):
        for i in range(Z.shape[0]):
            try:
                sumcolumns[k] += Z[i,k]
            except IndexError:
                print('IndexError')


    toreturn = [ 0 for x in range(K_dagger)]
    for k in range(0,K_dagger-1):
        toreturn[k] = sampler(mu, sumcolumns, alpha, k, K_dagger, N,  iterations)

    toreturn[K_dagger-1] = sample(mu,alpha, N , iterations)
    
    if toreturn is None:
        print('NONE ERROR')
        
    print toreturn
    
    
def sampler(m, sumcolumns, alpha, k, K_dagger, N, iterations ):

    if k == K_dagger -1 :
        return
    if k == 0:
        UP = 1
        LOW = m[1]

    if k >0 and k < K_dagger-1:
        LOW = m[k+1]
        UP = m[k-1]

    pts = [0 for i in range(iterations)]
    x = LOW/float(iterations)
    for j  in range(iterations):
        x +=  float(UP) / iterations
        if x <= UP and x >= LOW:
            pts[j] = x**( sumcolumns[k]-1) * (1- x)**(N - sumcolumns[k])
    


    summer = [0 for i in range(iterations)]
    area = [0 for i in range(iterations)]
    
    summer[0] = pts[0]
    for j in range(0,iterations):
        summer[j] = summer[j-1] + pts[j]
    for j in range(0,iterations-1):
        area[j] = (summer[j+1] - summer[j])* float(UP)/ iterations
    
    soum = sum(area)
    for i in range(iterations):
        area[i] = area[i] / float(soum)    
    num_new, = np.where(np.random.multinomial(1,area) == 1)[0]
    
    #print num_new
    #print pts[num_new]
    return pts[num_new]


if __name__ == "__main__":

    m = [0.1,0.09,0.08,0.07]
    Z = [[0 for x in range(4)] for y in range(4)]
    for i in range(4):
        for j in range(4):
            Z[i][j] = (i+j) %2     
    eq28(m, Z , 3, 4, 13, 100)


    
    


    
 
