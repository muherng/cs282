import numpy as NP
import numpy.random as NR
import scipy.stats as SPST
import scipy.special as SPS
import matplotlib.pyplot as plt
from pylab import *

def weakLimit(alpha,N,K):
    Z = NP.zeros((N,K))
    u = SPST.beta.rvs(float(alpha)/K,1,size=K)
    for j in range(K):
        Z[:,j] = SPST.bernoulli.rvs(u[j],size=N)
    
    return Z

def stickBreak(alpha,N,K):
    Z = NP.zeros((N,K))
    u = [SPST.beta.rvs(alpha,1)]
    for i in range(1,K):
        u.append(u[i-1]*SPST.beta.rvs(alpha,1))
    
    for j in range(K):
         Z[:,j] = SPST.bernoulli.rvs(u[j],size=N)
    return Z

#note that I chanced this name, 
def sample_Z_restaurant(N, alpha):
    Z = NP.ones((0,0))
    for i in range(1,N+1):
        z_old = (NR.uniform(0,1,(1,Z.shape[1])) < (Z.sum(axis=0).astype(NP.float)/i))
        num_new = NR.poisson(float(alpha)/i)
        z_new =  NP.ones((1,num_new))
        zi = NP.hstack((z_old,z_new))
        Z = NP.hstack((Z,NP.zeros((Z.shape[0],num_new))))
        Z = NP.vstack((Z,zi))
    feature_count = Z.shape[1]
    return (Z,feature_count)

#distribution of nonzero features per observation
def plotRowHistogram(Z):
    data = Z.sum(axis=1)
    binwidth = 1
    plt.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))
    plt.title("Poisson Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    fig = plt.gcf() 
    plt.show()
    
#growth of nonzero features as observations increase 
def plotMatrixHistogram(Z):
    data = []
    for i in range(Z.shape[0]):
        data.append(sum(x > 0 for x in np.squeeze(np.asarray(Z[0:i,:].sum(axis=0)))))
    plt.plot([y for y in range(1,Z.shape[0]+1)],data)
    plt.xlabel('number of observations')
    plt.ylabel('total features in Z')
    plt.title('Total Features by Observations')
    plt.show()

#asymptotics of features per observation varying with K
def Asymp_obs_K(Z):
    row,col = Z.shape
    data = []
    for i in range(1,col):
        data.append(Z[:,:i].sum()/row)
    plt.plot([y for y in range(1,col)],data)
    plt.xlabel('K')
    plt.ylabel('features per observation')
    plt.title('Features Per Observation Varying With K')
    plt.show()

def weak_obs_K(alpha,N,K):
    data = []
    for i in range(1,K):
        Z = weakLimit(alpha,N,i)
        row,col = Z.shape
        data.append(Z.sum()/row)
    plt.plot([y for y in range(1,K)],data)
    plt.xlabel('K')
    plt.ylabel('features per observation')
    plt.title('Features Per Observation Varying With K')
    plt.show()
    
def harmonic(alpha,N):
    cum = 0
    data = []
    for i in range(1,N):
        cum = cum + float(1)/i
        data.append(cum*alpha)
    plt.plot([y for y in range(1,N)],data)
    plt.xlabel('number of terms')
    plt.ylabel('harmonic sum')
    plt.title('Harmonic Sum with alpha = 100')
    plt.show()    
        
## plotting
#print('Restaurant Process')
#for i in range(5):
#    Z = restaurant(10,50)
#    figure(1)
#    imshow(Z, interpolation='nearest')
#    grid(True)
#    plt.show()
#
##make note of alpha = 100 in your writeup
#Z = restaurant(100,100)
#plotRowHistogram(Z)
#plotMatrixHistogram(Z)
#
#print('Stick Breaking Process')
#for i in range(5):
#    Z = stickBreak(10,50,50)
#    figure(1)
#    imshow(Z, interpolation='nearest')
#    grid(True)
#    plt.show()
#
#Z = stickBreak(100,100,1000)
#plotRowHistogram(Z)
#plotMatrixHistogram(Z)
#Asymp_obs_K(Z)
#    
#print('Weak Limit Approximation')
#for i in range(5):
#    Z = weakLimit(10,50,50)
#    figure(1)
#    imshow(Z, interpolation='nearest')
#    grid(True)
#    plt.show()
#
#Z = weakLimit(100,100,1000)
#plotRowHistogram(Z)
#plotMatrixHistogram(Z)
#weak_obs_K(100,10,500)
#
#harmonic(100,1000)