#! /usr/bin/python

# Imports 

import numpy as np
from numpy.linalg import inv
import numpy.random as npr 
import scipy.special as sps 
import scipy.stats as SPST
from scipy.misc import logsumexp 
from copy import copy
import pdb
from numpy.linalg import det
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
# --- Helper Functions --- # 
# Collapsed Likelihood 
def cllikelihood( data_set , Z , sigma_a=5.0 , sigma_n=.1 ):
    N,D = data_set.shape
    K = Z.shape[1]
    try: 
        tmp = inv(np.dot(Z.T, Z) + (float(sigma_n)/sigma_a)**2*np.eye(K))
    except np.linalg.linalg.LinAlgError:
        print('singular matrix')
        
    try:
        tmp = np.eye(N)-np.dot(np.dot(Z, tmp), Z.T)
    except ValueError:
        print('Value Error')
    tmp = -1/(2*sigma_n**2)*np.trace(np.dot(np.dot(data_set.T, tmp), data_set))
    #ll = np.exp(tmp)
    try:
        ll = tmp - (float(N*D)/2*np.log(2*np.pi)+(N-K)*D*np.log(sigma_n)+K*D*np.log(sigma_a)+float(D)/2*np.log(det(np.dot(Z.T,Z)+(float(sigma_n)/sigma_a)**2*np.eye(K))))
    except np.linalg.linalg.LinAlgError:
        print('LinAlgError')
    
    return ll 

#    X = data_set
#    N,K = Z.shape
#    ZT = np.transpose(Z)
#    XT = np.transpose(X)
#    I = np.identity(K)
#    ZTZI = np.dot(ZT,Z) + float(sigma_n**2)/sigma_a**2 * I
#    num = exp(-float(1)/(2*sigma_n**2) * np.dot(np.dot(XT,I - np.dot(np.dot(Z,np.linalg.inv(ZTZI)),ZT)),X)) 
#    den = 
#    return ll 



# Uncollapsed Likelihood 
def ullikelihood( data_set , Z , A , sigma_n=.1 ):
    N,D = data_set.shape
    X = data_set
    XZA = X - np.dot(Z,A)
    trXZA = np.trace(np.dot(XZA.T,XZA))
    ll = (-1.0*float(N*D)/2) * np.log(2*np.pi*sigma_n**2) - 1.0/(2*sigma_n**2) * trXZA
    return ll 
    
def data_ll_new(data_set,Z_old,A_old,k_new,Z_new,sigma_a,sigma_n):
    N,D = data_set.shape
    X = data_set
    novera = float(sigma_n)/sigma_a
    sigma_I = np.ones((k_new,k_new)) + novera**2*np.eye(k_new)
    XZAZ = np.dot((X - np.dot(Z_old,A_old)).T,Z_new)
    exp_term = 1.0/(2*sigma_n**2)*np.trace(np.dot(np.dot(XZAZ,inv(sigma_I)),XZAZ.T))
    ll = k_new*D*np.log(novera) - float(D)/2 * np.log(det(sigma_I)) + exp_term
    return ll
    
def A_new(data_set,Z_new,Z_old,A_old,sigma_a,sigma_n):
    N,D = data_set.shape
    K = Z_old.shape[1]
    k_new = Z_new.shape[1]
    X = data_set
    A_new = np.zeros((k_new,D))
    novera = float(sigma_n)/sigma_a
    one = np.ones((k_new,k_new))
    XZA = X - np.dot(Z_old,A_old)
    isigma_I = inv(one + novera**2*np.eye(k_new))
    mean = np.dot(np.dot(isigma_I,Z_new.T),XZA)
    cov = sigma_n**2*isigma_I
    
    for i in range(D):
        try:
            A_new[:,i] = SPST.multivariate_normal.rvs(np.squeeze(np.asarray(mean[:,i])),cov)
        except (ValueError,TypeError,IndexError):
            print('ValueError')

    #return A_new
    return A_new
# Compute how many times each history occurs [Kh]
def history_count_set( Z ):

    return count_set 

# Prior on Z
def lpriorZ( Z , alpha ):
    N,K = Z.shape
    m = [0 for x in range(K)]
    for j in range(K):
        for j in range(N):
            m[j] = m[j] + Z[i][j]
	
	harmonic = 0
	K = [0 for x in range(len(Z[0]))]
	limit = [ 0 for x in range(len(Z[0]))]

	for i in range(1,len(Z[0])+1):
		harmonic = harmonic + 1/(1.0*i)
	for i in range(len(Z[0])):
		for j in range(len(Z[1])):
			if Z[i][j] == 1:
				limit[i] = j
	K[0] = limit[0]	
	for i in range(1,len(Z[0])):
		K[i] = limit[i] - limit[i-1]
	

	prod1 = 1
	for i in range(len(Z[0])):
		prod = prod * K[i]
	prod2 = 1
	for k in range(len(Z[1])):
		(math.factorial(len(Z[0]) - m[k]) * math.factorial(m[k]-1)) / math.factorial(N)
    	
	lp = alpha**(len(Z[1])) / ( prod1) * np.exp(-alpha * harmonic) * prod2

	return lp 
    
# Prior on A
def lpriorA( A , sigma_a=5.0 ):
    
    return lp

# Mean function for A 
def mean_A( data_set , Z , sigma_a=.5 , sigma_n=.1 ):
    A = inv(np.dot(Z.T,Z) + (sigma_n/sigma_a)**2*np.eye(len(Z[1]))) * Z.T * data_set
    return A

# --- Resample Functions --- # 
# Resample A 
def resample_A( data_set , Z , sigma_a=5.0 , sigma_n=.1 ):
    X = data_set
    N,K = Z.shape
    D = data_set.shape[1]
    A = np.zeros((K,D))
    ZTZ = np.dot(Z.T,Z)
    novera = float(sigma_n)/sigma_a
    I = np.eye(K)
    ZTX = np.dot(Z.T,X)
    iZTZI = inv(ZTZ + novera**2*I)
    mean = np.dot(iZTZI,ZTX)
    cov = sigma_n**2*iZTZI
    print(mean.shape)
    #Note, that the covariance is defined for each column is just depressing
    for col in range(D):
        #might be in NP
        try:
            A[:,col] = SPST.multivariate_normal.rvs(np.squeeze(np.asarray(mean[:,col])),cov)
        except (ValueError,IndexError):
            print('ValueError')
        
    return A

# Sample next stick length in the stick-breaking representation
def sample_pi_next( pi_min , alpha , data_count ):

    return pi_next 

# --- Samplers --- # 
# Slice sampler from Teh, Gorur, and Ghahramani, fully uncollapsed 
def slice_sampler( data_set , alpha , sigma_a=5 , sigma_n=.1 , iter_count=35 , init_Z=None ):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    Z_set = list()
    A_set = list() 
    
    # Initialize the variables 
    
    # MCMC loop 
    for mcmc_iter in range( iter_count ):
        
        # Sampling existing pi
        
        # Sampling slice_var
            
        # Extending the matrix
        
        # Sampling existing Z

        # Sampling existing A 

        # Compute likelihoods and store 
        ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
        lp_set[ mcmc_iter ] = lpriorA( A , sigma_a ) + lpriorZ( Z , alpha ) 
        A_set.append( A ); Z_set.append( Z ) 

        # print
        print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 

    # return 
    return Z_set , A_set , ll_set , lp_set 
    
# The uncollapsed LG model. In a more real setting, one would want to
# additionally sample/optimize the hyper-parameters!  
def ugibbs_sampler( data_set , alpha , sigma_a=5.0 , sigma_n=.1 , iter_count=35 , init_Z=None ):
    data_count = data_set.shape[0]
    X = data_set
    N = data_count
    K_max = 5
    dim_count = data_set.shape[1] 
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    Z_set = list()
    A_set = list() 
     
    # Initialize Z randomly (explore how different initializations matter)
    Z = np.transpose(np.matrix(SPST.bernoulli.rvs(0.5,size=data_count)))
    active_K = 1      
    arraylog = [ 0 for x in range(iter_count)] 
    arrayfeat = [ 0 for x in range(iter_count)] 
    arraytime = [ 0 for x in range(iter_count)] 
    arrayitter = [ 0 for x in range(iter_count)] 

    for mcmc_iter in range( iter_count ):
        # Sampling existing A 
        A = resample_A(data_set,Z,sigma_a,sigma_n)
        for n in range(data_count):
            for k in range(active_K): 
                # Sampling existing Z 
                # Loop through instantiated Z
                try:
                    IBP_one = float(Z[:,k].sum() - Z[n,k])/(N-1)
                except IndexError:
                    print('Index Error')
                IBP_zero = 1 - IBP_one
                Z_one = np.copy(Z)
                Z_zero = np.copy(Z)
                Z_one[n,k] = 1
                Z_zero[n,k] = 0
                like_one = ullikelihood(data_set,Z_one,A)
                like_zero = ullikelihood(data_set,Z_zero,A)
                shift = max([like_one,like_zero])
                like_one = like_one - shift
                like_zero = like_zero - shift
                update_probability = float(IBP_one*np.exp(like_one))/(IBP_one*np.exp(like_one) + IBP_zero*np.exp(like_zero))
                if (math.isnan(update_probability)):
                    update_probability = 0
                try:
                    Z[n,k] = SPST.bernoulli.rvs(update_probability)
                except ValueError:
                    print('ValueError') 
            # Consider adding new features - decide whether it should be
            # collapsed or uncollapsed.
            pk_new = list()
            X_ll = list()
            for k_new in range(K_max):
                if k_new > 0:
                    Z_new = np.zeros((N,k_new))
                    Z_new[n,:] = np.ones(k_new)
                    X_ll.append(data_ll_new(data_set,Z,A,k_new,Z_new,sigma_a,sigma_n))
                else:  
                    #X_ll.append(ullikelihood(data_set,Z,A))
                    X_ll.append(0)
            shift = max(X_ll)
            pk_new = [SPST.poisson.pmf(i,float(alpha)/N)*np.exp(X_ll[i]-shift) for i in range(K_max)]
            normalise = sum(pk_new)
            pk_normalise = [float(p)/normalise for p in pk_new]
            try:
                num_new, = np.where(np.random.multinomial(1,pk_normalise) == 1)[0]
            except (ValueError,IndexError):
                print('ValueError')
            if num_new > 0:
                Z_new = np.zeros((N,num_new))
                Z_new[n,:] = np.ones(num_new)
                #sample A_new
                A = np.vstack((A,A_new(X,Z_new,Z,A,sigma_a,sigma_n))) 
                Z = np.hstack((Z,Z_new))
            active_K = Z.shape[1]               
        # Remove any unused
        # remove corresponding rows in A
        Z_sum = np.array(Z.sum(axis=0))
        nonzero = list()
        for j in range(Z_sum.shape[1]):
            if Z_sum[0,j] != 0:
                nonzero.append(j)
        Z = Z[:,nonzero] 
        A = A[nonzero,:]
        active_K = Z.shape[1]
        arraylog[mcmc_iter] = ullikelihood(data_set,Z,A)
        arrayfeat[mcmc_iter] = active_K
        arraytime[mcmc_iter] = time.time()
        arrayitter[mcmc_iter] = mcmc_iter
       
     
        
        # Compute likelihood and prior 
        ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
#        lp_set[ mcmc_iter ] = lpriorA( A , sigma_a ) + lpriorZ( Z , alpha ) 
        A_set.append( A ); Z_set.append( Z ) 
        # print
        #print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 
        lp_set = 0
    # return 
    print arrayfeat
    yauplot( 'USampler:Log-Likelihood VS Time','Time','Log-Likelihood',arraytime,arraylog)
    yauplot( 'USampler:Features VS Time','Time','Number of Features',arraytime,arrayfeat)
    yauplot( 'USampler:Log-Likelihood VS Number of Iterations','Number of Iterations','Log-Likelihood',arrayitter,arraylog)
    yauplot( 'USampler:Numbers of Features VS Number of Iterations','Number of Iterations','Number of Features',arrayitter,arrayfeat)
       
    return Z_set , A_set , ll_set , lp_set 

# The collapsed LG model from G&G.  In a more real setting, one would
# want to additionally sample/optimize the hyper-parameters!
def cgibbs_sampler( data_set , alpha , sigma_a=5 , sigma_n=.1 , iter_count=35 , init_Z=None ): 
    #no idea how to set K_max
    data_count = data_set.shape[0] 
    N = data_count
    K_max = max([3*float(alpha)/N,3])
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    Z_set = list()
    A_set = list() 
    # Initialize Z randomly (explore how different initializations matter)
    #G&G initialization
    Z = np.transpose(np.matrix(SPST.bernoulli.rvs(0.5,size=data_count)))
    active_K = 1    
    # MCMC loop 
    # MCMC loop 
    # MCMC loop 
    for mcmc_iter in range(iter_count):
        for n in range(data_count):
            for k in range(active_K): 
                #existing feature update
                try:
                    IBP_one = float(Z[:,k].sum() - Z[n,k])/(N-1)
                except IndexError:
                    print('Index Error')
                IBP_zero = 1 - IBP_one
                Z_one = np.copy(Z)
                Z_zero = np.copy(Z)
                Z_one[n,k] = 1
                Z_zero[n,k] = 0
                like_one = cllikelihood(data_set,Z_one)
                like_zero = cllikelihood(data_set,Z_zero)
                shift = max([like_one,like_zero])
                like_one = like_one - shift
                like_zero = like_zero - shift
                update_probability = float(IBP_one*np.exp(like_one))/(IBP_one*np.exp(like_one) + IBP_zero*np.exp(like_zero))
                if (math.isnan(update_probability)):
                    update_probability = 0
                try:
                    Z[n,k] = SPST.bernoulli.rvs(update_probability)
                except ValueError:
                    print('ValueError')
                
            # Consider adding new features
            pk_new = list()
            Z_new_ll = list()
            for k_new in range(K_max):
                if k_new > 0:
                    new_block = np.zeros((N,k_new))
                    new_block[n,:] = np.ones(k_new)
                    Z_new = np.hstack((Z,new_block)) 
                else:  
                    Z_new = Z
                Z_new_ll.append(cllikelihood(data_set,Z_new))
            shift = max(Z_new_ll)
            pk_new = [SPST.poisson.pmf(i,float(alpha)/N)*np.exp(Z_new_ll[i]-shift) for i in range(K_max)]
            normalise = sum(pk_new)
            pk_normalise = [float(p)/normalise for p in pk_new]
            try:
                num_new, = np.where(np.random.multinomial(1,pk_normalise) == 1)[0]
            except (ValueError,IndexError):
                print('ValueError')
            if num_new > 0:
                new_block = np.zeros((N,num_new))
                new_block[n,:] = np.ones(num_new)
                Z = np.hstack((Z,new_block)) 
            active_K = Z.shape[1]
            # Loop through instantiated Z 
                           
        # Remove any unused
        Z_sum = np.array(Z.sum(axis=0))
        nonzero = list()
        for j in range(Z_sum.shape[1]):
            if Z_sum[0,j] != 0:
                nonzero.append(j)
        Z = Z[:,nonzero] 
        active_K = Z.shape[1]
        # Compute likelihood and also the mean value of A, just so we
        # can visualize it later
        ll_set[ mcmc_iter ] = cllikelihood( data_set , Z , sigma_a , sigma_n )
        #lp_set[ mcmc_iter ] = lpriorZ( Z , alpha ) 
        A = mean_A( data_set , Z , sigma_a , sigma_n )
        A_set.append( A ); 
        Z_set.append(Z)

        # print
        #print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 
        
    # return 
    lp_set = 0
    return Z_set , A_set , ll_set , lp_set 
    
def yauplot(title,x_axis,y_axis,data_x,data_y):
    plt.plot(data_x,data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    from make_toy_data import generate_data
    iterate = 35
    alpha = 2
    data_count = 50
    data_set , Z , A = generate_data( data_count , 'gg' ) 
    print('THIS IS X')
    print(data_set)
    #Z_set, A_set, ll_set, lp_set = cgibbs_sampler(data_set, alpha)
    Z_set, A_set, ll_set, lp_set = ugibbs_sampler(data_set, alpha,iter_count = iterate)
    
    title = 'uncollapsed: log likelihood vs. iterations'
    x_axis = 'iterations'
    y_axis = 'log likelihood'
    data_x = [i for i in range(1,iterate+1)]
    data_y = ll_set
    plot(title,x_axis,y_axis,data_x,data_y)
    #print(ll_set)
