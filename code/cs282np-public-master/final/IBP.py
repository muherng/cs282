
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
from generate_data import generate_data,display_W
#from make_toy_data import generate_random_A

#you need a fair uncollapsed comparison.  
#pull up finale's paper
#produce a fair implementation.
#partially collapsed sampler  

# --- Helper Functions --- # 

# Uncollapsed Likelihood 
def ullikelihood( data_set , Z , A , sigma_n=.1 ):
    N,D = data_set.shape
    X = data_set
    XZA = X - np.dot(Z,A)
    trXZA = np.trace(np.dot(XZA.T,XZA))
    ll = (-1.0*float(N*D)/2) * np.log(2*np.pi*sigma_n**2) - 1.0/(2*sigma_n**2) * trXZA
    #ll = - np.log(2*np.pi*sigma_n**2) - 1.0/(2*sigma_n**2) * trXZA
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
    #print(mean.shape)
    #Note, that the covariance is defined for each column is just depressing
    for col in range(D):
        #might be in NP
        try:
            A[:,col] = SPST.multivariate_normal.rvs(np.squeeze(np.asarray(mean[:,col])),cov)
        except (ValueError,IndexError):
            print('ValueError')
        
    return A

#equation 25
#you pass mu[k-1]
def mu_new_update(mu, mu_prev, alpha, N):
    if (mu < mu_prev) & (mu>0):
        tmp=0
        for i in range(N):
            tmp += 1/float(i+1)*(1-mu)**(i+1)
        return np.exp(alpha*tmp)*mu**(alpha-1)*(1-mu)**N
    else:
        return 0
    
#equation 28
#you pass mu[k+1] and mu[k-1]
def mu_old_update(mu, m_k, mu_prev, mu_next, data_count):
    if (mu< mu_prev) & (mu>mu_next):
        return mu**(m_k-1)*(1-mu)**(data_count-m_k)
    else:
        return 0
    
#MH algorithm
#for equation 25
#pass m[k-1]
def mh_new_mu(mu_prev, alpha, N, sigma_g, T=100):
    mu = npr.uniform(0,mu_prev,1)[0]
    t=0
    sigma_g = max(sigma_g,0.001)
    while t<T:
        #print t
        mu_new = npr.normal(mu, sigma_g, 1)[0]
        if (mu_new>0) & (mu_new<mu_prev):
            A = min(1, mu_new_update(mu_new, mu_prev, alpha, N)/mu_new_update(mu, mu_prev, alpha, N))
            accept = npr.binomial(1, A, 1)
        else:
            accept = 0
        if accept==1:
            mu = mu_new
            t+=1
    return mu

#for equation 28
#pass m[k] , mu[k-1], mu[k+1]
def mh_old_mu(m_k, mu_prev, mu_next, data_count, sigma_g, T=100):
    mu = npr.uniform(mu_next,mu_prev,1)[0]
    t=0
    sigma_g = max(sigma_g,0.001)

    while t<T:
        mu_new = npr.normal(mu, sigma_g, 1)[0]
        if (mu_new>mu_next) & (mu_new<mu_prev):
            A = min(1, mu_old_update(mu_new, m_k, mu_prev, mu_next, data_count)/mu_old_update(mu, m_k, mu_prev, mu_next, data_count))
            accept = npr.binomial(1, A, 1)
        else:
            accept = 0
        if accept==1:
            mu = mu_new
            t+=1
    return mu
    
def generate_mu(alpha,number):
    mu = list()
    for i in range(number):
        if i >= 1:
            mu.append(SPST.beta.rvs(alpha,1)*mu[i-1])
        else:
            mu.append(SPST.beta.rvs(alpha,1))
    return mu
 
    
def log_data_zw(Y,Z,W,sig):
    delta = Y - np.dot(Z,W)
    delta_sum = np.trace(np.dot(delta.T,delta))
    ll =  -1./(2*sig**2) * delta_sum
    return ll

# The uncollapsed LG model. In a more real setting, one would want to
# additionally sample/optimize the hyper-parameters!  
def ugibbs_sampler(data_set,alpha,sigma_n,sigma_a,iter_count,Z_gen):
    data_count = data_set.shape[0]
    X = data_set
    N = data_count
    K_max = 5
    dim_count = data_set.shape[1] 
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
     
    # Initialize Z randomly (explore how different initializations matter)
    #I'm going to explore different initalizations
    #Z = np.transpose(np.matrix(SPST.bernoulli.rvs(0.25,size=data_count)))
    
    #Z = Z_gen
    Z = np.random.binomial(1,0.5,[N,1])
    active_K = 1      
    # MCMC loop 
    for mcmc_iter in range( iter_count ):
        print(mcmc_iter)
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
        print(Z_sum.shape)
        for j in range(Z_sum.shape[0]):
            if Z_sum[j] != 0:
                nonzero.append(j)
        Z = Z[:,nonzero] 
        A = A[nonzero,:]
        active_K = Z.shape[1]
       
        
        # Compute likelihood and prior 
        #ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
        ll_set[mcmc_iter]  = log_data_zw(data_set,Z,A,sigma_n)
    # return 
    return Z,A,ll_set

    
def plot(title,x_axis,y_axis,data_x,data_y):
    plt.plot(data_x,data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    F = 4 #features
    T = 36 #length of datapoint
    N = 100 #data size
    sig = 0.1 #noise
    sig_w = 0.7 #feature deviation
    Y,Z_gen = generate_data(F,N,T,sig)
    iterate = 10
    alpha = 1.0
    Z,W,ll_set = ugibbs_sampler(Y,alpha,sig,sig_w,iterate,Z_gen)
    
    approx = np.dot(Z,W)
    for i in range(10):
        print("sample: " + str(i))
        print("features probability")
        #print(prob_matrix[i,:])
        print("features selected")
        print(Z[i,:])
        display_W(approx[i:i+1,:])
        print("data: " + str(i))
        display_W(data_set[i:i+1,:])
#    display_W(W)
#    Z_final = Z_set[iterate-1]
#    plotRowHistogram(Z_final)
#    plotMatrixHistogram(Z_final)
#    title = 'Slice Sampler: log likelihood vs. iterations'
#    x_axis = 'iterations'
#    y_axis = 'log likelihood'
#    data_x = [i for i in range(1,iterate+1)]
#    data_y = ll_set
#    plot(title,x_axis,y_axis,data_x,data_y)
#    print(ll_set)
