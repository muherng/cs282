
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
from simulate import plotRowHistogram
from simulate import plotMatrixHistogram
from make_toy_data import generate_random_A

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
#    return ll 



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
    #print(mean.shape)
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

#To avoid errors, this is slow
#find mu_star the last 
def find_mu_star(Z,mu):
    Z_sum = np.asarray(Z.sum(axis=0))[0]
    #print(Z_sum)
    mu_star = 1.0
    cnt= 0 
    for i in range(Z.shape[1]):
        if Z_sum[i] > 0:
            cnt = cnt+1
            mu_star = mu[i]
    return mu_star

def find_K_dagger(Z):
    Z_sum = np.asarray(Z.sum(axis=0))[0]
    K_dagger = 1
    for i in range(Z.shape[1]-1,0,-1):
        if Z_sum[i] > 0:
            K_dagger = i+1
            break
    return K_dagger
 
def find_K_star(mu,s):
   
    K_star = 0
    for i in range(len(mu)):
        if mu[i] > s:
            K_star = i
        else:
            break
    return K_star



    

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

# --- Samplers --- # 
# Slice sampler from Teh, Gorur, and Ghahramani, fully uncollapsed 
def slice_sampler( data_set , alpha , sigma_a=5.0 , sigma_n=.1 , iter_count=35 , init_Z=None ):
    data_count = data_set.shape[0]
    X = data_set
    N = data_count
    dim_count = data_set.shape[1] 
    D = dim_count
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    Z_set = list()
    A_set = list() 
    
    # Initialize the variables 
    # Everything Zero Indexed 
    #Z will be initialized a single column, bernoulli flipping p=mu_1
    #mu_1 will be initialized according to stick breaking Beta(alpha,1)
    #s initialize unif(0,mu_1)
    #K_star is maximal feature such that mu_k_star > s, so init = 0
    #K_dagger init =  1
    # In order, mu, Z, s, K_star, K_dagger
    
    mu = list()
    #break first two sticks to avoid getting stuck in base case
    init = 1
    mu = generate_mu(alpha,init)
    
    Z = np.zeros((N,init))
    for j in range(init):
         Z[:,j] = SPST.bernoulli.rvs(mu[j],size=N)
    
    Z = np.matrix(Z)
    #Z = np.transpose(np.matrix(SPST.bernoulli.rvs(mu[0],size=data_count)))
    #Z = np.hstack((Z,np.zeros((N,1))))
    #Z = np.hstack((Z,np.transpose(np.matrix(SPST.bernoulli.rvs(mu[1],size=data_count)))))
    #A = generate_random_A(2,D)
    A = resample_A(data_set,Z,sigma_a,sigma_n)
    #A = np.vstack((A,generate_random_A(1,D)))
    K_active = Z.shape[1]
    s = SPST.uniform(0,mu[0])
    K_star = 0
    K_dagger = init-1
    
    # MCMC loop 
    for mcmc_iter in range( iter_count ):
        #print(Z)
        print(Z.shape[0] ,Z.shape[1])
        
        
        # Sampling existing pi
        
        # Sampling slice_var
            
        # Extending the matrix
        
        # Sampling existing Z

        # Sampling existing A 
        
        #From paper, sample slice, extend matrix (mu,Z,A), Z, A, mu (order is probably irrelevant)
        #Sampling slice_var
        mu_star = find_mu_star(Z,mu)
        #sample s
        s = SPST.uniform.rvs(0,mu_star)
        #extend matrix (mu,K_star,K_dagger,Z,A)
        old_K_dagger = K_dagger
        #you must generate mu until mu < s
        len_mu = len(mu)
        #print(len_mu)
        while mu[len_mu-1] > s:
            tmp = mh_new_mu(mu[len_mu-1], alpha, N, mu[len_mu-1] * 0.05, T=100) 
            len_mu = len_mu+1
            mu.append(tmp)
            K_star = find_K_star(mu,s) 

        K_dagger = len(mu) - 1
        new_features = K_dagger - old_K_dagger
        Z_new = np.zeros((N,new_features))
        for j in range(new_features):
            Z_new[:,j] = SPST.bernoulli.rvs(mu[j+old_K_dagger+1],size=N)
        Z = np.hstack((Z,Z_new))
        #Z = np.hstack((Z,np.zeros((N,new_features))))
        #print(Z)
        #Take Note! generate_random_A has sigma_a = 5.0, you did not pass it in
        #A = np.vstack((A,generate_random_A(new_features,D))) 
        A = resample_A(data_set,Z,sigma_a,sigma_n)
        #Update Z
        for i in range(N):
            for k in range(K_star+1):
                data_row = data_set[i,:]
                Z_one = np.asarray(np.copy(Z[i,:]))[0]
                Z_zero = np.asarray(np.copy(Z[i,:]))[0]
                try: 
                    Z_one[k] = 1
                except IndexError:
                    print('IndexError')
                    
                Z_zero[k] = 0
                like_one = ullikelihood(np.matrix(data_row),Z_one,A)
                like_zero = ullikelihood(np.matrix(data_row),Z_zero,A)
                shift = max([like_one,like_zero])
                like_one = like_one - shift
                like_zero = like_zero - shift
                #print('like one')
                #print(like_one)
                Z_one_matrix = np.matrix(np.copy(Z))
                Z_one_matrix[i,k] = 1
                mu_star_one = find_mu_star(Z_one_matrix,mu)
                Z_zero_matrix = Z_one_matrix
                Z_zero_matrix[i,k] = 0
                mu_star_zero = find_mu_star(Z_zero_matrix,mu)
#                if k == K_dagger:
#                    mu_star_one = mu[k]
#                    mu_star_zero = mu_star
#                else: 
#                    mu_star_one = mu_star
#                    mu_star_zero = find_mu_star(Z,mu)
                mu_k = mu[k]
                mu_frac_one = float(mu_k)/mu_star_one
                #Note the 1 - mu_k
                try:
                    mu_frac_zero = float(1 - mu_k)/mu_star_zero
                except ZeroDivisionError:
                    print('ZeroDivisionError')
                update_probability = float(mu_frac_one*np.exp(like_one))/(mu_frac_one*np.exp(like_one) + mu_frac_zero*np.exp(like_zero))
                #print('update probability')
                #print(update_probability)
                
                if (math.isnan(update_probability)):
                    print('Nan update')
                    update_probability = 0
                try:
                    Z[i,k] = SPST.bernoulli.rvs(update_probability)
                except ValueError:
                    print('ValueError') 
        #print(Z)
                
        #Update A
        #STOPPING ALL A UPDATES
        A = resample_A(data_set,Z,sigma_a,sigma_n)
    
        m = [ Z[:,k].sum() for k in range(K_dagger)]
        mu[0] = mh_old_mu(m[0], 1, mu[1], data_count, (1-mu[1])*0.05, 100)
        for k in range(1,K_dagger-1):
            mu[k] = mh_old_mu(m[k], mu[k-1], mu[k+1], data_count, (mu[k-1]-mu[k+1])*0.05, 100)
        mu[K_dagger]= mh_new_mu(mu[K_dagger-1],alpha, data_count, mu[K_dagger-1] * 0.05, 100)
#        mu[0] = mh_old_mu(m[0], 1, mu[1], data_count, (1-mu[1])*0.05, 100)
#        for k in range(1,K_dagger-1):
#            mu[k] = mh_old_mu(m[k], mu[k-1], mu[k+1], data_count, (mu[k-1]-mu[k+1])*0.05, 100)
#        mu[K_dagger]= mh_new_mu(mu[K_dagger-1],alpha, data_count, mu[K_dagger-1] * 0.05, 100)
        
        #print(mu)
                
        # Compute likelihoods and store 
        ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
        #lp_set[ mcmc_iter ] = lpriorA( A , sigma_a ) + lpriorZ( Z , alpha ) 
        #A_set.append( A ); Z_set.append( Z ) 

        # print
        #print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 
        
        
        Z_set.append(Z)
        A_set.append(A)
        #ll_set = 0
        lp_set = 0
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
    # MCMC loop 
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
       
        
        # Compute likelihood and prior 
        ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
#        lp_set[ mcmc_iter ] = lpriorA( A , sigma_a ) + lpriorZ( Z , alpha ) 
        A_set.append( A ); Z_set.append( Z ) 
        # print
        #print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 
        lp_set = 0
    # return 
    return Z_set , A_set , ll_set , lp_set 

# The collapsed LG model from G&G.  In a more real setting, one would
# want to additionally sample/optimize the hyper-parameters!
def cgibbs_sampler( data_set , alpha , sigma_a=5 , sigma_n=.1 , iter_count=35 , init_Z=None ): 
    #no idea how to set K_max
    data_count = data_set.shape[0] 
    N = data_count
    K_max = int(max([3*float(alpha)/N,3]))
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    Z_set = list()
    A_set = list() 
    # Initialize Z randomly (explore how different initializations matter)
    #G&G initialization
    Z = np.transpose(np.matrix(SPST.bernoulli.rvs(0.5,size=data_count)))
    active_K = 1    
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
    
def plot(title,x_axis,y_axis,data_x,data_y):
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
    Z_set, A_set, ll_set, lp_set = slice_sampler(data_set, alpha)
    print('THIS IS APPROX')
    approx = Z_set[34]*A_set[34]
    print(approx)
    print(ll_set)
    #Z_set, A_set, ll_set, lp_set = cgibbs_sampler(data_set, alpha)
#    Z_set, A_set, ll_set, lp_set = cgibbs_sampler(data_set, alpha,iter_count = iterate)
#    Z_final = Z_set[iterate-1]
#    plotRowHistogram(Z_final)
#    plotMatrixHistogram(Z_final)
#    title = 'collapsed: log likelihood vs. iterations'
#    x_axis = 'iterations'
#    y_axis = 'log likelihood'
#    data_x = [i for i in range(1,iterate+1)]
#    data_y = ll_set
#    plot(title,x_axis,y_axis,data_x,data_y)
    #print(ll_set)
