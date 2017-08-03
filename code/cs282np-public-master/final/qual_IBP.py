
# Imports 
import numpy as np
from numpy.linalg import inv
import numpy.random as npr 
#import scipy.special as sps 
#import scipy.stats as SPST
#from scipy.misc import logsumexp 
from copy import copy
import pdb
import time
from numpy.linalg import det
import math
#import matplotlib.pyplot as plt
from generate_data import generate_data,display_W,log_data_zw,construct_data,generate_gg_blocks
from random import randint
from itertools import combinations
#from make_toy_data import generate_random_A

#you need a fair uncollapsed comparison.  
#pull up finale's paper
#produce a fair implementation.
#partially collapsed sampler  

# --- Helper Functions --- # 

# Uncollapsed Likelihood 
def ullikelihood( data_set , Z , A , sigma_n):
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
            #print('ValueError')
            q = 3

    #return A_new
    return A_new
    

# Mean function for A 
def mean_A( data_set , Z , sigma_a=.5 , sigma_n=.1 ):
    A = inv(np.dot(Z.T,Z) + (sigma_n/sigma_a)**2*np.eye(len(Z[1]))) * Z.T * data_set
    return A

# --- Resample Functions --- # 
# Resample A 
def resample_A( data_set , Z , sigma_a , sigma_n ):
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
    for col in range(D):
        try:
            A[:,col] = np.random.multivariate_normal(np.squeeze(np.asarray(mean[:,col])),cov)
            #A[:,col] = SPST.multivariate_normal.rvs(np.squeeze(np.asarray(mean[:,col])),cov)
        except (ValueError,IndexError):
            #print('ValueError')
            q = 3
        
    return A
    
#def Z_posterior(z_row, Z):
#    N,K = Z.shape
#    Z_post = 1
#    z_prob = 1./(N+1) * np.sum(Z,axis=0)
#    #print(z_prob)
#    for i in range(K):
#        if z_row[i] == 1:
#            Z_post = Z_post + math.log(z_prob[i])
#        else:
#            Z_post = Z_post + math.log(1 - z_prob[i])
#    return Z_post
    
def Z_posterior(z_row, z_prob):
    Z_post = 0
    K = len(z_prob)
    for i in range(K):
        if z_row[i] == 1:
            if z_prob[i] == 0:
                Z_post = float('-Inf')
            else:
                Z_post = Z_post + math.log(z_prob[i])
        else:
            if 1 - z_prob[i] == 0:
                Z_post = float('-Inf')
            else:
                Z_post = Z_post + math.log(1 - z_prob[i])
    return Z_post
    
def pred_ll_IBP(held,Z,W,sig):
    #should you be comparing the predictive log likelihood?  I think you should
    R,T = held.shape
    N,K = Z.shape
    N,K = Z.shape
    z_prob = 1./(N+1) * np.sum(Z,axis=0)
    log_pred = 0
    for i in range(R):
        pred_row = 0
        for j in range(2**K):
            binary = list(map(int,"{0:b}".format(j)))
            pad_binary = [0]*(K-len(binary)) + binary
            log_z_post = Z_posterior(pad_binary,z_prob)
            total_z = np.array(pad_binary)
            pred_row = pred_row + np.exp(log_data_zw(held[i,:],total_z,W,sig) + log_z_post)
        log_pred = log_pred + np.log(pred_row)
    return log_pred

def sample_recover(held_out,observe,Z,A,sig,obs_indices): 
    _,obs = observe.shape 
    R,T = held_out.shape
    K,_ = A.shape
    N,_ = Z.shape
    #print("this is N")
    #print(N)
    z_prob = 1./(N+1) * np.sum(Z,axis=0)    
    log_recover = 0 
    Z_total = np.vstack((Z,np.zeros((R,K))))
    indices = [i for i in range(T)]
    hidden = [x for x in indices if x not in obs_indices]
    #print(hidden)
    iterate = 50
    for it in range(iterate):
        for n in range(N,N+R):
            for k in range(K): 
                # Sampling existing Z 
                # Loop through instantiated Z
                IBP_one = float(Z_total[:,k].sum() - Z_total[n,k])/(N+R-1)
                IBP_zero = 1 - IBP_one
                Z_one = np.copy(Z_total)
                Z_zero = np.copy(Z_total)
                Z_one[n,k] = 1
                Z_zero[n,k] = 0
                like_one = ullikelihood(observe,Z_one[N:N+R,:],A[:,obs_indices],sig)
                like_zero = ullikelihood(observe,Z_zero[N:N+R,:],A[:,obs_indices],sig)
                shift = max([like_one,like_zero])
                like_one = like_one - shift
                like_zero = like_zero - shift
                update_probability = float(IBP_one*np.exp(like_one))/(IBP_one*np.exp(like_one) + IBP_zero*np.exp(like_zero))
                if (math.isnan(update_probability)):
                    update_probability = 0
                Z_total[n,k] = np.random.binomial(1,update_probability)
    #print("sampled Z")
    #print(Z_total)
    for r in range(R):
        log_recover += Z_posterior(Z_total[N+r,:],z_prob)
        #print('Z row')
        #print(Z_total[N+r,:])
        #print('log likelihood of draw')
        #print(Z_posterior(Z_total[N+r,:],z_prob))
    log_recover += log_data_zw(held_out[:,hidden],Z_total[N:N+R,:],A[:,hidden],sig)
    return log_recover
    
def recover_IBP(held,observe,Z,W,sig,obs_indices):
    _,obs = observe.shape 
    R,T = held.shape
    K,T = W.shape
    N,T = Z.shape
    z_prob = 1./(N+1) * np.sum(Z,axis=0)    
    log_recover = 0
    upper_bound = 0
    lower_bound = 0
    for i in range(R):
        #print('row')
        #print(i)
        f_max = 0
        o_max = 0
        f_error = 0
        o_error = 0
        numu = 0
        numl = 0
        denu = 0
        denl = 0
        valid = True
        for j in range(2**K):
            binary = list(map(int,"{0:b}".format(j)))
            pad_binary = [0]*(K-len(binary)) + binary
            log_z_post = Z_posterior(pad_binary,z_prob)
            if math.isinf(log_z_post):
                continue
            total_z = np.array(pad_binary)
            #print(held[i,:].shape)
            #print(total_z.shape)
            #print(W.shape)
            fll = log_data_zw(held[i,:],total_z,W,sig) + log_z_post
            oll = log_data_zw(observe[i,:],total_z,W[:,obs_indices],sig) + log_z_post
            if valid:
                f_max = fll
                o_max = oll
                valid = False
            else:
                if fll > f_max:
                    f_error = f_max
                    f_max = fll
                if oll > o_max:
                    o_error = o_max
                    o_max = oll
        log_recover = log_recover + f_max - o_max
        if f_error == 0:
            numu = numu + f_max
        elif f_max - f_error > 10:
            numu = numu + f_max
        else:
            numu = numu + np.log(np.exp(f_max-f_error) + (2**K - 1)) + f_error
        
        numl = numl + f_max
        
        if o_error == 0:
            denu = denu + o_max
        elif o_max - o_error > 10:
            denu = denu + o_max
        else:
            denu = denu + np.log(np.exp(o_max-o_error) + (2**K - 1)) + o_error
        
        denl = denl + o_max
        
        upper_bound = upper_bound + numu - denl
        lower_bound = lower_bound + numl - denu
        #if math.isinf(lower_bound):
            #print("lower bound isinf")
    return log_recover

def truncate(Z,A,select):
    N,K = Z.shape
    z_sum = np.sum(Z,axis=0)
    index_sum = [(z_sum[i],i) for i in range(K)]
    index_sum.sort(key=lambda tup: tup[1])
    indices = [index_sum[i][1] for i in range(min(select,K))]
    Z_trunc = Z[:,indices]
    A_trunc = A[indices,:]
    return (Z_trunc, A_trunc)

def print_posterior(Z,W,data_dim):
    N,K = Z.shape
    for j in range(2**K):
        binary = list(map(int,"{0:b}".format(j)))
        pad_binary = [0]*(K-len(binary)) + binary
        prob = np.exp(Z_posterior(pad_binary,Z))
        if prob > 0.01:
            pad_binary = np.array(pad_binary)
            reconstruct = np.dot(pad_binary,W)
            #print("pad binary, reconstruct, probability")
            #print(pad_binary)
            #print(prob)
            display_W(reconstruct,data_dim,'four')
    
# The uncollapsed LG model. In a more real setting, one would want to
# additionally sample/optimize the hyper-parameters!  
def ugibbs_sampler(data_set,held_out,alpha,sigma_n,sigma_a,iter_count,select,trunc,observe,obs_indices,limit,data_dim=[3,3,2,2],init=False,display=False):
    data_count = data_set.shape[0]
    X = data_set
    N = data_count
    K_max = 5
    dim_count = data_set.shape[1] 
    ll_set = np.zeros( [ iter_count ] )
    lp_set = np.zeros( [ iter_count ] ) 
    iter_time = []  
    #Z = Z_gen
    #Z = np.random.binomial(1,0.25,[N,1])
    #Z = np.random.binomial(1,0.25,[N,1])
    Z = np.zeros((N,1))
    Z[randint(0,N-1),0] = 1
    active_K = 1  
    pred_ll = []  
    pred_prob = 0
    rec_ll = []
    rec = 0
    print('getting further')
    # MCMC loop 
    for mcmc_iter in range( iter_count ):
        if mcmc_iter == 0:
            start = time.time()
        if mcmc_iter > 0:
            if np.sum(iter_time) > limit:
                #print("break on time")
                #print(np.sum(iter_time))
                break 
        # Sampling existing A 
        start = time.time()
        A = resample_A(data_set,Z,sigma_a,sigma_n)
        
        for n in range(data_count):
            for k in range(active_K): 
                #print('inner loop')
                # Sampling existing Z 
                # Loop through instantiated Z
                try:
                    IBP_one = float(Z[:,k].sum() - Z[n,k])/(N-1)
                except IndexError:
                    #print('Index Error')
                    q = 3
                IBP_zero = 1 - IBP_one
                Z_one = np.copy(Z)
                Z_zero = np.copy(Z)
                Z_one[n,k] = 1
                Z_zero[n,k] = 0
                like_one = ullikelihood(data_set,Z_one,A,sigma_n)
                like_zero = ullikelihood(data_set,Z_zero,A,sigma_n)
                shift = max([like_one,like_zero])
                like_one = like_one - shift
                like_zero = like_zero - shift
                update_probability = float(IBP_one*np.exp(like_one))/(IBP_one*np.exp(like_one) + IBP_zero*np.exp(like_zero))
                if (math.isnan(update_probability)):
                    update_probability = 0
                try:
                    Z[n,k] = np.random.binomial(1,update_probability)
                except ValueError:
                    #print('ValueError')   
                    q = 3
        print('outer loop')
        #to quick return, indent 
        # Remove any unused
        # remove corresponding rows in A
        Z_sum = np.array(Z.sum(axis=0))
        nonzero = list()
        for j in range(Z_sum.shape[0]):
            if Z_sum[j] != 0:
                nonzero.append(j)
        Z = Z[:,nonzero] 
        A = A[nonzero,:]
        active_K = Z.shape[1]
        if active_K < trunc:   
            #Z_new = np.random.binomial(1,0.005,[N,1])
            #Z_new = np.zeros((N,1))
            Z_new = np.zeros((N,1))
            Z_new[randint(0,N-1)] = 1
            mean = np.zeros(dim_count)
            cov = sigma_a * np.eye(dim_count)
            A_new = np.random.multivariate_normal(mean,cov)
            A = np.vstack((A,A_new))
            Z = np.hstack((Z,Z_new))
            active_K = Z.shape[1]

        if mcmc_iter%1 == 0 and display:
            print("iteration: " + str(mcmc_iter))
            print("Sparsity: " + str(np.sum(Z,axis=0)))
            print('predictive log likelihood: ' + str(pred_prob))
            print('recovery log likelihood: ' + str(rec))
            print("active K: " + str(active_K))
        
        # Compute likelihood and prior 
        #beware of init variable--logically unsound
        if mcmc_iter%10 == 0 and mcmc_iter > 0 and init:
            pred_prob = 0
            pred_ll.append(pred_prob)
            rec = recover_IBP(held_out,observe,Z,A,sigma_n,obs_indices)
            #rec = sample_recover(held_out,observe,Z,A,sigma_n,obs_indices)
            rec_ll.append(rec)
            end = time.time()
            iter_time.append(end - start)
            start = time.time()
        ll_set[mcmc_iter]  = log_data_zw(data_set,Z,A,sigma_n)
    iter_time = np.cumsum(iter_time)
    return Z,A,ll_set,pred_ll,rec_ll,iter_time

    
def plot(title,x_axis,y_axis,data_x,data_y):
    plt.plot(data_x,data_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    iterate = 110
    alpha = 2.0
    data_count = 200
    held_out = 50
    sig = 0.1 #noise
    sig_w = 0.2 #feature deviation
    small_x = 3
    small_y = 3
    big_x = 3
    big_y = 3
    data_dim = (small_x,small_y,big_x,big_y)
    feature_count = big_x*big_y
    T = small_x*small_y*big_x*big_y
    select = 12
    data_type = 'corr'
    #full_data,Z_gen = generate_data(feature_count,data_count + held_out,T,sig,data_type)
    full_data,Z_gen = construct_data(data_dim,data_count + held_out,sig,data_type,corr_value=2)
    
    
    Y = full_data[:data_count,:]
    held_out = full_data[data_count:,:]
    Z,W,ll_set,pred_ll = ugibbs_sampler(Y,held_out,alpha,sig,sig_w,iterate,select)
    
    approx = np.dot(Z,W)
    for i in range(10):
        print("sample: " + str(i))
        print("features probability")
        #print(prob_matrix[i,:])
        print("features selected")
        print(Z[i,:])
        display_W(approx[i:i+1,:],data_dim,'nine')
        print("data: " + str(i))
        display_W(Y[i:i+1,:],data_dim,'nine')
    
    #Output the posterior 
    Z_trunc,W_trunc = truncate(Z,W,select)
    print_posterior(Z_trunc,W_trunc)
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


