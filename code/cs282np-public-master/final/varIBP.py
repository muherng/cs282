# This is a very basic file for variational inference, taken from the
# tech report.  
import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 
from generate_data import generate_data,pb_init,draw_Z,scale,display_W,draw_Z_tree,log_data_zw,construct_data 
import math

# Compute the elbo 
def compute_elbo( data_set , alpha , sigma_a , sigma_n , phi , Phi , nu , tau ):

    return elbo     

# Compute q and Elogstick variables for a given feature index
def compute_q_Elogstick( tau , feature_index ):

    # return
    return qk , Elogstick
    
    
def phi_update(N,K,D,X,nu,phi_mean,phi_cov,tau,sigma_a,sigma_n):
    for k in range(K):
        #phi_cov update
        sig = 1.0/(sigma_a**2) + float(nu[:,k].sum())/(sigma_n**2)
        isig = 1.0/sig
        phi_cov[k,:,:] = isig*np.eye(D)
        
        #phi_mean update
        nu_sum = np.zeros((1,D))
        for n in range(N):
            Xn = X[n,:]
            nu_phi = np.vstack([nu[n,l]*phi_mean[l,:].reshape((1,D)) for l in range(K)]).sum(axis=0) - nu[n,k]*phi_mean[k,:]
            nu_sum += nu[n,k]*(Xn - nu_phi)
        nu_sum = 1.0/(sigma_n**2) * nu_sum * isig 
        phi_mean[k,:] = nu_sum   
            
    return (phi_mean,phi_cov) 

def get_q(K,tau):
    q = np.zeros((K,K))
    for i in range(K):
        coeff = (sps.digamma(tau[i,1]) + np.array([sps.digamma(tau[m,0]) for m in range(i)]).sum()
                - np.array([sps.digamma(tau[m,0] + tau[m,1]) for m in range(i+1)]).sum())
        q[:,i] = np.repeat(np.exp(coeff),K)
    q = np.tril(q)
    for i in range(K):
        q[i,:] = 1.0/q[i,:].sum() * q[i,:]
    return q

def get_approx(tau,q,K):
    approx = np.zeros(K)
    for k in range(K):
        q_mult = q[k,:k+1]
        g_mult = np.array([sps.digamma(tau[m,1]) for m in range(k+1)])
        term1 = np.dot(q_mult,g_mult)
        q_mult = np.array([q[k,m+1:k+1].sum() for m in range(k)])
        g_mult = np.array([sps.digamma(tau[m,0]) for m in range(k)])
        term2 = np.dot(q_mult, g_mult)
        q_mult = np.array([q[k,m:k+1].sum() for m in range(k+1)])
        g_mult = np.array([sps.digamma(tau[m,0]+tau[m,1]) for m in range(k+1)])
        term3 =  np.dot(q_mult,g_mult)
        q_mult = q[k,:k+1]
        g_mult = np.log(q_mult)
        term4 = np.dot(q_mult,g_mult)
        approx[k] = term1 + term2 - term3 - term4
    return approx

def nu_update(approx,N,K,D,X,nu,tau,phi_mean,phi_cov,sigma_n):
    for n in range(N):
        for k in range(K):
            term1 = np.array([sps.digamma(tau[i,0]) - sps.digamma(tau[i,0] + tau[i,1]) for i in range(k)]).sum() 
            term2 = approx[k]
            term3 = 1.0/(2.0*sigma_n**2) * (np.trace(phi_cov[k,:,:]) + np.dot(phi_mean[k,:],phi_mean[k,:]))
            nu_phi = np.hstack([nu[n,l]*phi_mean[l,:].reshape((1,D)).T for l in range(K)]).sum(axis=1) - nu[n,k]*phi_mean[k,:].T
            term4 = 1.0/(sigma_n**2) * np.dot(phi_mean[k,:], X[n,:].T - nu_phi)  
            theta = term1 - term2 - term3 + term4
            nu[n,k] = 1.0/(1 + np.exp(-1.0*theta))
            #if nu[n,k] == 1.0:
            #    print('hitting boundary')
    return nu      
     
def tau_update(N,K,tau,alpha,nu,q):
    for k in range(K):
        tau[k,0] = alpha + nu[:N,k:K].sum() + np.array([(N - nu[:N,m].sum())*(q[m,k+1:m+1].sum()) for m in range(k+1,K)]).sum()
        tau[k,1] = 1 + np.array([(N - nu[:N,m].sum())*q[m,k] for m in range(k,K)]).sum()
    return tau

def marginal_likelihood(approx,alpha,D,K,N,nu,phi_mean,phi_cov,tau,sigma_a,sigma_n,X):
    term1 = np.array([np.log(alpha) + (alpha-1)*(sps.digamma(tau[k,0]) - sps.digamma(tau[k,0] + tau[k,1])) for k in range(K)]).sum()
    term2 = 0
    for k in range(K):
        for n in range(N):
            term2 += nu[n,k]*np.array([sps.digamma(tau[m,1] - sps.digamma(tau[m,0] + tau[m,1])) for m in range(k)]).sum()
            + (1 - nu[n,k])*approx[k]
    term3 = 0
    for k in range(K):
        term3 += -float(D)/2*np.log(2*np.pi*sigma_a**2) - 1.0/(2*sigma_a**2)*(np.trace(phi_cov[k,:,:]) + np.dot(phi_mean[k,:],phi_mean[k,:]))
    term4 = 0
    for n in range(N):
        Xn = X[n,:]
        sub1 = 2* np.array([nu[n,k]*np.dot(phi_mean[k,:],Xn) for k in range(K)]).sum()
        sub2 = 2* sum(map(sum,[[nu[n,k]*nu[n,k_prime]*np.dot(phi_mean[k,:],phi_mean[k_prime,:]) for k in range(k_prime)] for k_prime in range(K)]))
        sub3 = np.array([nu[n,k]*(np.trace(phi_cov[k,:,:]) + np.dot(phi_mean[k,:],phi_mean[k,:])) for k in range(K)]).sum()
        term4 += -float(D)/2 * np.log(2*np.pi*sigma_n**2) - 1.0/(2*sigma_n**2) * (np.dot(Xn,Xn) - sub1 + sub2 + sub3)
    term5 = 0
    for k in range(K):
        term5 += np.log(sps.beta(tau[k,0],tau[k,1])) 
        - (tau[k,0] - 1)*sps.digamma(tau[k,0]) - (tau[k,1] - 1)*sps.digamma(tau[k,1]) 
        + (tau[k,0] + tau[k,1] - 2)*sps.digamma(tau[k,0] + tau[k,1])
    sub1 = 1.0/2 * np.log((2*np.pi*np.e)**D*np.linalg.det(phi_cov[k,:,:]))
    sub2_matrix = np.zeros((N,K))
    for n in range(N):
        for k in range(K):
            if nu[n,k] == 1.0:
                sub2_matrix[n,k] = 0.0
            else:
                sub2 = np.array([[-nu[n,k]*np.log(nu[n,k]) - (1 - nu[n,k])*np.log(1 - nu[n,k]) for k in range(K)] for n in range(N)]).sum()
    sub2 = sub2_matrix.sum()
    term6 = sub1 + sub2
    return term1 + term2 + term3 + term4 + term5 + term6
    
#def log_data_zw(Y,Z,W,sig):
#    delta = Y - np.dot(Z,W)
#    if len(Y.shape) == 1:
#        delta_sum = np.dot(delta,delta)   
#    else:
#        delta_sum = np.trace(np.dot(delta.T,delta))
#    ll =  -1./(2*sig**2) * delta_sum
#    return ll

def nu_to_z(nu):
    #print("This is nu")
    #print(nu)
    N,T = nu.shape
    Z = np.zeros((N,T))
    for i in range(N):
        for j in range(T):
            if nu[i,j] > 0.5:
                Z[i,j] = 1.0
            else:
                Z[i,j] = 0.0
    return Z
    
def Z_posterior(z_row, Z):
    N,K = Z.shape
    Z_post = 1
    z_prob = 1./(N+1) * np.sum(Z,axis=0)
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
    log_pred = 0
    for i in range(R):
        pred_row = 0
        for j in range(2**K):
            binary = list(map(int,"{0:b}".format(j)))
            pad_binary = [0]*(K-len(binary)) + binary
            log_z_post = Z_posterior(pad_binary,Z)
            total_z = np.array(pad_binary)
#            print("cluster conditional")
#            print(log_data_zw(held[i,:],total_z,W,sig))
#            print("cluster weight")
#            print(log_z_post)
            #term = np.exp(log_data_zw(held[i,:],total_z,W,sig) + log_z_post)
            term = np.exp(log_data_zw(held[i,:],total_z,W,sig) + log_z_post)
            pred_row = pred_row + term
        log_pred = log_pred + np.log(pred_row)
    return log_pred
    
    #return 0
# Run the VI  
def run_vi(data_set,held_out, alpha , sigma_a, sigma_n, iter_count, feature_count):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    X = data_set
    N = data_count
    D = dim_count
    K = 15
    #elbo_set = np.zeros( [ iter_count ] )
    pred_ll = list() #predictive log likelihood
    nu_set = list()   # nu are the varitional parameters on Z 
    phi_set = list()  # phi mean param of A 
    Phi_set = list()  # Phi cov param of A, per feat -> same for all dims 
    tau_set = list()  # tau are the variational parameters on the stick betas
    
    # Initialize objects 
    nu = SPST.uniform.rvs(0,1,size = (N,K))
    phi_mean = np.zeros((K,D))
    phi_cov = np.stack([np.dot(sigma_a**2,np.eye(D)) for _ in range(K)],axis=0)
    tau = np.ones((K,2))

    # Optimization loop 
    for vi_iter in range( iter_count ):
        print(vi_iter)
        # Update Phi and phi
        phi_mean,phi_cov = phi_update(N,K,D,X,nu,phi_mean,phi_cov,tau,sigma_a,sigma_n)
        # Get the intermediate variables
        q = get_q(K,tau)
        approx = get_approx(tau,q,K)
        # Update tau, nu
        nu =  nu_update(approx,N,K,D,X,nu,tau,phi_mean,phi_cov,sigma_n)
        tau = tau_update(N,K,tau,alpha,nu,q)
        #elbo = marginal_likelihood(approx,alpha,D,K,N,nu,phi_mean,phi_cov,tau,sigma_a,sigma_n,X)
        nu_set.append( nu )
        phi_set.append( phi_mean )
        Phi_set.append( phi_cov ) 
        tau_set.append( tau )
        Z = nu_to_z(nu)
        if vi_iter%50 == 0 and vi_iter > 0:
            pred_ll.append(pred_ll_IBP(held_out, Z, phi_mean,sigma_n))
            print(pred_ll)
    return nu_set , phi_set , Phi_set , tau_set, pred_ll    

    
if __name__ == "__main__":
    iterate = 1000
    alpha = 2
    data_count = 500
    held_out = 50
    sig = 0.1
    sig_w = 0.3
    small_x = 3
    small_y = 3
    big_x = 3
    big_y = 3
    feature_count = big_x*big_y
    T = small_x*small_y*big_x*big_y
    data_type = 'random'
    #full_data,Z_gen = generate_data(feature_count,data_count + held_out,T,sig,data_type)
    full_data,Z_gen = construct_data(small_x,small_y,big_x,big_y,data_count + held_out,sig,data_type,corr_value=2)
    Y = full_data[:data_count,:]
    held_out = full_data[data_count:,:]
    sig_alg = 0.1
    nu_set,phi_set,Phi_set,tau_set,pred_ll = run_vi(Y,held_out,alpha,sig_w,sig_alg,iterate,feature_count)
    W = phi_set[iterate-1]
    display_W(W,small_x,small_y,big_x,big_y,'nine')
    print(pred_ll)
    #A = phi_set[34]
#    print(elbo_set)
#    nu = nu_set[iterate-1]
#    print(nu)
#    print(Phi_set[iterate-1])
#    N,K  = nu.shape
#    nu_var = np.zeros((N,K))
#    for n in range(N):
#        for k in range(K):
#            nu_var[n,k] = nu[n,k]*(1-nu[n,k])
#    variance = nu_var.sum()
#    print(variance)

