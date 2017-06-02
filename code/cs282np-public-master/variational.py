# This is a very basic file for variational inference, taken from the
# tech report.  
import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 

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
        q_mult = q[k,:k]
        g_mult = np.array([sps.digamma(tau[m,1]) for m in range(k)])
        term1 = np.dot(q_mult,g_mult)
        q_mult = np.array([q[k,m+1:k].sum() for m in range(k-1)])
        g_mult = np.array([sps.digamma(tau[m,0]) for m in range(k-1)])
        term2 = np.dot(q_mult, g_mult)
        q_mult = np.array([q[k,m:k].sum() for m in range(k)])
        g_mult = np.array([sps.digamma(tau[m,0]+tau[m,1]) for m in range(k)])
        term3 =  np.dot(q_mult,g_mult)
        q_mult = q[k,:k]
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
# Run the VI  
def run_vi( data_set , alpha , sigma_a=5 , sigma_n=.1 , iter_count=1000 , feature_count=4 ):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    X = data_set
    N = data_count
    D = dim_count
    K = feature_count
    elbo_set = np.zeros( [ iter_count ] )
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
        #print(approx.shape)
        # Update tau, nu
        nu =  nu_update(approx,N,K,D,X,nu,tau,phi_mean,phi_cov,sigma_n)
        tau = tau_update(N,K,tau,alpha,nu,q)
        elbo = marginal_likelihood(approx,alpha,D,K,N,nu,phi_mean,phi_cov,tau,sigma_a,sigma_n,X)
        #for k in range( int( feature_count ) ):
            
            # update nu_k
            # update tau
        # Compute the ELBO
        #elbo = compute_elbo( data_set , alpha , sigma_a , sigma_n , phi , Phi , nu , tau )

        # Store things and report
        #print(elbo)
        elbo_set[ vi_iter ] = elbo 
        nu_set.append( nu )
        phi_set.append( phi_mean )
        Phi_set.append( phi_cov ) 
        tau_set.append( tau )
        #print vi_iter , elbo 

    # return
    return nu_set , phi_set , Phi_set , tau_set , elbo_set    

    

if __name__ == "__main__":
    from make_toy_data import generate_data
    iterate = 20
    alpha = 2
    data_count = 50
    data_set , Z , A = generate_data( data_count , 'infinite-random' )
    nu_set,phi_set,Phi_set,tau_set,elbo_set = run_vi(data_set,alpha,iter_count = iterate)
    #A = phi_set[34]
    print(elbo_set)
    nu = nu_set[iterate-1]
    print(nu)
    print(Phi_set[iterate-1])
    N,K  = nu.shape
    nu_var = np.zeros((N,K))
    for n in range(N):
        for k in range(K):
            nu_var[n,k] = nu[n,k]*(1-nu[n,k])
    variance = nu_var.sum()
    print(variance)

