# Imports 
from make_toy_data import generate_data 
from mcmc_mh import cgibbs_sampler , ugibbs_sampler , slice_sampler 
from variational import run_vi
import matplotlib.pyplot as plt 
import numpy as np 
import pdb 

# ---- Set up ---- #
# Generate data from the toy G&G blocks (easy to visualize) 
data_count = 20
data_type = 'infinite-random' 
data_set , Z , A = generate_data( data_count , data_type )


# Inference Parameters (note, these do not necessarily match the true parameters!) 
alpha = 2.0  

# ---- Inference ---- #
iterate = 15
runs = 10
c_data = np.zeros((runs,iterate))
u_data = np.zeros((runs,iterate))
s_data = np.zeros((runs,iterate))
v_data = np.zeros((runs,iterate))

for i in range(runs):
    print('cgibbs')
    _,_,ll_c,_ = cgibbs_sampler( data_set , alpha, iter_count = iterate) 
    c_data[i,:] = ll_c
    print('ugibbs')
    Z_set , A_set , ll_u , lp_set = ugibbs_sampler( data_set , alpha, iter_count = iterate)
    u_data[i,:] = ll_u
    print('slice')
    Z_set , A_set , ll_s , lp_set = slice_sampler( data_set , alpha, iter_count = iterate)
    s_data[i,:] = ll_s
    print('vi')
    nu_set , phi_set , Phi_set , tau_set , ll_v = run_vi( data_set , alpha, iter_count  = iterate)
    v_data[i,:] = ll_v

c = c_data.sum(axis=0)/runs
u = u_data.sum(axis=0)/runs
v = v_data.sum(axis=0)/runs
s = s_data.sum(axis=0)/runs

#will change as we plot over time
x = [i for i in range(1,iterate+1)]
data_x = [x,x,x,x]
data_y = [c,u,v,s]
legend = ['collapsed','uncollapsed','variational','slice']
def plot(title,x_axis,y_axis,data_x,data_y):
    for i in range(len(data_x)):
        plt.plot(data_x[i],data_y[i],label = legend[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

title = 'log likelihood vs iterations'
x_axis = 'iterations'
y_axis = 'predictive log likelihood'
plot(title,x_axis,y_axis,data_x,data_y)
    
    
    
    
