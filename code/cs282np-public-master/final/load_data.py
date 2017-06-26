import numpy as np 
import os 

path = '/n/regal/doshi-velez_lab/mamasood/NMF_Data/'

#data = np.load(path + 'SVD_reconHubble_small.npz')

#X = data['X'].T

#size = []

#for filename in os.listdir(path):
#    if filename.endswith(".npz"): 
#        print(filename)
#        data = np.load(path + filename)
#        X = data['X'].T
#        print(X.shape)
#        size.append((X.shape[0]*X.shape[1], filename))
#        
#order = sorted(size, key=lambda x: x[0])
#print(order)

def load(filename):
    data = np.load(path + filename)
    X = data['X'].T
    return X

