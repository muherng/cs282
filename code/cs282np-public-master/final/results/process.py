import numpy as np
from ast import literal_eval as make_tuple
import operator

path = '/Users/morrisyau/data_one/yrpaintbox3_'

#IBP_full = np.zeros((30,6))
#paintbox_full= np.zeros((30,10))
IBP_full = np.zeros((30,6))
paintbox_full = np.zeros((30,10))
IBP_time = np.zeros((30,6))
paintbox_time = np.zeros((30,10))
for run in range(1,31):
    file = open(path + str(run) + '.out','r')
    data =  file.readlines()

    #file = open('qpaintbox3_7.out','r')
    #data =  file.readlines()
    IBP_data = []
    paintbox_data = []
    flag = -1
    for i in range(len(data)):
        val = data[i][:-1]
        print(val)
        if flag == 1:
            paintbox_data.append(make_tuple(val)) 
            print(val)
            print(make_tuple(val))
        if val == 'paintbox':
            print('flag set to one')
            flag = 1
        if flag == 0:
            IBP_data.append(make_tuple(val))
            print(val)
            print(make_tuple(val))
        if val == 'IBP':
            print('flag set to zero')
            flag = 0
    IBP_t,IBP_ll = zip(*IBP_data)
    paintbox_t,paintbox_ll = zip(*paintbox_data)
    IBP_time[run-1,:] = np.array(IBP_t)
    IBP_full[run-1,:] = np.array(IBP_ll)
    paintbox_time[run-1,:] = np.array(paintbox_t)
    paintbox_full[run-1,:] = np.array(paintbox_ll)
print("CODE RUNNING")   

IBP_trunc_ll = np.zeros((10,6))
IBP_trunc_time = np.zeros((10,6))
it = 0
for row in range(3,len(IBP_full)+1,3):
    index, value = max(enumerate(IBP_full[row-3:row,5]), key=operator.itemgetter(1))
    IBP_trunc_ll[it,:] = np.array(IBP_full[row - 3 + index,:])
    it = it + 1

IBP_avg = 1./10*np.sum(IBP_trunc_ll,axis=0) 
IBP_avg_time = 1./10*np.sum(IBP_trunc_ll,axis=0)
print("IBP Averages")
print(IBP_avg) 

p_trunc_ll = np.zeros((10,10))
p_trunc_time = np.zeros((10,10))
it = 0
for row in range(3,len(paintbox_full)+1,3):
    index, value = max(enumerate(paintbox_full[row-3:row,5]), key=operator.itemgetter(1))
    p_trunc_ll[it,:] = np.array(paintbox_full[row - 3 + index,:])
    it = it + 1

paintbox_avg = 1./10*np.sum(p_trunc_ll,axis=0) 
paintbox_avg_time = 1./10*np.sum(p_trunc_ll,axis=0)
print('Paintbox Averages')
print(paintbox_avg)   
