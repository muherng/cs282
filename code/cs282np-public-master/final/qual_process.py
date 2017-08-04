import numpy as np
from ast import literal_eval as make_tuple
import operator
import matplotlib.pyplot as plt
import scipy
from matplotlib import pylab
#path = '/Users/morrisyau/yhdir/df'
#path = '/Users/morrisyau/Documents/paintbox/cs282/code/cs282np-public-master/final/d3_files/d3_runs/df'
path = '/Users/morrisyau/qtdir/q1'
runs = 100
#points = 20
init = 1


param = [.001]
held = [.7]
for par in param:
    for he in held:
        for run in range(1,runs+1):
            file = open(path + str(par) + str(he) + '_' + str(run) + '.out','r')
            data =  file.readlines()
            IBP_data = []
            paintbox_data = []
            flag = -1
            for i in range(len(data)):
                val = data[i][:-1]
                #print(val)
                if flag == 1:
                    paintbox_data.append(make_tuple(val)) 
                    #print(val)
                    #print(make_tuple(val))
                if val == 'paintbox':
                    #print('flag set to one')
                    flag = 1
                if flag == 0:
                    IBP_data.append(make_tuple(val))
                    #print(val)
                    #print(make_tuple(val))
                if val == 'IBP':
                    #print('flag set to zero')
                    flag = 0
            if init:
                libp = len(IBP_data)
                lp = len(paintbox_data)
                IBP_full = np.zeros((runs,libp))
                paintbox_full = np.zeros((runs,lp))
                IBP_time = np.zeros((runs,libp))
                paintbox_time = np.zeros((runs,lp))
                init = 0
            try:
                IBP_t,IBP_ll = zip(*IBP_data)
            except ValueError:
                print('ERROR HANDLED')
                continue
            paintbox_t,paintbox_ll = zip(*paintbox_data)
            IBP_time[run-1,:] = np.array(IBP_t)
            IBP_full[run-1,:] = np.array(IBP_ll)
            paintbox_time[run-1,:] = np.array(paintbox_t)
            paintbox_full[run-1,:] = np.array(paintbox_ll)
            #print((IBP_ll[len(IBP_ll)-1],paintbox_ll[len(paintbox_ll)-1]))
        #print("CODE RUNNING")   
        interval = 1
        restart = int(runs/interval)
        IBP_trunc_ll = np.zeros((restart,libp))
        IBP_trunc_time = np.zeros((restart,libp))
        it = 0
        for row in range(interval,len(IBP_full)+1,interval):
            index, value = max(enumerate(IBP_full[row-interval:row,libp-1]), key=operator.itemgetter(1))
            IBP_trunc_ll[it,:] = np.array(IBP_full[row - interval + index,:])
            IBP_trunc_time[it,:] = np.array(IBP_time[row - interval + index,:])
            it = it + 1
        error_IBP = [np.std(IBP_trunc_ll[:,i])/np.sqrt(restart) for i in range(libp)]
        IBP_avg = 1./restart*np.sum(IBP_trunc_ll,axis=0) 
        IBP_avg_time = 1./restart*np.sum(IBP_trunc_time,axis=0)
        print(str(par) + str(he) + ": IBP")
        print(IBP_avg) 
        print(error_IBP)
        p_trunc_ll = np.zeros((restart,lp))
        p_trunc_time = np.zeros((restart,lp))
        it = 0
        for row in range(interval,len(paintbox_full)+1,interval):
            index, value = max(enumerate(paintbox_full[row-interval:row,lp-1]), key=operator.itemgetter(1))
            p_trunc_ll[it,:] = np.array(paintbox_full[row - interval + index,:])
            p_trunc_time[it,:] = np.array(paintbox_time[row - interval + index,:])
            it = it + 1
        error_paintbox = [np.std(p_trunc_ll[:,i])/np.sqrt(restart) for i in range(lp)]
        paintbox_avg = 1./restart*np.sum(p_trunc_ll,axis=0) 
        paintbox_avg_time = 1./restart*np.sum(p_trunc_time,axis=0)
        
        print(str(par) + str(he) + ': Paintbox')
        print(paintbox_avg)
        print(error_paintbox)
        print('\n\n\n')
        
        start = 6
        last = 32
        f1 = IBP_avg_time[start:last]
        l1 = IBP_avg[start:last]
        l1.sort()
        l1_error = error_IBP[start:last]
        
        start2 = 4
        last2 = 50
        f2 = paintbox_avg_time[start2:last2]
        l2 = paintbox_avg[start2:last2]
        l2.sort()
        l2_error = error_paintbox[start2:last2]
        f2 = [f2[2*i] for i in range(len(f2)/2)]
        l2 = [l2[2*i] for i in range(len(l2)/2)]
        l2_error = [l2_error[2*i] for i in range(len(l2_error)/2)]
        
              

        p1 = plt.errorbar(f1,l1,marker='o',linestyle='--',color='red',yerr=l1_error,label = 'IBP')
        p2 = plt.errorbar(f2,l2,marker='o',linestyle='--',color='blue',yerr=l2_error, label = 'paintbox')
        plt.legend(handles=[p1, p2],loc='lower right')
        plt.title("reconstructed log likelihood vs. time: United Nations")
        plt.xlabel("time")
        plt.ylabel("log likelihood")
        plt.show()