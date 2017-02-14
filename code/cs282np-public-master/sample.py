#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.special as sps
import scipy.stats as SPST



def sample(m , alpha, iterations):

	N = len(m)
	pts = [0 for i in range(iterations)]
	x = 0
	for j  in range(iterations):
		x +=  float(m[N-1]) / iterations
		tmp = 0
		for i in range(N):
			tmp = tmp + 1/float(i+1) * (1-x)**i
	
		tmp = np.exp(alpha *tmp)
		tmp = tmp * (1-x)**(alpha-1) * (1-x)**N
		pts[j] = x


	summer = [0 for i in range(iterations)]
	area = [0 for i in range(iterations)]
	
	summer[0] = pts[0]
	for j in range(0,iterations):
		summer[j] = summer[j-1] + pts[j]
	for j in range(0,iterations-1):
		area[j] = (summer[j+1] - summer[j])* float(m[N-1])/ iterations

	soum = sum(area)
	for i in range(iterations):
		area[i] = area[i] / float(soum)	
	num_new, = np.where(np.random.multinomial(1,area) == 1)[0]
	
	#print num_new
	#print pts[num_new]
	return pts[num_new]

if __name__ == "__main__":

	print sample( [1,3,4,3,5,6], alpha = 1, iterations = 10000)

	
 
