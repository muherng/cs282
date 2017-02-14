#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random



class ARS():


	def __init__(self, f, fprima,xi=[0,1,4], lb=-np.Inf, ub=np.Inf, use_lower=False, ns=50, **fargs):


		self.lb = lb
		self.ub = ub
		self.f = f
		self.fprima = fprima
		self.fargs = fargs

        #set limit on how many points to maintain on hull
		self.ns = 50
		self.x = np.array(xi) # initialize x, the vector of absicassae at which the function h has been evaluated
		self.h = self.f(self.x, **self.fargs)
		self.hprime = self.fprima(self.x, **self.fargs)

        #Avoid under/overflow errors. the envelope and pdf are only
        # proporitional to the true pdf, so can choose any constant of proportionality.
		self.offset = np.amax(self.h)
		self.h = self.h-self.offset 

        # Derivative at first point in xi must be > 0
        # Derivative at last point in xi must be < 0
		if not(self.hprime[0] > 0): raise IOError('initial anchor points must span mode of PDF')
		if not(self.hprime[-1] < 0): raise IOError('initial anchor points must span mode of PDF')
		self.insert() 


	def draw(self, N):
		samples = np.zeros(N)
		n=0
		while n < N:
			[xt,i] = self.sampleUpper()
			ht = self.f(xt, **self.fargs)
			hprimet = self.fprima(xt, **self.fargs)
			ht = ht - self.offset
			ut = self.h[i] + (xt-self.x[i])*self.hprime[i]

            # Accept sample? - Currently don't use lower
			u = random.random()
			if u < np.exp(ht-ut):
				samples[n] = xt
				n +=1

            # Update hull with new function evaluations
			if self.u.__len__() < self.ns:
				self.insert([xt],[ht],[hprimet])

		return samples


	def insert(self,xnew=[],hnew=[],hprimenew=[]):
        
		if xnew.__len__() > 0:
			x = np.hstack([self.x,xnew])
			idx = np.argsort(x)
			self.x = x[idx]
			self.h = np.hstack([self.h, hnew])[idx]
			self.hprime = np.hstack([self.hprime, hprimenew])[idx]

		self.z = np.zeros(self.x.__len__()+1)
		self.z[1:-1] = (np.diff(self.h) - np.diff(self.x*self.hprime))/-np.diff(self.hprime) 

		self.z[0] = self.lb; self.z[-1] = self.ub
		N = self.h.__len__()
		self.u = self.hprime[[0]+range(N)]*(self.z-self.x[[0]+range(N)]) + self.h[[0]+range(N)]

		self.s = np.hstack([0,np.cumsum(np.diff(np.exp(self.u))/self.hprime)])
		self.cu = self.s[-1]


	def sampleUpper(self):
        	u = random.random()

        # Find the largest z such that sc(z) < u
        	i = np.nonzero(self.s/self.cu < u)[0][-1] 

        # Figure out x from inverse cdf in relevant sector
        	xt = self.x[i] + (-self.h[i] + np.log(self.hprime[i]*(self.cu*u - self.s[i]) + 
        	np.exp(self.u[i]))) / self.hprime[i]

        	return [xt,i]



def indicator (x,y):
	if x >=0 and x <= y:
		return 1
	return 0

def f(xi,mu,k,N):
    ans = 1
    for i in range(N):
        ans = ans + 1/(float(i+1)) * (1-mu[k])**(i+1) 
    ans = np.exp(a* ans)
    ans = ans* mu[k]**(alpha - 1) * (1-mu[k])**N * indicator(mu[k], mu[k-1])
    return ans
 
def fprima (xi,mu,k,N):
	tmp = 1
	for i in range(N):
		tmp = tmp + 1/(float(i+1)) * (1-mu[k])**(i+1)
	tmp = np.exp(alpha*tmp) 

	term1 = tmp * ( (alpha-1)*mu[k]**(alpha-2) * (1-mu[k])**N - mu[k]**(alpha-1) *N*(1-mu[k])**(N-1)   )

	term2 = 1
	for i in range(N):
		term2 = term2 +   (1-mu[k])**i
	term2 = alpha * term2 * np.exp(tmp )

	ans = term1 + term2

	return ans



if __name__ == "__main__":

    mu = [1,2,3,3]
    N= 4
    k= 3
    ars = ARS (f, fprima, xi = [-4,1,40 ], lb=-np.Inf , mu = [1,2,3,3] , k=2 , N=4) 
    samples = ars.drawn(1)
	
	#x = np.linspace(-100,100,100)
	#ars = ARS(f, fprima, xi = [-4,1,40], mu=2, sigma=3)
	#samples = ars.draw(10000)
	#plt.hist(samples, bins=100, normed=True)
	#plt.show()


