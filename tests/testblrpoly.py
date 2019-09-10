# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

#sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
from bayesreg import BLR
from gp import GPR, CovSqExp

X = np.arange(0,10,0.1)
N = len(X)
dimpoly = 3

# this also works but is less like matlab (transposed)
#Phi = np.zeros((dimpoly,X.shape[0]))
#colid = np.arange(0,1)
#for d in range(1,dimpoly+1):
#    print colid
#    Phi[colid,:] = X ** d
#    colid += 1

# generate some data
Phi = np.zeros((X.shape[0],dimpoly))
colid = np.arange(0,1)
for d in range(1,dimpoly+1):
    Phi[:,colid] = np.vstack(X ** d)
    colid += 1

b = [0.5, -0.1, 0.005] # true regression coefficients
s2 = 0.05 # noise variance

y = Phi.dot(b) + np.random.normal(0,s2,N)

yid = range(0,N,1)

hyp0 = np.zeros(2)
hyp0 = np.zeros(4)
B = BLR(hyp0,Phi[yid,:],y[yid])
#B = BLR()
#B.post(hyp0,Phi[yid,:],y[yid])
B.loglik(hyp0,Phi[yid,:],y[yid])
B.dloglik(hyp0,Phi[yid,:],y[yid])
hyp = B.estimate(hyp0,Phi[yid,:],y[yid])

yhat,s2 = B.predict(hyp,Phi[yid,:],y[yid],Phi)

plt.plot(X,y)
plt.plot(X,yhat)
plt.show()

## test GPR
#y = y-y.mean()
#hyp0 = np.zeros(3)
#G = GPR(hyp0, covSqExp, Phi[yid,:], y[yid])
#G.loglik(hyp0, covSqExp, Phi[yid,:], y[yid])
#G.dloglik(hyp0, covSqExp, Phi[yid,:], y[yid])
#hyp = G.estimate(hyp0,covSqExp, Phi[yid,:], y[yid])
#
#yhat,s2 = G.predict(hyp0,Phi[yid,:],y[yid],Phi)
#
#plt.plot(X,y)
#plt.plot(X,yhat)
#plt.show()
##sp.optimize.check_grad(B.loglik, B.dloglik, np.array([0, 0]), Phi,y)
#
##out = sp.optimize.fmin_cg(B.loglik,hyp0,B.dloglik,(Phi,y),full_output=1)
##hyp = np.exp(out[0])
vari = 1/hyp

print(B.nlZ)
print(vari)
print(B.m)