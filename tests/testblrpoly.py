# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/preclineu/andmar/sfw/nispat/nispat')
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from bayesreg import BLR

#from gp import GPR, CovSqExp

X = np.arange(0,10,0.1)
N = len(X)
dimpoly = 3

# generate some data
Phi = np.zeros((X.shape[0],dimpoly))
colid = np.arange(0,1)
for d in range(1,dimpoly+1):
    Phi[:,colid] = np.vstack(X ** d)
    colid += 1
    
# generate test data
Xs =  np.arange(0,10,0.01)
Phis = np.zeros((Xs.shape[0],dimpoly))
colid = np.arange(0,1)
for d in range(1,dimpoly+1):
    Phis[:,colid] = np.vstack(Xs ** d)
    colid += 1

# generative model
b = [0.5, -0.1, 0.005]  # true regression coefficients
s2 = 0.05               # noise variance
y = Phi.dot(b) + np.random.normal(0,s2,N)

yid = range(0,N,1)

hyp0 = np.zeros(2)
#hyp0 = np.zeros(4) # use ARD
B = BLR(hyp0, Phi[yid,:], y[yid])#,var_groups=np.ones(N))
B.loglik(hyp0, Phi[yid,:], y[yid])
B.dloglik(hyp0, Phi[yid,:], y[yid])
hyp = B.estimate(hyp0, Phi[yid,:], y[yid])

yhat,s2 = B.predict(hyp, Phi, y, Phis)
plt.fill_between(Xs,yhat-1.96*np.sqrt(s2), yhat+1.96*np.sqrt(s2), alpha = 0.2)
plt.scatter(X,y)
plt.plot(Xs,yhat)
plt.show()

print(B.nlZ)
print(1/hyp)
print(B.m)

print("Estimating a model with heteroskedastic noise ...")
# set up some indicator variables for the variance groups
n_site = 3
idx = []
idx.append(np.where(X < 2)[0])
idx.append(np.where((X > 2) & (X < 4))[0])
idx.append(np.where(X > 4)[0])
idx_te = []
idx_te.append(np.where(Xs < 2)[0])
idx_te.append(np.where((Xs > 2) & (Xs < 4))[0])
idx_te.append(np.where(Xs > 4)[0])

# set up site indicator variables
sids = np.ones(N)
sids_te = np.ones(Xs.shape[0])
for i in range(n_site):
    sids[idx[i]] = i+1
    sids_te[idx_te[i]] = i+1
cols = ['blue','red','green']

# generative model
b0 = [-0.5, 0.25, -0.3] # intercepts
bh = [-0.025, 0.001, 0] # slopes
s2h = [0.01, 0.2, 0.1] # noise
y = Phi.dot(bh)

# add intercepts and heteroskedastic noise
for s in range(n_site):
    sidx = np.where(sids == s+1)[0]
    y[sidx] += b0[s] + np.random.normal(0,np.sqrt(s2h[s]),len(sidx))

# add the site specific intercepts to the design matrix
for s in range(n_site):
    site = np.zeros((N,1))
    site[idx[s]] = 1
    Phi = np.concatenate((Phi, site), axis=1)
    
    site_te = np.zeros((Xs.shape[0],1))
    site_te[idx_te[s]] = 1
    Phis = np.concatenate((Phis, site_te), axis=1)

hyp0=np.zeros(4)
Bh = BLR(hyp0,Phi,y,var_groups=sids)
Bh.loglik(hyp0,Phi[yid,:],y[yid])
Bh.dloglik(hyp0,Phi[yid,:],y[yid])
hyp = Bh.estimate(hyp0,Phi[yid,:],y[yid])

yhat,s2 = Bh.predict(hyp, Phi, y, Phis, var_groups_test=sids_te)

for s in range(n_site):
    plt.scatter(X[idx[s]], y[idx[s]])
    plt.plot(Xs[idx_te[s]],yhat[idx_te[s]], color=cols[s])
    plt.fill_between(Xs[idx_te[s]], 
                     yhat[idx_te[s]] - 1.96 * np.sqrt(s2[idx_te[s]]), 
                     yhat[idx_te[s]] + 1.96 * np.sqrt(s2[idx_te[s]]),
                     alpha=0.2, color=cols[s])

## run GP for comparison
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
    
# check gradients
##sp.optimize.check_grad(B.loglik, B.dloglik, np.array([0, 0]), Phi,y)
#
##out = sp.optimize.fmin_cg(B.loglik,hyp0,B.dloglik,(Phi,y),full_output=1)
##hyp = np.exp(out[0])
