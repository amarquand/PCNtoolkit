import sys
# sys.path.append('/home/preclineu/andmar/sfw/PCNtoolkit/pcntoolkit')
# sys.path.append('/home/preclineu/chafra/Desktop/PCNtoolkit/')

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import bspline
from bspline import splinelab
from pcntoolkit.model.bayesreg import BLR
from pcntoolkit.model.gp import GPR
from pcntoolkit.util.utils import WarpBoxCox, WarpAffine, WarpCompose, WarpSinArcsinh


def create_noise(type_noise, N, parameters=None):
    """Function to create different noise distributions"""
    if type_noise == 'exp':
        scale = parameters
        n = 2*np.random.exponential(scale, N)
    elif type_noise == 'gamma':
        shape = parameters
        n = 2*np.random.gamma(shape, scale=2, size=N)
    elif type_noise == 'skewed_right':
        gaussian_rv = np.random.normal(0, 1, N)
        n = np.concatenate((2.5*gaussian_rv[gaussian_rv > 0], np.random.normal(
            0, 1, np.abs(N-len(gaussian_rv[gaussian_rv > 0])))))
    elif type_noise == 'skewed_left':
        gaussian_rv = np.random.normal(0, 1, N)
        n = np.concatenate((2.5*gaussian_rv[gaussian_rv < 0], np.random.normal(
            0, 1, np.abs(N-len(gaussian_rv[gaussian_rv < 0])))))
    elif type_noise == 'heavy_tailed':
        n = np.concatenate((np.random.normal(0, 1, int(N/2))
                           * 2.5, np.random.normal(0, 1, int(N/2))))
    elif type_noise == 'light_tailed':
        n = np.random.normal(0, 0.6, N)
    elif type_noise == 'gaussian':
        mu = 0
        sigma = parameters
        n = np.random.normal(mu, sigma, N)
    elif type_noise == 'bimodal':
        N = int(N/2)
        x1 = 2*np.random.normal(-1, 0.25, N)
        x2 = np.random.normal(1, 0.25, N)
        n = np.concatenate([x1, x2])
    elif type_noise == 'skewed_bimodal':
        N = int(N/2)
        x1 = 2*np.random.normal(-1, 0.25, N)
        x2 = np.random.normal(1, 0.25, N)
        gaussian_rv = np.random.normal(0, 1, N)
        x2 = np.concatenate((10*gaussian_rv[gaussian_rv > 0], np.random.normal(
            0, 1, np.abs(N-len(gaussian_rv[gaussian_rv > 0])))))
        n = np.concatenate([x1, x2])
    elif type_noise == 'tdist':
        dof = parameters
        n = np.random.standard_t(dof, N)
    plt.figure()
    plt.hist(n, bins=50)
    plt.title('Noise distribution')
    return n


print('First do a simple evaluation of B-splines regression...')

# generate some data
X = np.arange(0, 10, 0.05)
Xs = np.arange(0, 10, 0.01)
N = len(X)
dimpoly = 3

# Polynomial basis (used for data generation)
Phip = np.zeros((X.shape[0], dimpoly))
colid = np.arange(0, 1)
for d in range(1, dimpoly+1):
    Phip[:, colid] = np.vstack(X ** d)
    colid += 1
Phips = np.zeros((Xs.shape[0], dimpoly))
colid = np.arange(0, 1)
for d in range(1, dimpoly+1):
    Phips[:, colid] = np.vstack(Xs ** d)
    colid += 1

# generative model
b = [0.5, -0.1, 0.005]  # true regression coefficients
s2 = 0.05               # noise variance
# y = Phip.dot(b) + create_noise('gaussian', N, s2)
y = Phip.dot(b) + create_noise('exp', N, 2)
# y = Phip.dot(b) + create_noise('skewed_right', N)
# y = Phip.dot(b) + create_noise('skewed_left', N)
# y = Phip.dot(b) + create_noise('heavy_tailed', N)
# y = Phip.dot(b) + create_noise('light_tailed', N)
# y = Phip.dot(b) + create_noise('bimodal', N)
# y = Phip.dot(b) + create_noise('skewed_bimodal', N)
# y = Phip.dot(b) + create_noise('tdist', N, 3)

# cubic B-spline basis (used for regression)
p = 3       # order of spline (3 = cubic)
nknots = 5  # number of knots (endpoints only counted once)
knots = np.linspace(0, 10, nknots)
k = splinelab.augknt(knots, p)       # pad the knot vector
B = bspline.Bspline(k, p)
Phi = np.array([B(i) for i in X])
Phis = np.array([B(i) for i in Xs])

hyp0 = np.zeros(2)
# hyp0 = np.zeros(4) # use ARD
# B = BLR(hyp0, Phi, y)
B = BLR()
hyp = B.estimate(hyp0, Phi, y, optimizer='powell')

yhat, s2 = B.predict(hyp, Phi, y, Phis)
plt.figure()
plt.fill_between(Xs, yhat-1.96*np.sqrt(s2), yhat+1.96*np.sqrt(s2), alpha=0.2)
plt.scatter(X, y)
plt.plot(Xs, yhat)
plt.show()

print(B.nlZ)
print(1/hyp)
print(B.m)

print('demonstrate likelihood warping ...')
# generative model
# b = [0.4, -0.01, 0.]  # true regression coefficients
# s2 = 0.1              # noise variance
# y = Phip.dot(b) + 2*np.random.exponential(1,N)
plt.scatter(X, y)

W = WarpBoxCox()
# W = WarpSinArcsinh()

Phix = X[:, np.newaxis]
Phixs = Xs[:, np.newaxis]

Bw = BLR(warp=W)
# hyp0 = 0.1*np.ones(2+W.get_n_params())
# hyp = Bw.estimate(hyp0, Phi, y, optimizer='powell')
# yhat, s2 = Bw.predict(hyp, Phi, y, Phis)
hyp0 = 0.1*np.ones(2+W.get_n_params())
hyp = Bw.estimate(hyp0, Phi, y, optimizer='powell', var_covariates=Phix)
yhat, s2 = Bw.predict(hyp, Phi, y, Phis, var_covariates_test=Phixs)

warp_param = hyp[1:W.get_n_params()+1]
med, pr_int = W.warp_predictions(yhat, s2, warp_param)

plt.plot(Xs, med, 'b')
plt.fill_between(Xs, pr_int[:, 0], pr_int[:, 1], alpha=0.2, color='blue')

# for the Box-Cox warp use closed form expression for the mode (+ve support)
if len(warp_param == 1):
    lam = np.exp(warp_param[0])
    mod = (0.5*(1+lam*yhat + np.sqrt((1+lam*yhat)**2 + 4*s2*lam*(lam-1))))**(1/lam)
    plt.plot(Xs, mod, 'b--')
    plt.legend(('median', 'mode'))
plt.show()

xx = np.linspace(-5, 5, 100)
plt.plot(xx, W.invf(xx, warp_param))
plt.title('estimated warping function')
plt.show()

# estimate a model with heteroskedastic noise
print('demonstrate heteroskedastic noise...')
# generative model
b = [0.4, -0.01, 0.]  # true regression coefficients
s2 = 0.1              # noise variance
y = Phip.dot(b) + Phip[:, 0]*np.random.normal(size=N)
plt.scatter(X, y)

# new version
Bh = BLR()
hyp0 = np.zeros(8)
hyp = Bh.estimate(hyp0, Phi, y, optimizer='l-bfgs-b',
                  var_covariates=Phi, verbose=True)
yhat, s2 = Bh.predict(hyp, Phi, y, Phis, var_covariates_test=Phis)

# old version
# Bh = BLR(hetero_noise=7)
# hyp0 = np.zeros(8)
# hyp = Bh.estimate(hyp0, Phi, y, optimizer='l-bfgs-b', hetero_noise=7, verbose=True)
# yhat,s2 = Bh.predict(hyp, Phi, y, Phis)

print(hyp)
plt.fill_between(Xs, yhat-1.96*np.sqrt(s2), yhat+1.96*np.sqrt(s2), alpha=0.2)
plt.show()

print("Estimate a model with site-specific noise ...")
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
cols = ['blue', 'red', 'green']

# generative model
b0 = [-0.5, 0.25, -0.3]  # intercepts
bh = [-0.025, 0.001, 0]  # slopes
s2h = [0.01, 0.2, 0.1]  # noise
y = Phip.dot(bh)

# add intercepts and heteroskedastic noise
for s in range(n_site):
    sidx = np.where(sids == s+1)[0]
    y[sidx] += b0[s] + np.random.normal(0, np.sqrt(s2h[s]), len(sidx))

# add the site specific intercepts to the design matrix
for s in range(n_site):
    site = np.zeros((N, 1))
    site[idx[s]] = 1
    Phi = np.concatenate((Phi, site), axis=1)

    site_te = np.zeros((Xs.shape[0], 1))
    site_te[idx_te[s]] = 1
    Phis = np.concatenate((Phis, site_te), axis=1)

hyp0 = np.zeros(4)
Bh = BLR(var_groups=sids)
Bh.loglik(hyp0, Phi, y)
Bh.dloglik(hyp0, Phi, y)
hyp = Bh.estimate(hyp0, Phi, y)

yhat, s2 = Bh.predict(hyp, Phi, y, Phis, var_groups_test=sids_te)

for s in range(n_site):
    plt.scatter(X[idx[s]], y[idx[s]])
    plt.plot(Xs[idx_te[s]], yhat[idx_te[s]], color=cols[s])
    plt.fill_between(Xs[idx_te[s]],
                     yhat[idx_te[s]] - 1.96 * np.sqrt(s2[idx_te[s]]),
                     yhat[idx_te[s]] + 1.96 * np.sqrt(s2[idx_te[s]]),
                     alpha=0.2, color=cols[s])
plt.show()
