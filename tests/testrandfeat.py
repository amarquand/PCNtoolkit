import sys
import numpy as np
import torch
from matplotlib import pyplot as plt

# load as a module
sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
from gp import GPR, CovSqExp, CovSqExpARD, CovLin
from bayesreg import BLR

def plot_dist(x, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x,mean, color_mean)

def f(X):
    y = -0.1*X+0.02*X**2+np.sin(0.9*X)
    return 10*y

N = 500
sn2 = 2
X = np.random.uniform(-10,10,N)
Xs = np.linspace(-10,10,100)
#Xs = np.atleast_2d(np.linspace(-10,10,100)).T

if len(X.shape) < 2:
    X = X[:, np.newaxis]
if len(Xs.shape) < 2:
    Xs = Xs[:, np.newaxis]    
    
y = f(X) + np.random.normal(0,sn2,X.shape)
ys = f(Xs)

cov = CovSqExp(X)
ell2 = 1
sf2 = 1
hyp0 = [np.log(0.5*sf2), np.log(0.5*ell2), np.log(0.5*sf2)]

#cov = CovSqExpARD(X)
#hyp0 = np.zeros(X.shape[1]+2)

G = GPR(hyp0, cov, X, y)
hyp = G.estimate(hyp0,cov, X, y)
yhat,s2 = G.predict(hyp,X,y,Xs)

s2 = np.diag(s2)

[sn2_est, ell2_est, sf2_est] = np.exp(2*hyp)
#s2 = s2 + sn2_est

plt.plot(Xs,ys,'r')
plt.plot(X,y,'xr')
plt.plot(Xs,yhat,'b')
plot_dist(Xs.ravel(), yhat.ravel(), 
          yhat.ravel()-2*np.sqrt(s2).ravel(),
          yhat.ravel()+2*np.sqrt(s2).ravel(),'b','b')

# Random feature approximation
Nf = 300
Omega = np.zeros((1,Nf))
for f in range(Nf):
    Omega[:,f] = np.random.normal(0, ell2_est, (Omega.shape[0], 1))

XO = X.dot(Omega)
Phi = np.sqrt(sn2_est/Nf)*np.c_[np.cos(XO),np.sin(XO)]
XsO = Xs.dot(Omega)
Phis = np.sqrt(sn2_est/Nf)*np.c_[np.cos(XsO),np.sin(XsO)]

hyp_blr = np.asarray([np.log(1/sn2_est), np.log(1/sf2_est)])
B = BLR(hyp_blr, Phi, y)
yhat_blr, s2_blr = B.predict(hyp_blr, Phi, y, Phis)
#s2_blr = np.diag(s2_blr)

Omega = torch.zeros((1,Nf), dtype=torch.double)
for f in range(Nf):
    Omega[:,f] = torch.sqrt(torch.tensor(ell2_est)) * \
                 torch.randn((Omega.shape[0], 1))

XO = torch.mm(torch.from_numpy(X), Omega) 

torch.sqrt(sn2_est/Nf) * torch.cat([torch.cos(XO), torch.sin(XO)])

plot_dist(Xs.ravel(), yhat_blr.ravel(), 
          yhat_blr.ravel() - 2*np.sqrt(s2_blr).ravel(),
          yhat_blr.ravel() + 2*np.sqrt(s2_blr).ravel(),'y','y')
