import sys
import numpy as np
import torch
from utils import create_poly_basis
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# load as a module
sys.path.append('/home/mrstats/andmar/sfw/nispat/nispat')
from gp import GPR, CovSqExp, CovSqExpARD, CovLin, CovSum
from bayesreg import BLR
from rfa import GPRRFA

def plot_dist(x, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x,mean, color_mean)

def f(X):
    y = -0.1*X+0.02*X**2+np.sin(0.9*X)    +np.cos(0.1*X)
    
    y = -X + 0.1*X**2
    
    #y = -0.1*X+0.02*X**2+0.3*np.sin(0.3*X)
    return 10*y

N = 500
sn2 = 5 #2
X = np.random.uniform(-10,10,N)
Xs = np.linspace(-10,10,100)

if len(X.shape) < 2:
    X = X[:, np.newaxis]
if len(Xs.shape) < 2:
    Xs = Xs[:, np.newaxis]    
    
y = f(X) + np.random.normal(0,sn2,X.shape)
ys = f(Xs)

my = np.mean(y, axis=0)
sy = np.std(y,  axis=0)
y = (y - my) / sy
ys = (ys - my) / sy

# add a polynomial basis expansion to make sure it works with p > 1
#X = np.c_[X, X **2]
#Xs = np.c_[Xs, Xs **2]
X = np.c_[X, np.ones((N,1)) , 0.01*np.random.randn(N,1)]
Xs = np.c_[Xs, np.ones((Xs.shape[0],1)), 0.01*np.random.randn(Xs.shape[0],1)]

#cov = CovSqExp(X)
#cov = CovSqExpARD(X)
cov = CovSum(X, ('CovLin', 'CovSqExpARD'))
hyp0 = np.zeros(cov.get_n_params() + 1)

print('running GPR')
G = GPR(hyp0, cov, X, y)
hyp = G.estimate(hyp0,cov, X, y)
yhat,s2 = G.predict(hyp,X,y,Xs)
s2 = np.diag(s2)

# extract parameters
sn2_est = np.exp(2*hyp[0])
ell2_est = np.exp(2*hyp[1:-1])
sf2_est = np.exp(2*hyp[-1])
print('sn2 =',sn2_est,'ell =',np.sqrt(ell2_est),'sf2=',sf2_est)

plt.plot(Xs[:,0],ys,'r')
plt.plot(X[:,0],y,'xr')
plt.plot(Xs[:,0],yhat,'b')
plot_dist(Xs[:,0].ravel(), yhat.ravel(), 
          yhat.ravel()-2*np.sqrt(s2).ravel(),
          yhat.ravel()+2*np.sqrt(s2).ravel(),'b','b')

print('running BLR ...')
# Random feature approximation
# ----------------------------
Nf = 250
D = X.shape[1]
Omega = np.zeros((D,Nf))
for f in range(Nf):
    Omega[:,f] = np.sqrt(ell2_est) * np.random.randn(Omega.shape[0])

XO = X.dot(Omega)
Phi = np.sqrt(sf2_est/Nf)*np.c_[np.cos(XO),np.sin(XO)]
XsO = Xs.dot(Omega)
Phis = np.sqrt(sf2_est/Nf)*np.c_[np.cos(XsO),np.sin(XsO)]

# add linear component
Phi = np.c_[Phi, X]
Phis = np.c_[Phis, Xs]

hyp_blr = np.asarray([np.log(1/sn2_est), np.log(1)])
B = BLR(hyp_blr, Phi, y)
B.loglik(hyp_blr, Phi, y)
yhat_blr, s2_blr = B.predict(hyp_blr, Phi, y, Phis)

#plt.plot(Xs[:,0],yhat_blr,'y')
#plot_dist(Xs[:,0].ravel(), yhat_blr.ravel(), 
#          yhat_blr.ravel() - 2*np.sqrt(s2_blr).ravel(),
#          yhat_blr.ravel() + 2*np.sqrt(s2_blr).ravel(),'y','y')

print('running RFA ...')
R = GPRRFA(hyp, X, y, n_feat = Nf)
# find good starting hyperparameters
lm = LinearRegression()
lm.fit(create_poly_basis(X,3), y)
yhat = lm.predict(create_poly_basis(X,3))
hyp0 = np.zeros(D + 2)
hyp0[0] = np.log(np.sqrt(np.var(y - yhat)))
hyp = R.estimate(hyp0,X,y)
yhat_rfa,s2_rfa = R.predict(hyp,X,y,Xs)

plot_dist(Xs[:,0].ravel(), yhat_rfa.ravel(), 
          yhat_rfa.ravel() - 2*np.sqrt(s2_rfa).ravel(),
          yhat_rfa.ravel() + 2*np.sqrt(s2_rfa).ravel(),'k','k')
sn2_est = np.exp(2*hyp[0])
ell2_est = np.exp(2*hyp[1:-1])
sf2_est = np.exp(2*hyp[-1])
print('sn2 =',sn2_est,'ell =',np.sqrt(ell2_est),'sf2=',sf2_est)

## Random features (torch)
## -----------------------
## init
#hyp = torch.tensor(hyp, requires_grad=True)
## hyp = [log(sn), log(ell), log(sf)]
##sn2_est = np.exp(2*hyp[0])
##ell2_est = np.exp(2*hyp[1:-1])
##sf2_est = np.exp(2*hyp[-1])
#
#Omega = torch.zeros((D,Nf), dtype=torch.double)
#for f in range(Nf):
#    Omega[:,f] = torch.exp(hyp[1:-1]) * \
#                 torch.randn((Omega.shape[0], 1), dtype=torch.double).squeeze()
#
##Omega = torch.from_numpy(Omega)
#XO = torch.mm(torch.from_numpy(X), Omega) 
#Phi = torch.exp(hyp[-1])/np.sqrt(Nf) * torch.cat((torch.cos(XO), torch.sin(XO)), 1)
## concatenate linear weights 
#Phi = torch.cat((Phi, torch.from_numpy(X)), 1)
#N, D = Phi.shape
#y = torch.from_numpy(y)
#
## post
#iSigma = torch.eye(D, dtype=torch.double)
#A = torch.mm(torch.t(Phi), Phi) / torch.exp(2*hyp[0]) + iSigma
#m = torch.mm(torch.gesv(torch.t(Phi), A)[0], y) / torch.exp(2*hyp[0])
#
## predict
#Xs = torch.from_numpy(Xs)
#XsO = torch.mm(Xs, Omega) 
#Phis = torch.exp(hyp[-1])/np.sqrt(Nf) * torch.cat((torch.cos(XsO), torch.sin(XsO)), 1)
#Phis = torch.cat((Phis, Xs), 1)
#ys = torch.mm(Phis, m)
#
##s2_blr = sn2_est + torch.diag(torch.mm(Phis, torch.gesv(torch.t(Phis), A)[0]))
## avoiding computing off-diagonal entries
#s2_blr = torch.exp(2*hyp[0]) + torch.sum(Phis * torch.t(torch.gesv(torch.t(Phis), A)[0]), 1)
#
#
#logdetA = 2*torch.sum(torch.log(torch.diag(torch.cholesky(A))))
#
## compute negative marginal log likelihood
#nlZ = -0.5 * (N*torch.log(1/torch.exp(2*hyp[0])) - N*np.log(2*np.pi) -
#              torch.mm(torch.t(y-torch.mm(Phi,m)), (y-torch.mm(Phi,m)))/torch.exp(2*hyp[0]) -
#              torch.mm(torch.t(m), m) -
#              logdetA
#              )
#nlZ.backward()
#
#plot_dist(Xs[:,0].detach().numpy().ravel(), ys.detach().numpy(), 
#          ys.detach().numpy().ravel() - 2*np.sqrt(s2_blr.detach().numpy()).ravel(),
#          ys.detach().numpy().ravel() + 2*np.sqrt(s2_blr.detach().numpy()).ravel(),'k','k')