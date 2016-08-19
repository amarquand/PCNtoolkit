from __future__ import print_function

import numpy as np
import scipy as sp

class blr:
    """
    Bayesian linear regression
    
    Estimation and prediction of Bayesian linear regression models
    Basic usage: 
    
        >> B = blr()
        >> hyp = B.estimate(hyp0, X, t)
        >> ts,s2 = B.predict(hyp, X, t, Xs)   
    
    where the variables are:
        hyp  : vector of hyperparmaters. hyp = [log(beta); log(alpha)]
        hyp0 : starting estimates for hyperparameter optimisation
        X    : N x D data array
        t    : 1D Array of targets (length N)
        Xs   : Nte x D array of test cases
        ts   : predictive mean
        s2   : predictive variance

    The implementation and notation mostly follows Bishop (2006). 
    The hyperparameter beta is the noise precision and alpha is the precision
    over lengthscale parameters. This can be either a scalar variable (a
    common lengthscale for all input variables), or a vector of length D (a
    different lengthscale for each input variable, derived using an automatic
    relevance determination formulation). These are estimated using conjugate 
    gradient optimisation of the marginal likelihood. 
    
    Reference:
    Bishop (2006) Pattern Recognition and Machine Learning, Springer
    
    Written by A. Marquand
    """
    
    def __init__(self, n_iter=1000, tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan       
        self.tol = tol         # not used at present
        self.n_iter = n_iter   # not used at present
        self.verbose = verbose # not used at present
        
    # ------------------- Compute posterior distribution ----------------------
    def post(self, hyp, X, t):
        
        N = X.shape[0]
        if len(X.shape) == 1:
            D = 1
        else:
            D = X.shape[1]
        
        if (hyp == self.hyp).all() and hasattr(self, 'N'):
            print("hyperparameters have not changed, exiting")
            return
            
        # hyperparameters
        beta = np.exp(hyp[0])   # noise precision
        alpha = np.exp(hyp[1:]) # precision for the coefficients
        
        if self.verbose:
            print("estimating posterior ... | hyp=",hyp)
        
        # prior variance
        if len(alpha) == 1 or len(alpha) == D:
            Sigma = np.diag(np.ones(D))/alpha
            iSigma = np.diag(np.ones(D))*alpha
        else:
            print("hyperparameter vector has invalid length")
            raise     
                
        # compute posterior
        A = beta*X.T.dot(X) + iSigma;               # posterior precision       
        m = beta*sp.linalg.solve(A,X.T).dot(t)           # posterior mean
            
        # save stuff
        self.N = N
        self.D = D
        self.m = m
        self.A = A
        self.hyp = hyp
        self.Sigma = Sigma
        self.iSigma = iSigma

    # ------------------- Compute (marginal) likelihood -----------------------
    def loglik(self, hyp, X, t):
            
        # hyperparameters (only beta needed)
        beta = np.exp(hyp[0]) # noise precision
        
        # load posterior and prior covariance
        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            try:            
                self.post(hyp,X,t)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                nlZ = 1/np.finfo(float).eps
                return nlZ                
                
        m = self.m
        A = self.A
        
        # set generic variables
        N = self.N 
        Sigma = self.Sigma
        iSigma = self.iSigma
        
        nlZ = -0.5 * ( N*np.log(beta) - N*np.log(2*np.pi) - 
                       np.log(sp.linalg.det(Sigma)) - 
                       beta*(t-X.dot(m)).T.dot(t-X.dot(m) ) -
                       m.T.dot(iSigma).dot(m) - np.log(sp.linalg.det(A))
                       ) 
        
        # make sure the output is finite to stop the minimizer getting upset          
        if not np.isfinite(nlZ):
            nlZ = 1/np.finfo(float).eps
            
        if self.verbose :
            print("nlZ= ",nlZ," | hyp=",hyp)
        
        #self.hyp = hyp
        self.nlZ = nlZ         
        return nlZ
        
    # ---------------------- Compute derivatives -----------------------------
    def dloglik(self, hyp, X, t):
        
        # hyperparameters
        beta = np.exp(hyp[0])
        alpha = np.exp(hyp[1:])
           
        # load posterior and prior covariance
        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            try:            
                self.post(hyp,X,t)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ
            
        A = self.A
        m = self.m
            
        # set generic variables
        N = self.N
        D = self.D
        iSigma = self.iSigma
        
        # useful quantities        
        XX = X.T.dot(X)
        Q = sp.linalg.solve(A,X.T)
        b = (np.eye(D) - beta*Q.dot(X)).dot(Q).dot(t)
        
        # initialise derivatives
        dnlZ = np.zeros(hyp.shape)
        
        # noise precision
        dnlZ[0] = - ( N/(2*beta) -0.5*t.dot(t) + t.dot(X).dot(m) + 
                      beta * t.T.dot(X).dot(b) -
                      0.5  * m.T.dot(XX).dot(m)  - 
                      beta * b.T.dot(iSigma).dot(m) -
                      0.5  * np.trace(Q.dot(X))
                    ) * beta
                    
        # scaling parameter(s)           
        for i in range(0,len(alpha)):
            # are we using ARD?
            if len(alpha) == D:
                dSigma = np.zeros((D,D))
                dSigma[i,i] = -alpha[i] ** -2
            else:
                dSigma = -alpha[i] ** -2*np.eye(D)
            
            F = -iSigma.dot(dSigma).dot(iSigma)
            c = -beta*F.dot(X.T).dot(t)
            
            dnlZ[i+1]= -( -0.5 * np.trace(iSigma.dot(dSigma)) +
                          beta * t.T.dot(X).dot(c) -
                          beta * c.T.dot(XX).dot(m) -
                          c.T.dot(iSigma).dot(m) - 
                          0.5 * m.T.dot(F).dot(m) -
                          0.5*np.trace(sp.linalg.solve(A,F))
                        ) * alpha[i]
                        
        # make sure the gradient is finite to stop the minimizer getting upset       
        if not all(np.isfinite(dnlZ)):                
            bad = np.where(np.logical_not(np.isfinite(dnlZ)))
            for b in bad:
                dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps
        
        if self.verbose :
            print("dnlZ= ",dnlZ," | hyp=",hyp)

        self.dnlZ = dnlZ
        return dnlZ
        
    # -------------- model estimation (optimization) -------------------------
    def estimate(self, hyp0,X,t):
        from scipy.optimize import fmin_powell #,fmin_cg
        
        #out = fmin_cg(self.loglik,hyp0,self.dloglik,(X,t), 
        #              disp=True, gtol=self.tol, 
        #              maxiter=self.n_iter, full_output=1)
                   
        #out = fmin_l_bfgs_b(self.loglik,hyp0,self.dloglik,(X,t))   
         
        # Powell's method works fine if there aren't too many parameters    
        out = fmin_powell(self.loglik,hyp0,(X,t),full_output=1)
   
        hyp = out[0]
        nlZ = out[1]
        
        self.hyp = hyp
        self.nlZ = nlZ
        
        return hyp
    
    def predict(self, hyp, X, t, Xs):
        
        if (hyp != self.hyp).all() or not(hasattr(self, 'A')):
            self.post(hyp,X,t)
        A = self.A
        m = self.m
        
        # hyperparameters
        beta = np.exp(hyp[0])
        
        ys = Xs.dot(m)
        s2 = 1/beta + np.diag(Xs.dot(sp.linalg.solve(A,Xs.T)))
        
        return ys, s2
        