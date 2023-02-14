from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import optimize , linalg
from scipy.linalg import LinAlgError


class BLR:
    """Bayesian linear regression

    Estimation and prediction of Bayesian linear regression models

    Basic usage::

        B = BLR()
        hyp = B.estimate(hyp0, X, y)
        ys,s2 = B.predict(hyp, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters.
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns: * ys - predictive mean
              * s2 - predictive variance

    The hyperparameters are::

        hyp = ( log(beta), log(alpha) )  # hyp is a list or numpy array

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

    def __init__(self, **kwargs):
        # parse arguments
        n_iter = kwargs.get('n_iter', 100)
        tol = kwargs.get('tol', 1e-3)
        verbose = kwargs.get('verbose', False)
        var_groups = kwargs.get('var_groups', None)
        var_covariates = kwargs.get('var_covariates', None)
        warp = kwargs.get('warp', None)
        warp_reparam = kwargs.get('warp_reparam', False)
                
        if var_groups is not None and var_covariates is not None:
            raise ValueError("var_covariates and var_groups cannot both be used")
            
        # basic parameters
        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter
        self.verbose = verbose
        self.var_groups = var_groups
        if var_covariates is not None:
            self.hetero_var = True
        else:
            self.hetero_var = False
        if self.var_groups is not None:
            self.var_ids = set(self.var_groups)
            self.var_ids = sorted(list(self.var_ids))

        # set up warped likelihood
        if verbose:
            print('warp:', warp, 'warp_reparam:', warp_reparam)
        if warp is None:
            self.warp = None
            self.n_warp_param = 0
        else:
            self.warp = warp
            self.n_warp_param = warp.get_n_params()
            self.warp_reparam = warp_reparam
            
        self.gamma = None
    
    def _parse_hyps(self, hyp, X, Xv=None):

        N = X.shape[0]
        
        # noise precision
        if Xv is not None:
            if len(Xv.shape) == 1:
                Dv = 1
                Xv = Xv[:, np.newaxis]
            else:
                Dv = Xv.shape[1]           
            w_d = np.asarray(hyp[0:Dv]) 
            beta = np.exp(Xv.dot(w_d))
            n_lik_param = len(w_d)
        elif self.var_groups is not None:
            beta = np.exp(hyp[0:len(self.var_ids)])
            n_lik_param = len(beta)
        else:
            beta = np.asarray([np.exp(hyp[0])]) 
            n_lik_param = len(beta)
         
        # parameters for warping the likelhood function
        if self.warp is not None:
            gamma = hyp[n_lik_param:(n_lik_param + self.n_warp_param)]
            n_lik_param += self.n_warp_param
        else:
            gamma = None

        # precision for the coefficients
        if isinstance(beta, list) or type(beta) is np.ndarray:
            alpha = np.exp(hyp[n_lik_param:])
        else:
            alpha = np.exp(hyp[1:])

        # reparameterise the warp (WarpSinArcsinh only)
        if self.warp is not None and self.warp_reparam:
            delta = np.exp(gamma[1])
            beta = beta/(delta**2)
    
        # Create precision matrix from noise precision
        if Xv is not None:
            self.lambda_n_vec = beta
        elif self.var_groups is not None:
            beta_all = np.ones(N)
            for v in range(len(self.var_ids)):
                beta_all[self.var_groups == self.var_ids[v]] = beta[v]
            self.lambda_n_vec = beta_all
        else:     
            self.lambda_n_vec = np.ones(N)*beta
    
        return beta, alpha, gamma
        
    def post(self, hyp, X, y, Xv=None):
        """ Generic function to compute posterior distribution.

            This function will save the posterior mean and precision matrix as
            self.m and self.A and will also update internal parameters (e.g.
            N, D and the prior covariance (Sigma_a) and precision (Lambda_a).
            
            :param hyp: hyperparameter vector
            :param X: covariates
            :param y: responses
            :param Xv: covariates for heteroskedastic noise 
        """

        N = X.shape[0]
        if len(X.shape) == 1:
            D = 1
        else:
            D = X.shape[1]

        if (hyp == self.hyp).all() and hasattr(self, 'N'):
            print("hyperparameters have not changed, exiting")
            return
        
        beta, alpha, gamma = self._parse_hyps(hyp, X, Xv)

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        # prior variance
        if len(alpha) == 1 or len(alpha) == D:
            self.Sigma_a = np.diag(np.ones(D))/alpha
            self.Lambda_a = np.diag(np.ones(D))*alpha
        else:
            raise ValueError("hyperparameter vector has invalid length")

        # compute posterior precision and mean
        # this is equivalent to the following operation but makes much more 
        # efficient use of memory by avoiding the need to store Lambda_n
        #
        # self.A = X.T.dot(self.Lambda_n).dot(X) + self.Lambda_a
        # self.m = linalg.solve(self.A, X.T, 
        #                      check_finite=False).dot(self.Lambda_n).dot(y)
        
        XtLambda_n = X.T*self.lambda_n_vec
        self.A = XtLambda_n.dot(X) + self.Lambda_a
        invAXt = linalg.solve(self.A, X.T, check_finite=False)
        self.m = (invAXt*self.lambda_n_vec).dot(y)

        # save stuff
        self.N = N
        self.D = D
        self.hyp = hyp

    def loglik(self, hyp, X, y, Xv=None):
        """ Function to compute compute log (marginal) likelihood """

        # hyperparameters (alpha not needed)
        beta, alpha, gamma = self._parse_hyps(hyp, X, Xv) 

        # warp the likelihood?
        if self.warp is not None:
            if self.verbose:
                print('warping input...')
            y_unwarped = y
            y = self.warp.f(y, gamma)
            
        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
            try:
                self.post(hyp, X, y, Xv)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                nlZ = 1/np.finfo(float).eps
                return nlZ
                    
        try:
            # compute the log determinants in a numerically stable way
            logdetA = 2*sum(np.log(np.diag(np.linalg.cholesky(self.A))))
        except (ValueError, LinAlgError):
            print("Warning: Estimation of posterior distribution failed")
            nlZ = 1/np.finfo(float).eps
            return nlZ

        logdetSigma_a = sum(np.log(np.diag(self.Sigma_a))) # diagonal
        logdetSigma_n = sum(np.log(1/self.lambda_n_vec))
        
        # compute negative marginal log likelihood
        X_y_t_sLambda_n = (y-X.dot(self.m))*np.sqrt(self.lambda_n_vec)
        nlZ = -0.5 * (-self.N*np.log(2*np.pi) -
                      logdetSigma_n -
                      logdetSigma_a -
                      X_y_t_sLambda_n.T.dot(X_y_t_sLambda_n) -
                      self.m.T.dot(self.Lambda_a).dot(self.m) -
                      logdetA
                      )
        
        
        if self.warp is not None:
            # add in the Jacobian 
            nlZ = nlZ - sum(np.log(self.warp.df(y_unwarped, gamma)))

        # make sure the output is finite to stop the minimizer getting upset
        if not np.isfinite(nlZ):
            nlZ = 1/np.finfo(float).eps

        if self.verbose:
            print("nlZ= ", nlZ, " | hyp=", hyp)

        self.nlZ = nlZ
        return nlZ
    
    def penalized_loglik(self, hyp, X, y, Xv=None, l=0.1, norm='L1'):
        """ Function to compute the penalized log (marginal) likelihood 
             
            :param hyp: hyperparameter vector
            :param X: covariates
            :param y: responses
            :param Xv: covariates for heteroskedastic noise 
            :param l: regularisation penalty
            :param norm: type of regulariser (L1 or L2)
        """

        if norm.lower() == 'l1':
            L = self.loglik(hyp, X, y, Xv) + l * sum(abs(hyp))
        elif norm.lower() == 'l2':
            L = self.loglik(hyp, X, y, Xv) + l * sum(np.sqrt(hyp**2))
        else:
            print("Requested penalty not recognized, choose between 'L1' or 'L2'.")
        return L

    def dloglik(self, hyp, X, y, Xv=None):
        """ Function to compute derivatives """

        # hyperparameters
        beta, alpha, gamma = self._parse_hyps(hyp, X, Xv)
        
        if self.warp is not None:
            raise ValueError('optimization with derivatives is not yet ' + \
                             'supported for warped liklihood')

        # load posterior and prior covariance
        if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
            try:
                self.post(hyp, X, y, Xv)
            except ValueError:
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # precompute re-used quantities to maximise speed 
        # todo: revise implementation to use Cholesky throughout 
        #       that would remove the need to explicitly compute the inverse
        S = np.linalg.inv(self.A)                       # posterior covariance
        SX = S.dot(X.T)
        XLn = X.T*self.lambda_n_vec # = X.T.dot(self.Lambda_n)
        XLny = XLn.dot(y)
        SXLny = S.dot(XLny)       
        XLnXm = XLn.dot(X).dot(self.m)
      
        # initialise derivatives
        dnlZ = np.zeros(hyp.shape)
        dnl2 = np.zeros(hyp.shape)
        
        # noise precision parameter(s)
        for i in range(0, len(beta)):
            # first compute derivative of Lambda_n with respect to beta
            dL_n_vec = np.zeros(self.N)
            if self.var_groups is None:
                dL_n_vec = np.ones(self.N)
            else:    
                dL_n_vec[np.where(self.var_groups == self.var_ids[i])[0]] = 1
            dLambda_n = np.diag(dL_n_vec)
            
            # compute quantities used multiple times
            XdLnX = X.T.dot(dLambda_n).dot(X)
            dA = XdLnX
            
            # derivative of posterior parameters with respect to beta
            b = -S.dot(dA).dot(SXLny) + SX.dot(dLambda_n).dot(y)
            
            # compute np.trace(self.Sigma_n.dot(dLambda_n)) efficiently
            trSigma_ndLambda_n = sum((1/self.lambda_n_vec)*np.diag(dLambda_n))

            # compute  y.T.dot(Lambda_n) efficiently
            ytLn = (y*self.lambda_n_vec).T
            
            # compute derivatives
            dnlZ[i] = - (0.5 * trSigma_ndLambda_n - 
                         0.5 * y.dot(dLambda_n).dot(y) +
                         y.dot(dLambda_n).dot(X).dot(self.m) +
                         ytLn.dot(X).dot(b) -
                         0.5 * self.m.T.dot(XdLnX).dot(self.m) -    
                         b.T.dot(XLnXm) -
                         b.T.dot(self.Lambda_a).dot(self.m) -
                         0.5 * np.trace(S.dot(dA))
                         ) * beta[i]

        # scaling parameter(s)
        for i in range(0, len(alpha)):
            # first compute derivatives with respect to alpha
            if len(alpha) == self.D:  # are we using ARD?
                dLambda_a = np.zeros((self.D, self.D))
                dLambda_a[i, i] = 1
            else:
                dLambda_a = np.eye(self.D)

            F = dLambda_a
            c = -S.dot(F).dot(SXLny)
            
            # compute np.trace(self.Sigma_a.dot(dLambda_a)) efficiently
            trSigma_adLambda_a = sum(np.diag(self.Sigma_a)*np.diag(dLambda_a))
            
            dnlZ[i+len(beta)] = -(0.5* trSigma_adLambda_a +
                                  XLny.T.dot(c) -
                                  c.T.dot(XLnXm) -
                                  c.T.dot(self.Lambda_a).dot(self.m) -
                                  0.5 * self.m.T.dot(F).dot(self.m) -
                                  0.5*np.trace(linalg.solve(self.A, F))
                                  ) * alpha[i]

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(dnlZ)))
            for b in bad:
                dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        if self.verbose:
            print("dnlZ= ", dnlZ, " | hyp=", hyp)

        self.dnlZ = dnlZ
        return dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, X, y, **kwargs):
        """ Function to estimate the model 
                 
            :param hyp: hyperparameter vector
            :param X: covariates
            :param y: responses
            :param optimizer: optimisation algorithm ('cg','powell','nelder-mead','l0bfgs-b') 
        """
        
        optimizer = kwargs.get('optimizer','cg')
        
        # covariates for heteroskedastic noise
        Xv = kwargs.get('var_covariates', None)
        
        # options for l-bfgs-b
        l = float(kwargs.get('l', 0.1))
        epsilon = float(kwargs.get('epsilon', 0.1))
        norm = kwargs.get('norm', 'l2')

        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik, (X, y, Xv),
                                   disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)
        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (X, y, Xv),
                                       full_output=1)
        elif optimizer.lower() == 'nelder-mead':
            out = optimize.fmin(self.loglik, hyp0, (X, y, Xv),
                                       full_output=1)
        elif optimizer.lower() == 'l-bfgs-b':
            all_hyp_i = [hyp0]
            def store(X):
                hyp = X
                all_hyp_i.append(hyp)
            try:
                out = optimize.fmin_l_bfgs_b(self.penalized_loglik, x0=hyp0,
                                          args=(X, y, Xv, l, norm), approx_grad=True,
                                          epsilon=epsilon, callback=store)
            # If the matrix becomes singular restart at last found hyp
            except np.linalg.LinAlgError:
                print(f'Restarting estimation at hyp = {all_hyp_i[-1]}, due to *** numpy.linalg.LinAlgError: Matrix is singular.')
                out = optimize.fmin_l_bfgs_b(self.penalized_loglik, x0=all_hyp_i[-1],
                                          args=(X, y, Xv, l, norm), approx_grad=True,
                                          epsilon=epsilon)
        else:
            raise ValueError("unknown optimizer")

        self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs, 
                var_groups_test=None, 
                var_covariates_test=None, **kwargs):
        """ Function to make predictions from the model
             
            :param hyp: hyperparameter vector
            :param X: covariates for training data
            :param y: responses for training data
            :param Xs: covariates for test data
            :param var_covariates_test: test covariates for heteroskedastic noise 
            
            This always returns Gaussian predictions, i.e. 
            
            :returns: * ys - predictive mean
                      * s2 - predictive variance
        """
        
        Xvs = var_covariates_test
        if Xvs is not None and len(Xvs.shape) == 1:
            Xvs = Xvs[:, np.newaxis]
        
        if X is None or y is None:
            # set dummy hyperparameters
            beta, alpha, gamma = self._parse_hyps(hyp, np.zeros((self.N, self.D)), Xvs)
        else:
            
            # set hyperparameters
            beta, alpha, gamma = self._parse_hyps(hyp, X, Xvs)
            
            # do we need to re-estimate the posterior?
            if (hyp != self.hyp).any() or not(hasattr(self, 'A')):
                raise(ValueError, 'posterior not properly estimated')

        N_test = Xs.shape[0]

        ys = Xs.dot(self.m)
        
        if self.var_groups is not None:
            if len(var_groups_test) != N_test:
                raise(ValueError, 'Invalid variance groups for test')
            # separate variance groups
            s2n = np.ones(N_test)
            for v in range(len(self.var_ids)):
                s2n[var_groups_test == self.var_ids[v]] = 1/beta[v]
        else:
            s2n = 1/beta
                    
        # compute xs.dot(S).dot(xs.T) avoiding computing off-diagonal entries
        s2 = s2n + np.sum(Xs*linalg.solve(self.A, Xs.T).T, axis=1)
        
        return ys, s2       
        
    def predict_and_adjust(self, hyp, X, y, Xs=None, 
                             ys=None, 
                             var_groups_test=None,
                             var_groups_adapt=None, **kwargs):
        """ Function to transfer the model to a new site. This is done by
            first making predictions on the adaptation data given by X,
            adjusting by the residuals with respect to y.
            
            :param hyp: hyperparameter vector
            :param X: covariates for adaptation (i.e. calibration) data
            :param y: responses for adaptation data
            :param Xs: covariate data (for which predictions should be adjusted)
            :param ys: true response variables (to be adjusted)
            :param var_groups_test: variance groups (e.g. sites) for test data
            :param var_groups_adapt: variance groups for adaptation data
            
            There are two possible ways of using this function, depending on
            whether ys or Xs is specified
                
            If ys is specified, this is applied directly to the data, which is
            assumed to be in the input space (i.e. not warped). In this case 
            the adjusted true data points are returned in the same space
            
            Alternatively, Xs is specified, then the predictions are made and 
            adjusted. In this case the predictive variance are returned in the 
            warped (i.e. Gaussian) space.
            
            This function needs to know which sites are associated with which 
            data points, which provided by var_groups_xxx, which is a list or 
            array of scalar ids .
        """
        
        if ys is None:
            if Xs is None:
                raise ValueError('Either ys or Xs must be specified')
            else:
                N = Xs.shape[0]
        else:
            if len(ys.shape) < 1:
                raise ValueError('ys is specified but has insufficent length')
            N = ys.shape[0]
        
        if var_groups_test is None: 
            var_groups_test = np.ones(N)
            var_groups_adapt = np.ones(X.shape[0])
        
        ys_out = np.zeros(N)
        s2_out = np.zeros(N)
        for g in np.unique(var_groups_test):
            idx_s = var_groups_test == g
            idx_a = var_groups_adapt == g
            
            if sum(idx_a) < 2:
                raise ValueError('Insufficient adaptation data to estimate variance')
            
            # Get predictions from old model on new data X
            ys_ref, s2_ref = self.predict(hyp, None, None, X[idx_a,:])
        
            # Subtract the predictions from true data to get the residuals
            if self.warp is None:
                residuals = ys_ref-y[idx_a]
            else:
                 # Calculate the residuals in warped space
                 y_ref_ws = self.warp.f(y[idx_a], hyp[1:self.warp.get_n_params()+1])
                 residuals = ys_ref - y_ref_ws 
            
            residuals_mu = np.mean(residuals)
            residuals_sd = np.std(residuals)

            # Adjust the mean with the mean of the residuals
            if ys is None:
                # make and adjust predictions
                ys_out[idx_s], s2_out[idx_s] = self.predict(hyp, None, None, Xs[idx_s,:])
                ys_out[idx_s] = ys_out[idx_s] - residuals_mu 
                
                # Set the deviation to the devations of the residuals
                s2_out[idx_s] = np.ones(len(s2_out[idx_s]))*residuals_sd**2
            else:
                # adjust the data 
                if self.warp is not None:
                    y_ws = self.warp.f(ys[idx_s], hyp[1:self.warp.get_n_params()+1])
                    ys_out[idx_s] = y_ws + residuals_mu 
                    ys_out[idx_s] = self.warp.invf(ys_out[idx_s], hyp[1:self.warp.get_n_params()+1])
                else:
                    ys = ys - residuals_mu    
                s2_out = None

        return ys_out, s2_out
        
