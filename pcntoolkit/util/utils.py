from __future__ import print_function

import os
import sys
import numpy as np
from scipy import stats
from subprocess import call
from scipy.stats import genextreme, norm
from six import with_metaclass
from abc import ABCMeta, abstractmethod
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import bspline
from bspline import splinelab
from sklearn.datasets import make_regression
import pymc3 as pm
from io import StringIO
import subprocess
import re

try:  # run as a package if installed
    from pcntoolkit import configs
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    rootpath = os.path.dirname(path) # parent directory 
    if rootpath not in sys.path:
        sys.path.append(rootpath)
    del path, rootpath
    import configs
    
PICKLE_PROTOCOL = configs.PICKLE_PROTOCOL

# -----------------
# Utility functions
# -----------------
def create_poly_basis(X, dimpoly):
    """ compute a polynomial basis expansion of the specified order"""
    
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D*dimpoly))
    colid = np.arange(0, D)
    for d in range(1, dimpoly+1):
        Phi[:, colid] = X ** d
        colid += D
        
    return Phi

def create_bspline_basis(xmin, xmax, p = 3, nknots = 5):
    """ compute a Bspline basis set where:
        
        :param p: order of spline (3 = cubic)
        :param nknots: number of knots (endpoints only counted once)
    """
    
    knots = np.linspace(xmin, xmax, nknots)
    k = splinelab.augknt(knots, p)       # pad the knot vector
    B = bspline.Bspline(k, p) 
    return B

def create_design_matrix(X, intercept = True, basis = 'bspline',
                         basis_column = 0, site_ids=None, all_sites=None,
                         **kwargs):
    """ Prepare a design matrix from a set of covariates sutiable for
        running Bayesian linar regression. This design matrix consists of 
        a set of user defined covariates, optoinal site intercepts 
        (fixed effects) and also optionally a nonlinear basis expansion over 
        one of the columns
        
        :param X: matrix of covariates
        :param basis: type of basis expansion to use
        :param basis_column: which colume to perform the expansion over?
        :param site_ids: list of site ids (one per data point)
        :param all_sites: list of unique site ids
        :param p: order of spline (3 = cubic)
        :param nknots: number of knots (endpoints only counted once)
        
        if site_ids is specified, this must have the same number of entries as
        there are rows in X. If all_sites is specfied, these will be used to 
        create the site identifiers in place of site_ids. This accommocdates
        the scenario where not all the sites used to create the model are 
        present in the test set (i.e. there will be some empty site columns)
    """
    
    xmin = kwargs.pop('xmin', 0)
    xmax = kwargs.pop('xmax', 100)
    
    N = X.shape[0]
    
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    
    # add intercept column 
    if intercept: 
        Phi = np.concatenate((np.ones((N, 1)), X), axis=1)
    else:
        Phi = X

    # add dummy coded site columns    
    if all_sites is None: 
        if site_ids is not None:
            all_sites = sorted(pd.unique(site_ids)) 
        
    if site_ids is None:
        if all_sites is None:
            site_cols = None
        else:
            # site ids are not specified, but all_sites are
            site_cols = np.zeros((N, len(all_sites)))
    else: 
        # site ids are defined
        # make sure the data are in pandas format
        if type(site_ids) is not pd.Series:
            site_ids = pd.Series(data=site_ids)
        #site_ids = pd.Series(data=site_ids)
        
        # make sure all_sites is defined
        if all_sites is None: 
            all_sites = sorted(pd.unique(site_ids)) 
        
        # dummy code the sites        
        site_cols = np.zeros((N, len(all_sites)))
        for i, s in enumerate(all_sites):
            site_cols[:, i] = site_ids == s
        
        if site_cols.shape[0] != N: 
            raise ValueError('site cols must have the same number of rows as X')
    
    if site_cols is not None:
        Phi = np.concatenate((Phi, site_cols), axis=1)
       
    # create Bspline basis set 
    if basis == 'bspline':
        B = create_bspline_basis(xmin, xmax, **kwargs)  
        Phi = np.concatenate((Phi, np.array([B(i) for i in X[:,basis_column]])), axis=1)
    elif basis == 'poly': 
        Phi = np.concatenate(Phi, create_poly_basis(X[:,basis_column], **kwargs))
    
    return Phi

def squared_dist(x, z=None):
    """ compute sum((x-z) ** 2) for all vectors in a 2d array"""

    # do some basic checks
    if z is None:
        z = x
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(z.shape) == 1:
        z = z[:, np.newaxis]

    nx, dx = x.shape
    nz, dz = z.shape
    if dx != dz:
        raise ValueError("""
                Cannot compute distance: vectors have different length""")

    # mean centre for numerical stability
    m = np.mean(np.vstack((np.mean(x, axis=0), np.mean(z, axis=0))), axis=0)
    x = x - m
    z = z - m

    xx = np.tile(np.sum((x*x), axis=1)[:, np.newaxis], (1, nz))
    zz = np.tile(np.sum((z*z), axis=1), (nx, 1))

    dist = (xx - 2*x.dot(z.T) + zz)

    return dist


def compute_pearsonr(A, B):
    """ Manually computes the Pearson correlation between two matrices.

        Basic usage::

            compute_pearsonr(A, B)

        :param A: an N * M data array
        :param cov: an N * M array

        :returns Rho: N dimensional vector of correlation coefficients
        :returns ys2: N dimensional vector of p-values

        Notes::

            This function is useful when M is large and only the diagonal entries
            of the resulting correlation matrix are of interest. This function
            does not compute the full correlation matrix as an intermediate step"""

    # N = A.shape[1]
    N = A.shape[0]

    # first mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del(Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)

    # Fisher r-to-z
    Zr = (np.arctanh(Rho) - np.arctanh(0)) * np.sqrt(N - 3)
    N = stats.norm()
    pRho = 2*N.cdf(-np.abs(Zr))
    # pRho = 1-N.cdf(Zr)
    
    return Rho, pRho

def explained_var(ytrue, ypred):
    """ Computes the explained variance of predicted values.

        Basic usage::

            exp_var = explained_var(ytrue, ypred)

        where

        :ytrue: n*p matrix of true values where n is the number of samples 
                and p is the number of features. 
        :ypred: n*p matrix of predicted values where n is the number of samples 
                and p is the number of features. 

        :returns exp_var: p dimentional vector of explained variances for each feature.
        
     """

    exp_var = 1 - (ytrue - ypred).var(axis = 0) / ytrue.var(axis = 0)
    
    return exp_var

def compute_MSLL(ytrue, ypred, ypred_var, train_mean = None, train_var = None): 
    """ Computes the MSLL or MLL (not standardized) if 'train_mean' and 'train_var' are None.
    
        Basic usage::
            
            MSLL = compute_MSLL(ytrue, ypred, ytrue_sig, noise_variance, train_mean, train_var)
        
        where
        
        :param ytrue     : n*p matrix of true values where n is the number of samples 
                           and p is the number of features. 
        :param ypred     : n*p matrix of predicted values where n is the number of samples 
                           and p is the number of features. 
        :param ypred_var : n*p matrix of summed noise variances and prediction variances where n is the number of samples 
                           and p is the number of features.
            
        :param train_mean: p dimensional vector of mean values of the training data for each feature.
        
        :param train_var : p dimensional vector of covariances of the training data for each feature.

        :returns loss    : p dimensional vector of MSLL or MLL for each feature.

    """
    
    if train_mean is not None and train_var is not None: 
        
        # make sure y_train_mean and y_train_sig have right dimensions (subjects x voxels):
        Y_train_mean = np.repeat(train_mean, ytrue.shape[0], axis = 0)
        Y_train_sig = np.repeat(train_var, ytrue.shape[0], axis = 0)
        
        # compute MSLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) + (ytrue - ypred)**2 / (2 * ypred_var) - 
                       0.5 * np.log(2 * np.pi * Y_train_sig) - (ytrue - Y_train_mean)**2 / (2 * Y_train_sig), axis = 0)
        
    else:   
        # compute MLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) + (ytrue - ypred)**2 / (2 * ypred_var), axis = 0)
        
    return loss

class WarpBase(with_metaclass(ABCMeta)):
    """ Base class for likelihood warping following:
        Rios and Torab (2019) Compositionally-warped Gaussian processes
        https://www.sciencedirect.com/science/article/pii/S0893608019301856

        All Warps must define the following methods::

            Warp.get_n_params() - return number of parameters
            Warp.f() - warping function (Non-Gaussian field -> Gaussian)
            Warp.invf() - inverse warp
            Warp.df() - derivatives
            Warp.warp_predictions() - compute predictive distribution
    """

    def __init__(self):
        self.n_params = np.nan

    def get_n_params(self):
        """ Report the number of parameters required """

        assert not np.isnan(self.n_params), \
            "Warp function not initialised"

        return self.n_params

    def warp_predictions(self, mu, s2, param, percentiles=[0.025, 0.975]):
        """ Compute the warped predictions from a gaussian predictive
            distribution, specifed by a mean (mu) and variance (s2)
            
            :param mu: Gassian predictive mean 
            :param s2: Predictive variance
            :param param: warping parameters
            :param percentiles: Desired percentiles of the warped likelihood

            :returns: * median - median of the predictive distribution
                      * pred_interval - predictive interval(s)
        """

        # Compute percentiles of a standard Gaussian
        N = norm
        Z = N.ppf(percentiles)
        
        # find the median (using mu = median)
        median = self.invf(mu, param)

        # compute the predictive intervals (non-stationary)
        pred_interval = np.zeros((len(mu), len(Z)))
        for i, z in enumerate(Z):
            pred_interval[:,i] = self.invf(mu + np.sqrt(s2)*z, param)

        return median, pred_interval
    
    @abstractmethod
    def f(self, x, param):
        """ Evaluate the warping function (mapping non-Gaussian respone 
            variables to Gaussian variables)"""

    @abstractmethod
    def invf(self, y, param):
        """ Evaluate the warping function (mapping Gaussian latent variables 
            to non-Gaussian response variables) """

    @abstractmethod
    def df(self, x, param):
        """ Return the derivative of the warp, dw(x)/dx """

class WarpAffine(WarpBase):
    """ Affine warp
        y = a + b*x
    """

    def __init__(self):
        self.n_params = 2
    
    def _get_params(self, param):
        if len(param) != self.n_params:
            raise(ValueError, 
                  'number of parameters must be ' + str(self.n_params))
        return param[0], param[1]

    def f(self, x, params):
        a, b = self._get_params(params)
        
        y = a + b*x 
        return y
    
    def invf(self, y, params):
        a, b = self._get_params(params)
        
        x = (y - a) / b 
       
        return x

    def df(self, x, params):
        a, b = self._get_params(params)
        
        df = np.ones(x.shape)*b
        return df

class WarpBoxCox(WarpBase):
    """ Box cox transform having a single parameter (lambda), i.e.
        
        y = (sign(x) * abs(x) ** lamda - 1) / lambda 
        
        This follows the generalization in Bicken and Doksum (1981) JASA 76
        and allows x to assume negative values. 
    """

    def __init__(self):
        self.n_params = 1
    
    def _get_params(self, param):
        
        return np.exp(param)

    def f(self, x, params):
        lam = self._get_params(params)
        
        if lam == 0:
            y = np.log(x)
        else:
            y = (np.sign(x) * np.abs(x) ** lam - 1) / lam 
        return y
    
    def invf(self, y, params):
        lam = self._get_params(params)
     
        if lam == 0:
            x = np.exp(y)
        else:
            x = np.sign(lam * y + 1) * np.abs(lam * y + 1) ** (1 / lam)

        return x

    def df(self, x, params):
        lam = self._get_params(params)
        
        dx = np.abs(x) ** (lam - 1)
        
        return dx

class WarpSinArcsinh(WarpBase):
    """ Sin-hyperbolic arcsin warp having two parameters (a, b) and defined by 
    
        y = sinh(b *  arcsinh(x) - a)
        
        Using the parametrisation of Rios et al, Neural Networks 118 (2017)
        where a controls skew and b controls kurtosis, such that:
        
        * a = 0 : symmetric
        * a > 0 : positive skew
        * a < 0 : negative skew
        * b = 1 : mesokurtic
        * b > 1 : leptokurtic
        * b < 1 : platykurtic
        
        where b > 0. However, it is more convenentent to use an alternative 
        parameterisation, where

        y = sinh(b * arcsinh(x) + epsilon * b)
        
        and a = -epsilon*b
    
        see Jones and Pewsey A (2009) Biometrika, 96 (4) (2009)
    """

    def __init__(self):
        self.n_params = 2
    
    def _get_params(self, param):
        if len(param) != self.n_params:
            raise(ValueError, 
                  'number of parameters must be ' + str(self.n_params))

        epsilon = param[0]
        b = np.exp(param[1])
        a = -epsilon*b
        
        return a, b

    def f(self, x, params):
        a, b = self._get_params(params)
        
        y = np.sinh(b * np.arcsinh(x) - a)
        return y
    
    def invf(self, y, params):
        a, b = self._get_params(params)
     
        x = np.sinh((np.arcsinh(y)+a)/b)
        
        return x

    def df(self, x, params):
        a, b = self._get_params(params)
        
        dx = (b *np.cosh(b * np.arcsinh(x) - a))/np.sqrt(1 + x ** 2)
        
        return dx
    
class WarpCompose(WarpBase):
    """ Composition of warps. These are passed in as an array and
        intialised automatically. For example::

            W = WarpCompose(('WarpBoxCox', 'WarpAffine'))

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, warpnames=None):

        if warpnames is None:
            raise ValueError("A list of warp functions is required")
        self.warps = []
        self.n_params = 0
        for wname in warpnames:
            warp = eval(wname + '()')
            self.n_params += warp.get_n_params()
            self.warps.append(warp)

    def f(self, x, theta):
        theta_offset = 0

        for ci, warp in enumerate(self.warps):
            n_params_c = warp.get_n_params()
            theta_c = [theta[c] for c in
                          range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c                

            if ci == 0:
                fw = warp.f(x, theta_c)
            else:
                fw = warp.f(fw, theta_c)
        return fw

    def invf(self, x, theta):
        theta_offset = 0
        for ci, warp in enumerate(self.warps):
            n_params_c = warp.get_n_params()
            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c
            
            if ci == 0:
                finvw = warp.invf(x, theta_c)
            else:
                finvw = warp.invf(finvw, theta_c)
            
        return finvw
    
    def df(self, x, theta):
        theta_offset = 0
        for ci, warp in enumerate(self.warps):
            n_params_c = warp.get_n_params()

            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c
            
            if ci == 0:
                dfw = warp.df(x, theta_c)
            else:
                dfw = warp.df(dfw, theta_c)
            
        return dfw

# -----------------------
# Functions for inference
# -----------------------

class CustomCV:
    """ Custom cross-validation approach. This function does not do much, it
        merely provides a wrapper designed to be compatible with
        scikit-learn (e.g. sklearn.model_selection...)

        :param train: a list of indices of training splits (each itself a list)
        :param test: a list of indices of test splits (each itself a list)

        :returns tr: Indices for training set
        :returns te: Indices for test set """

    def __init__(self, train, test, X=None, y=None):
        self.train = train
        self.test = test
        self.n_splits = len(train)
        if X is not None:
            self.N = X.shape[0]
        else:
            self.N = None

    def split(self, X, y=None):
        if self.N is None:
            self.N = X.shape[0]

        for i in range(0, self.n_splits):
            tr = self.train[i]
            te = self.test[i]
            yield tr, te

# -----------------------
# Functions for inference
# -----------------------

def bashwrap(processing_dir, python_path, script_command, job_name,
             bash_environment=None):

    """ This function wraps normative modelling into a bash script to run it
    on a torque cluster system.

    :param processing_dir: Full path to the processing dir
    :param python_path: Full path to the python distribution
    :param script_command: python command to execute
    :param job_name: Name for the bash script output by this function
    :param covfile_path: Full path to covariates
    :param respfile_path: Full path to response variables
    :param cv_folds: Number of cross validations
    :param testcovfile_path: Full path to test covariates
    :param testrespfile_path: Full path to tes responses
    :param bash_environment: A file containing enviornment specific commands
                                
    :returns: A .sh file containing the commands for normative modelling

    witten by Thomas Wolfers
    """

    # change to processing dir
    os.chdir(processing_dir)
    output_changedir = ['cd ' + processing_dir + '\n']

    # sets bash environment if necessary
    if bash_environment is not None:
        bash_environment = [bash_environment]
        print("""Your own environment requires in any case:
              #!/bin/bash\n export and optionally OMP_NUM_THREADS=1\n""")
    else:
        bash_lines = '#!/bin/bash\n\n'
        bash_cores = 'export OMP_NUM_THREADS=1\n'
        bash_environment = [bash_lines + bash_cores]

    command = [python_path + ' ' + script_command + '\n']
    
    # writes bash file into processing dir
    bash_file_name = os.path.join(processing_dir, job_name + '.sh')
    with open(bash_file_name, 'w') as bash_file:
        bash_file.writelines(bash_environment + output_changedir + command)

    # changes permissoins for bash.sh file
    os.chmod(bash_file_name, 0o700)
    
    return bash_file_name

def qsub(job_path, memory, duration, logdir=None):
    """
    This function submits a job.sh scipt to the torque custer using the qsub
    command.

    ** Input:
        * job_path      -> Full path to the job.sh file
        * memory        -> Memory requirements written as string for example
                           4gb or 500mb
        * duration       -> The approximate duration of the job, a string with
                           HH:MM:SS for example 01:01:01

    ** Output:
        * Submission of the job to the (torque) cluster

    witten by Thomas Wolfers
    """
    if logdir is None:
        logdir = os.path.expanduser('~')

    # created qsub command
    qsub_call = ['echo ' + job_path + ' | qsub -N ' + job_path + ' -l ' +
                 'mem=' + memory + ',walltime=' + duration + 
                 ' -e ' + logdir + ' -o ' + logdir]

    # submits job to cluster
    call(qsub_call, shell=True)
    
def extreme_value_prob_fit(NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        temp = temp[0:int(np.floor(0.90*temp.shape[0]))]
        m[i] = np.mean(temp)
    params = genextreme.fit(m)
    return params
    
def extreme_value_prob(params, NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        temp = temp[0:int(np.floor(0.90*temp.shape[0]))]
        m[i] = np.mean(temp)
        probs = genextreme.cdf(m,*params)
    return probs

def ravel_2D(a):
    s = a.shape
    return np.reshape(a,[s[0], np.prod(s[1:])]) 

def unravel_2D(a, s):
    return np.reshape(a,s)

def threshold_NPM(NPMs, fdr_thr=0.05, npm_thr=0.1):
    """ Compute voxels with significant NPMs. """
    p_values = stats.norm.cdf(-np.abs(NPMs))
    results = np.zeros(NPMs.shape) 
    masks = np.full(NPMs.shape, False, dtype=bool)
    for i in range(p_values.shape[0]): 
        masks[i,:] = FDR(p_values[i,:], fdr_thr)
        results[i,] = NPMs[i,:] * masks[i,:].astype(np.int)
    m = np.sum(masks,axis=0)/masks.shape[0] > npm_thr
    #m = np.any(masks,axis=0)
    return results, masks, m
    
def FDR(p_values, alpha):
    """ Compute the false discovery rate in all voxels for a subject. """
    dim = np.shape(p_values)
    p_values = np.reshape(p_values,[np.prod(dim),])
    sorted_p_values = np.sort(p_values)
    sorted_p_values_idx = np.argsort(p_values);  
    testNum = len(p_values)
    thresh = ((np.array(range(testNum)) + 1)/np.float(testNum))  * alpha
    h = sorted_p_values <= thresh
    unsort = np.argsort(sorted_p_values_idx)
    h = h[unsort]
    h = np.reshape(h, dim)
    return h


def calibration_error(Y,m,s,cal_levels):
    ce = 0
    for cl in cal_levels:
        z = np.abs(norm.ppf((1-cl)/2))
        ub = m + z * s
        lb = m - z * s
        ce = ce + np.abs(cl - np.sum(np.logical_and(Y>=lb,Y<=ub))/Y.shape[0])
    return ce


def simulate_data(method='linear', n_samples=100, n_features=1, n_grps=1, 
                  working_dir=None, plot=False, random_state=None, noise=None):
    """
    This function simulates linear synthetic data for testing pcntoolkit methods.
    
    :param method: simulate 'linear' or 'non-linear' function.
    :param n_samples: number of samples in each group of the training and test sets. 
        If it is an int then the same sample number will be used for all groups. 
        It can be also a list of size of n_grps that decides the number of samples 
        in each group (default=100).
    :param n_features: A positive integer that decides the number of features 
        (default=1).
    :param n_grps: A positive integer that decides the number of groups in data
        (default=1).
    :param working_dir: Directory to save data (default=None). 
    :param plot: Boolean to plot the simulated training data (default=False).
    :param random_state: random state for generating random numbers (Default=None).
    :param noise: Type of added noise to the data. The options are 'gaussian', 
        'exponential', and 'hetero_gaussian' (The defauls is None.). 
    
    :returns:
         X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test, coef
    
    """
    
    if isinstance(n_samples, int):
        n_samples = [n_samples for i in range(n_grps)]
        
    X_train, Y_train, X_test, Y_test = [], [], [], []
    grp_id_train, grp_id_test = [], []
    coef = []
    for i in range(n_grps):
        bias = np.random.randint(-10, high=10)
        
        if method == 'linear':
            X_temp, Y_temp, coef_temp = make_regression(n_samples=n_samples[i]*2, 
                                    n_features=n_features, n_targets=1, 
                                    noise=10 * np.random.rand(), bias=bias, 
                                    n_informative=1, coef=True, 
                                    random_state=random_state)
        elif method == 'non-linear':
            X_temp = np.random.randint(-2,6,[2*n_samples[i], n_features]) \
                    + np.random.randn(2*n_samples[i], n_features)
            Y_temp = X_temp[:,0] * 20 * np.random.rand() + np.random.randint(10,100) \
                        * np.sin(2 * np.random.rand() + 2 * np.pi /5 * X_temp[:,0]) 
            coef_temp = 0
        elif method == 'combined':
            X_temp = np.random.randint(-2,6,[2*n_samples[i], n_features]) \
                    + np.random.randn(2*n_samples[i], n_features)
            Y_temp = (X_temp[:,0]**3) * np.random.uniform(0, 0.5) \
                    + X_temp[:,0] * 20 * np.random.rand() \
                    + np.random.randint(10, 100)
            coef_temp = 0
        else:
            raise ValueError("Unknow method. Please specify valid method among \
                             'linear' or  'non-linear'.")
        coef.append(coef_temp/100)
        X_train.append(X_temp[:X_temp.shape[0]//2])
        Y_train.append(Y_temp[:X_temp.shape[0]//2]/100)
        X_test.append(X_temp[X_temp.shape[0]//2:])
        Y_test.append(Y_temp[X_temp.shape[0]//2:]/100)
        grp_id = np.repeat(i, X_temp.shape[0])
        grp_id_train.append(grp_id[:X_temp.shape[0]//2])
        grp_id_test.append(grp_id[X_temp.shape[0]//2:])
        
        if noise == 'hetero_gaussian':
            t = np.random.randint(5,10)
            Y_train[i] = Y_train[i] + np.random.randn(Y_train[i].shape[0]) / t \
                        * np.log(1 + np.exp(X_train[i][:,0]))
            Y_test[i] = Y_test[i] + np.random.randn(Y_test[i].shape[0]) / t \
                        * np.log(1 + np.exp(X_test[i][:,0]))
        elif noise == 'gaussian':
            t = np.random.randint(3,10)
            Y_train[i] = Y_train[i] + np.random.randn(Y_train[i].shape[0])/t
            Y_test[i] = Y_test[i] + np.random.randn(Y_test[i].shape[0])/t
        elif noise == 'exponential':
            t = np.random.randint(1,3)
            Y_train[i] = Y_train[i] + np.random.exponential(1, Y_train[i].shape[0]) / t
            Y_test[i] = Y_test[i] + np.random.exponential(1, Y_test[i].shape[0]) / t
        elif noise == 'hetero_gaussian_smaller':
            t = np.random.randint(5,10)
            Y_train[i] = Y_train[i] + np.random.randn(Y_train[i].shape[0]) / t \
                        * np.log(1 + np.exp(0.3 * X_train[i][:,0]))
            Y_test[i] = Y_test[i] + np.random.randn(Y_test[i].shape[0]) / t \
                        * np.log(1 + np.exp(0.3 * X_test[i][:,0]))
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)
    grp_id_train = np.expand_dims(np.concatenate(grp_id_train), axis=1)
    grp_id_test = np.expand_dims(np.concatenate(grp_id_test), axis=1)
    
    for i in range(n_features):
        plt.figure()
        for j in range(n_grps):
            plt.scatter(X_train[grp_id_train[:,0]==j,i],
                Y_train[grp_id_train[:,0]==j,], label='Group ' + str(j))
        plt.xlabel('X' + str(i))
        plt.ylabel('Y')
        plt.legend()
        
    if working_dir is not None:
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        with open(os.path.join(working_dir ,'trbefile.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(grp_id_train),file, protocol=PICKLE_PROTOCOL)
        with open(os.path.join(working_dir ,'tsbefile.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(grp_id_test),file, protocol=PICKLE_PROTOCOL)
        with open(os.path.join(working_dir ,'X_train.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(X_train),file, protocol=PICKLE_PROTOCOL)
        with open(os.path.join(working_dir ,'X_test.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(X_test),file, protocol=PICKLE_PROTOCOL)
        with open(os.path.join(working_dir ,'Y_train.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(Y_train),file, protocol=PICKLE_PROTOCOL)
        with open(os.path.join(working_dir ,'Y_test.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(Y_test),file, protocol=PICKLE_PROTOCOL)
        
    return X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test, coef


def divergence_plot(nm, ylim=None):
    
    if nm.hbr.configs['n_chains'] > 1 and nm.hbr.model_type != 'nn':
        a = pm.summary(nm.hbr.trace).round(2)
        plt.figure()
        plt.hist(a['r_hat'],10)
        plt.title('Gelman-Rubin diagnostic for divergence')

    divergent = nm.hbr.trace['diverging']
        
    tracedf = pm.trace_to_dataframe(nm.hbr.trace)
    
    _, ax = plt.subplots(2, 1, figsize=(15, 4), sharex=True, sharey=True)
    ax[0].plot(tracedf.values[divergent == 0].T, color='k', alpha=.05)
    ax[0].set_title('No Divergences', fontsize=10)
    ax[1].plot(tracedf.values[divergent == 1].T, color='C2', lw=.5, alpha=.5)
    ax[1].set_title('Divergences', fontsize=10)
    plt.ylim(ylim)
    plt.xticks(range(tracedf.shape[1]), list(tracedf.columns))
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.show()
    
    
def load_freesurfer_measure(measure, data_path, subjects_list):
    
    """
    This is a utility function to load different Freesurfer measures in a pandas
    Dataframe.
    
    Inputs

        :param measure: a string that defines the type of Freesurfer measure we want to load. \
            The options include:

        * 'NumVert': Number of Vertices in each cortical area based on Destrieux atlas.
        * 'SurfArea: Surface area for each cortical area based on Destrieux atlas.
        * 'GrayVol': Gary matter volume in each cortical area based on Destrieux atlas.
        * 'ThickAvg': Average Cortical thinckness in each cortical area based on Destrieux atlas.
        * 'ThickStd': STD of Cortical thinckness in each cortical area based on Destrieux atlas.
        * 'MeanCurv': Integrated Rectified Mean Curvature in each cortical area based on Destrieux atlas.
        * 'GausCurv': Integrated Rectified Gaussian Curvature in each cortical area based on Destrieux atlas.
        * 'FoldInd': Folding Index in each cortical area based on Destrieux atlas.
        * 'CurvInd': Intrinsic Curvature Index in each cortical area based on Destrieux atlas.
        * 'brain': Brain Segmentation Statistics from aseg.stats file.
        * 'subcortical_volumes': Subcortical areas volume.
        
        :param data_path: a string that specifies the path to the main Freesurfer folder.
        :param subjects_list: A Pythin list containing the list of subject names to load the data for. \
            The subject names should match the folder name for each subject's Freesurfer data folder.
        
    Outputs:
        - df: A pandas datafrmae containing the subject names as Index and target Freesurfer measures.
        - missing_subs: A Python list of subject names that miss the target Freesurefr measures.
            
    """
    
    df = pd.DataFrame()
    missing_subs = []
    
    if measure in ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                   'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']:
        l = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                   'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
        col = l.index(measure) + 1
        for i, sub in enumerate(subjects_list):
            try:
                data = dict()
           
                a = pd.read_csv(data_path + sub + '/stats/lh.aparc.a2009s.stats', 
                                delimiter='\s+', comment='#', header=None)
                temp = dict(zip(a[0], a[col]))
                for key in list(temp.keys()):
                    temp['L_'+key] = temp.pop(key)
                data.update(temp)
               
                a = pd.read_csv(data_path + sub + '/stats/rh.aparc.a2009s.stats', 
                                delimiter='\s+', comment='#', header=None)
                temp = dict(zip(a[0], a[col]))
                for key in list(temp.keys()):
                    temp['R_'+key] = temp.pop(key)
                data.update(temp)
                
                df_temp = pd.DataFrame(data,index=[sub])         
                df = pd.concat([df, df_temp])
                print('%d / %d: %s is done!' %(i, len(subjects_list), sub))
            except:
                missing_subs.append(sub)
                print('%d / %d: %s is missing!' %(i, len(subjects_list), sub))
                continue
    
    elif measure == 'brain':
        for i, sub in enumerate(subjects_list):
            try:
                data = dict()
                s = StringIO()
                with open(data_path + sub + '/stats/aseg.stats') as f:
                    for line in f:
                        if line.startswith('# Measure'):
                            s.write(line)
                s.seek(0) # "rewind" to the beginning of the StringIO object
                a = pd.read_csv(s, header=None) # with further parameters?
                data_brain = dict(zip(a[1], a[3]))
                data.update(data_brain)
                df_temp = pd.DataFrame(data,index=[sub])         
                df = pd.concat([df, df_temp])
                print('%d / %d: %s is done!' %(i, len(subjects_list), sub))
            except:
                missing_subs.append(sub)
                print('%d / %d: %s is missing!' %(i, len(subjects_list), sub))
                continue
    
    elif measure == 'subcortical_volumes':
        for i, sub in enumerate(subjects_list):
            try:
                data = dict()
                s = StringIO()
                with open(data_path + sub + '/stats/aseg.stats') as f:
                    for line in f:
                        if line.startswith('# Measure'):
                            s.write(line)
                s.seek(0) # "rewind" to the beginning of the StringIO object
                a = pd.read_csv(s, header=None) # with further parameters?
                a = dict(zip(a[1], a[3]))
                if ' eTIV' in a.keys():
                    tiv = a[' eTIV']
                else:
                    tiv = a[' ICV']
                a = pd.read_csv(data_path + sub + '/stats/aseg.stats', delimiter='\s+', comment='#', header=None)
                data_vol = dict(zip(a[4]+'_mm3', a[3]))
                for key in data_vol.keys():
                    data_vol[key] = data_vol[key]/tiv
                data.update(data_vol)
                data = pd.DataFrame(data,index=[sub])         
                df = pd.concat([df, data])
                print('%d / %d: %s is done!' %(i, len(subjects_list), sub))
            except:
                missing_subs.append(sub)
                print('%d / %d: %s is missing!' %(i, len(subjects_list), sub))
                continue
    
    return df, missing_subs


class scaler:
    
    def __init__(self, scaler_type='standardize', tail=0.01):
        
        self.scaler_type = scaler_type
        self.tail = tail
        
        if self.scaler_type not in ['standardize', 'minmax', 'robminmax']:
             raise ValueError("Undifined scaler type!")  
        
        
    def fit(self, X):
        
        if self.scaler_type == 'standardize':
            
            self.m = np.mean(X, axis=0)
            self.s = np.std(X, axis=0)
            
        elif self.scaler_type == 'minmax':
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        
        elif self.scaler_type == 'robminmax':
            self.min = np.zeros([X.shape[1],])
            self.max = np.zeros([X.shape[1],])
            for i in range(X.shape[1]):
                self.min[i] = np.median(np.sort(X[:,i])[0:int(np.round(X.shape[0] * self.tail))])
                self.max[i] = np.median(np.sort(X[:,i])[-int(np.round(X.shape[0] * self.tail)):])   
                
                
    def transform(self, X, adjust_outliers=False):
        
        if self.scaler_type == 'standardize':
            
            X = (X - self.m) / self.s 
        
        elif self.scaler_type in ['minmax', 'robminmax']:
            
            X = (X - self.min) / (self.max - self.min)
            
            if adjust_outliers:
                
                X[X < 0] = 0
                X[X > 1] = 1
            
        return X
    
    def inverse_transform(self, X, index=None):
        
        if self.scaler_type == 'standardize':
            if index is None:
                X = X * self.s + self.m
            else:
                X = X * self.s[index] + self.m[index]
        
        elif self.scaler_type in ['minmax', 'robminmax']:
            if index is None:
                X = X * (self.max - self.min) + self.min 
            else:
                X = X * (self.max[index] - self.min[index]) + self.min[index]
        return X
    
    def fit_transform(self, X, adjust_outliers=False):
        
        if self.scaler_type == 'standardize':
            
            self.m = np.mean(X, axis=0)
            self.s = np.std(X, axis=0)
            X = (X - self.m) / self.s 
            
        elif self.scaler_type == 'minmax':
            
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            X = (X - self.min) / (self.max - self.min)
        
        elif self.scaler_type == 'robminmax':
            
            self.min = np.zeros([X.shape[1],])
            self.max = np.zeros([X.shape[1],])
            
            for i in range(X.shape[1]):
                self.min[i] = np.median(np.sort(X[:,i])[0:int(np.round(X.shape[0] * self.tail))])
                self.max[i] = np.median(np.sort(X[:,i])[-int(np.round(X.shape[0] * self.tail)):])   
            
            X = (X - self.min) / (self.max - self.min)
            
            if adjust_outliers:             
                X[X < 0] = 0
                X[X > 1] = 1
        
        return X
    
    
    
def retrieve_freesurfer_eulernum(freesurfer_dir, subjects=None, save_path=None):
    
    '''
    This function receives the freesurfer directory (including processed data 
    for several subjects) and retrieves the Euler number from the log files. If
    the log file does not exist, this function uses 'mris_euler_number' to recompute
    the Euler numbers (ENs). The function returns the ENs in a dataframe and 
    the list of missing subjects (that for which computing EN is failed). If 
    'save_path' is specified then the results will be saved in a pickle file.

    Basic usage::

        ENs, missing_subjects = retrieve_freesurfer_eulernum(freesurfer_dir)

    where the arguments are defined below.

    :param freesurfer_dir: absolute path to the Freesurfer directory.
    :param subjects: List of subject that we want to retrieve the ENs for. 
     If it is 'None' (the default), the list of the subjects will be automatically
     retreived from existing directories in the 'freesurfer_dir' (i.e. the ENs
     for all subjects will be retrieved).
    :param save_path: The path to save the results. If 'None' (default) the 
     results are not saves on the disk.


    :outputs: * ENs - A dataframe of retrieved ENs.
              * missing_subjects - The list of missing subjects.
              
    Developed by S.M. Kia
    
    '''
    
    if subjects is None:
        subjects = [temp for temp in os.listdir(freesurfer_dir) 
                    if os.path.isdir(os.path.join(freesurfer_dir ,temp))]
        
    df = pd.DataFrame(index=subjects, columns=['lh_en','rh_en','avg_en'])
    missing_subjects = []
    
    for s, sub in enumerate(subjects):
        sub_dir = os.path.join(freesurfer_dir, sub)
        log_file = os.path.join(sub_dir, 'scripts', 'recon-all.log')
        
        if os.path.exists(sub_dir):
            if os.path.exists(log_file):    
                with open(log_file) as f:
                    for line in f:
                        # find the part that refers to the EC
                        if re.search('orig.nofix lheno', line):
                            eno_line = line
                f.close()
                eno_l = eno_line.split()[3][0:-1] # remove the trailing comma
                eno_r = eno_line.split()[6]
                euler = (float(eno_l) + float(eno_r)) / 2
                
                df.at[sub, 'lh_en'] = eno_l
                df.at[sub, 'rh_en'] = eno_r
                df.at[sub, 'avg_en'] = euler
                
                print('%d: Subject %s is successfully processed. EN = %f' 
                      %(s, sub, df.at[sub, 'avg_en']))
            else:
                print('%d: Subject %s is missing log file, running QC ...' %(s, sub))
                try:
                    bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/lh.orig.nofix>' + 'temp_l.txt 2>&1'
                    res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_l.txt', mode = 'r', encoding = 'utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip() for item in line.split(' ')])
                    eno_l = np.float32(words[0][12])
                    
                    bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/rh.orig.nofix>' + 'temp_r.txt 2>&1'
                    res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_r.txt', mode = 'r', encoding = 'utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip() for item in line.split(' ')])
                    eno_r = np.float32(words[0][12])
                    
                    df.at[sub, 'lh_en'] = eno_l
                    df.at[sub, 'rh_en'] = eno_r
                    df.at[sub, 'avg_en'] = (eno_r + eno_l) / 2
                
                    print('%d: Subject %s is successfully processed. EN = %f' 
                          %(s, sub, df.at[sub, 'avg_en']))
                    
                except:
                    e = sys.exc_info()[0]
                    missing_subjects.append(sub)
                    print('%d: QC is failed for subject %s: %s.' %(s, sub, e))
                
        else:
            missing_subjects.append(sub)
            print('%d: Subject %s is missing.' %(s, sub))
        df = df.dropna()
        
        if save_path is not None:
            with open(save_path, 'wb') as file:
                pickle.dump({'ENs':df}, file)
             
    return df, missing_subjects
