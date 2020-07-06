from __future__ import print_function

import os
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

        where

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
        
        :ytrue          : n*p matrix of true values where n is the number of samples 
                          and p is the number of features. 
        :ypred          : n*p matrix of predicted values where n is the number of samples 
                          and p is the number of features. 
        :ypred_var      : n*p matrix of summed noise variances and prediction variances where n is the number of samples 
                          and p is the number of features.
        :train_mean     : p dimensional vector of mean values of the training data for each feature.
        :train_var      : p dimensional vector of covariances of the training data for each feature.
        
        :returns loss   : p dimensional vector of MSLL or MLL for each feature.
    
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
    
        see Jones and Pewsey A (2009) Biometrika, 96 (4) (2009)
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

    ** Input:
        * processing_dir     -> Full path to the processing dir
        * python_path        -> Full path to the python distribution
        * command to execute -> python command to execute
        * job_name           -> Name for the bash script that is the output of
                                this function
        * covfile_path       -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                responsefile
        * respfile_path      -> Full path to a .txt that contains all features
                                (subjects x features)
        * cv_folds           -> Number of cross validations
        * testcovfile_path   -> Full path to a .txt file that contains all
                                covariats (subjects x covariates) for the
                                testresponse file
        * testrespfile_path  -> Full path to a .txt file that contains all
                                test features
        * bash_environment   -> A file containing the necessary commands
                                for your bash environment to work

    ** Output:
        * A bash.sh file containing the commands for normative modelling saved
        to the processing directory

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
    
    - Inputs:
        
        - method: simulate 'linear' or 'non-linear' function.
        
        - n_samples: number of samples in each group of the training and test sets. 
        If it is an int then the same sample number will be used for all groups. 
        It can be also a list of size of n_grps that decides the number of samples 
        in each group (default=100).
        
        - n_features: A positive integer that decides the number of features 
        (default=1).
        
        - n_grps: A positive integer that decides the number of groups in data
        (default=1).
        
        - working_dir: Directory to save data (default=None). 
        
        - plot: Boolean to plot the simulated training data (default=False).
        
        - random_state: random state for generating random numbers (Default=None).
        
        - noise: Type of added noise to the data. The options are 'gaussian', 
        'exponential', and 'hetero_gaussian' (The defauls is None.). 
    
    - Outputs:
        
        - X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test, coef
    
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
            pickle.dump(pd.DataFrame(grp_id_train),file)
        with open(os.path.join(working_dir ,'tsbefile.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(grp_id_test),file)
        with open(os.path.join(working_dir ,'X_train.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(X_train),file)
        with open(os.path.join(working_dir ,'X_test.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(X_test),file)
        with open(os.path.join(working_dir ,'Y_train.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(Y_train),file)
        with open(os.path.join(working_dir ,'Y_test.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(Y_test),file)
        
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