from __future__ import print_function

import os
import numpy as np
from scipy import stats
from subprocess import call

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
    N = len(A[:, 0])

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