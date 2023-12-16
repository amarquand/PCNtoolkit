from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from pcntoolkit.model.gp import GPR, CovSum
    from pcntoolkit.normative_model.norm_base import NormBase
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from model.gp import GPR, CovSum
    from norm_base import NormBase


class NormGPR(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, **kwargs):  # X=None, y=None, theta=None,
        """
        Initialize the NormGPR object.

        This function initializes the NormGPR object with the given arguments. It requires a data matrix 'X' and optionally takes a target 'y' and parameters 'theta'.
        It also initializes the covariance function and the Gaussian Process Regression (GPR) model.

        :param kwargs: Keyword arguments which should include:
            - 'X': Data matrix. Must be specified.
            - 'y': Target values. Optional.
            - 'theta': Parameters for the model. Optional.
        """
        X = kwargs.pop('X', None)
        y = kwargs.pop('y', None)
        theta = kwargs.pop('theta', None)

        self.covfunc = CovSum(X, ('CovLin', 'CovSqExpARD'))
        self.theta0 = np.zeros(self.covfunc.get_n_params() + 1)
        self.theta = self.theta0

        if (theta is not None) and (X is not None) and (y is not None):
            self.gpr = GPR(theta, self.covfunc, X, y)
            self._n_params = self.covfunc.get_n_params() + 1
        else:
            self.gpr = GPR()

    @property
    def n_params(self):
        if not hasattr(self, '_n_params'):
            self._n_params = self.covfunc.get_n_params() + 1

        return self._n_params

    @property
    def neg_log_lik(self):
        return self.gpr.nlZ

    def estimate(self, X, y, **kwargs):
        """
        Estimate the parameters of the Gaussian Process Regression model.

        This function estimates the parameters of the Gaussian Process Regression (GPR) model given the data matrix 'X' and target 'y'. 
        If 'theta' is provided in kwargs, it is used as the initial guess for the parameters. 
        Otherwise, the initial guess is set to the current value of 'self.theta0'.

        :param X: Data matrix.
        :param y: Target values.
        :param kwargs: Keyword arguments which may include:
            - 'theta': Initial guess for the parameters. Optional.
        :return: The instance of the NormGPR object.
        """

        theta = kwargs.pop('theta', None)
        if theta is None:
            theta = self.theta0
            self.gpr = GPR(theta, self.covfunc, X, y)
        self.theta = self.gpr.estimate(theta, self.covfunc, X, y)

        return self

    def predict(self, Xs, X, y, **kwargs):
        """
        Predict the target values for the given test data.

        This function predicts the target values for the given test data 'Xs' using the estimated parameters of the Gaussian Process Regression (GPR) model. 
        If 'X' and 'y' are provided, they are used to update the model before prediction. 
        If 'theta' is provided in kwargs, it is used as the parameters for prediction. 
        Otherwise, the current value of 'self.theta' is used.

        :param Xs: Test data matrix.
        :param X: Training data matrix. Optional.
        :param y: Training target values. Optional.
        :param kwargs: Keyword arguments which may include:
            - 'theta': Parameters for prediction. Optional.
        :return: A tuple containing the predicted target values and the marginal variances for the test data.
        """
        theta = kwargs.pop('theta', None)
        if theta is None:
            theta = self.theta
        yhat, s2 = self.gpr.predict(theta, X, y, Xs)

        # only return the marginal variances
        if len(s2.shape) == 2:
            s2 = np.diag(s2)

        return yhat, s2
