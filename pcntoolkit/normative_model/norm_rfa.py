from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np

try:  # run as a package if installed
    from pcntoolkit.normative_model.norm_base import NormBase
    from pcntoolkit.model.rfa import GPRRFA
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
    del path

    from model.rfa import GPRRFA
    from norm_base import NormBase


class NormRFA(NormBase):
    """ Classical GPR-based normative modelling approach
    """

    def __init__(self, X, y=None, theta=None, n_feat=None):
        """
        Initialize the NormRFA object.

        This function initializes the NormRFA object with the given arguments. It requires a data matrix 'X' and optionally takes a target 'y', parameters 'theta', and the number of random features 'n_feat'.
        It initializes the Gaussian Process Regression with Random Feature Approximation (GPRRFA) model and sets the initial parameters.

        :param X: Data matrix. Must be specified.
        :param y: Not used.
        :param theta: Parameters for the model. Optional.
        :param n_feat: Number of random features for the GPRRFA model. Optional.
        :raises ValueError: If 'X' is not specified.
        """
        if (X is not None):
            if n_feat is None:
                print("initialising RFA")
            else:
                print("initialising RFA with", n_feat, "random features")
            self.gprrfa = GPRRFA(theta, X, n_feat=n_feat)
            self._n_params = self.gprrfa.get_n_params(X)
        else:
            raise ValueError('Covariates not specified')
            return

        if theta is None:
            self.theta0 = np.zeros(self._n_params)
        else:
            if len(theta) == self._n_params:
                self.theta0 = theta
            else:
                raise ValueError('hyperparameter vector has incorrect size')

        self.theta = self.theta0

    @property
    def n_params(self):

        return self._n_params

    @property
    def neg_log_lik(self):
        return self.gprrfa.nlZ

    def estimate(self, X, y, theta=None):
        """
        Estimate the parameters of the Random Feature Approximation model.

        This function estimates the parameters of the Random Feature Approximation (RFA) model given the data matrix 'X' and target 'y'. 
        If 'theta' is provided, it is used as the initial parameters for estimation. 
        Otherwise, the current value of 'self.theta0' is used.

        :param X: Data matrix.
        :param y: Target values.
        :param theta: Initial parameters for estimation. Optional.
        :return: The instance of the NormRFA object with updated parameters.
        """
        if theta is None:
            theta = self.theta0
        self.gprrfa = GPRRFA(theta, X, y)
        self.theta = self.gprrfa.estimate(theta, X, y)

        return self

    def predict(self, Xs, X, y, theta=None):
        """
        Predict the target values for the given test data.

        This function predicts the target values for the given test data 'Xs' using the Random Feature Approximation (RFA) model. 
        If 'X' and 'y' are provided, they are used to update the model before prediction. 
        If 'theta' is provided, it is used as the parameters for prediction. 
        Otherwise, the current value of 'self.theta' is used.

        :param Xs: Test data matrix.
        :param X: Training data matrix.
        :param y: Training target values.
        :param theta: Parameters for prediction. Optional.
        :return: A tuple containing the predicted target values and the marginal variances for the test data.
        """
        if theta is None:
            theta = self.theta
        yhat, s2 = self.gprrfa.predict(theta, X, y, Xs)

        return yhat, s2
