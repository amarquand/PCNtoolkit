from six import with_metaclass
from abc import ABCMeta, abstractmethod

class NormBase(with_metaclass(ABCMeta)):
    """ Base class for normative model back-end.

        All normative modelling approaches must define the following methods::

            NormativeModel.estimate()
            NormativeModel.predict()
    """

    def __init__(self, x=None):
        pass

    @abstractmethod
    def estimate(self, X, y):
        """ Return the full covariance (or cross-covariance if z is given) """

    @abstractmethod
    def predict(self, X, y, Xs):
        """ Return the derivative of the covariance function with respect to
            the i-th hyperparameter """

    @property
    @abstractmethod
    def n_params(self):
        """ Report the number of parameters required by the model """
