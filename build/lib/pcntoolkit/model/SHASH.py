import theano.tensor
from pymc3.distributions import Continuous, draw_values, generate_samples
import theano.tensor as tt
import numpy as np
from pymc3.distributions.dist_math import bound
import scipy.special as spp
from theano import as_op
from theano.gof.fg import NullType
from theano.gof.op import Op
from theano.gof.graph import Apply
from theano.gradient import grad_not_implemented

"""
@author: Stijn de Boer (AuguB)

See: Jones et al. (2009), Sinh-Arcsinh distributions.
"""


class K(Op):
    """
    Modified Bessel function of the second kind, theano implementation
    """
    def make_node(self, p, x):
        p = theano.tensor.as_tensor_variable(p, 'floatX')
        x = theano.tensor.as_tensor_variable(x, 'floatX')
        return Apply(self, [p,x], [p.type()])

    def perform(self, node, inputs, output_storage, params=None):
        # Doing this on the unique values avoids doing A LOT OF double work, apparently scipy doesn't do this by itself
        unique_inputs, inverse_indices = np.unique(inputs[0], return_inverse=True)
        unique_outputs = spp.kv(unique_inputs, inputs[1])
        outputs = unique_outputs[inverse_indices].reshape(inputs[0].shape)
        output_storage[0][0] = outputs

    def grad(self, inputs, output_grads):
        # Approximation of the derivative. This should suffice for using NUTS
        dp = 1e-10
        p = inputs[0]
        x = inputs[1]
        grad = (self(p+dp,x) - self(p, x))/dp
        return [output_grads[0]*grad, grad_not_implemented(0,1,2,3)]

class SHASH(Continuous):
    """
    SHASH described by Jones et al., based on a standard normal
    All SHASH subclasses inherit from this
    """
    def __init__(self, epsilon, delta, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = tt.as_tensor_variable(epsilon)
        self.delta = tt.as_tensor_variable(delta)
        self.K = K()


    def random(self, point=None, size=None):
        epsilon, delta = draw_values([self.epsilon, self.delta],
                                     point=point, size=size)

        def _random(epsilon, delta, size=None):
            samples_transformed = np.sinh((np.arcsinh(np.random.randn(*size)) + epsilon) / delta)
            return samples_transformed

        return generate_samples(_random, epsilon=epsilon, delta=delta, dist_shape=self.shape, size=size)

    def logp(self, value):
        epsilon = self.epsilon
        delta = self.delta + tt.np.finfo(np.float32).eps

        this_S = self.S(value)
        this_S_sqr = tt.sqr(this_S)
        this_C_sqr = 1+this_S_sqr
        frac1 = -tt.log(tt.constant(2 * tt.np.pi))/2
        frac2 = tt.log(delta) + tt.log(this_C_sqr)/2 - tt.log(1 + tt.sqr(value)) / 2
        exp = -this_S_sqr / 2

        return bound(frac1 + frac2 + exp, delta > 0)

    def S(self, x):
        """

        :param epsilon:
        :param delta:
        :param x:
        :return: The sinharcsinh transformation of x
        """
        return tt.sinh(tt.arcsinh(x) * self.delta - self.epsilon)

    def S_inv(self, x):
        return tt.sinh((tt.arcsinh(x) + self.epsilon) / self.delta)

    def C(self, x):
        """
        :param epsilon:
        :param delta:
        :param x:
        :return: the cosharcsinh transformation of x
        Be aware that this is sqrt(1+S(x)^2), so you may save some compute if you can re-use the result from S.
        """
        return tt.cosh(tt.arcsinh(x) * self.delta - self.epsilon)

    def P(self, q):
        """
        The P function as given in Jones et al.
        :param q:
        :return:
        """
        frac = np.exp(1 / 4) / np.power(8 * np.pi, 1 / 2)
        K1 = self.K((q+1)/2,1/4)
        K2 = self.K((q-1)/2,1/4)
        a = (K1 + K2) * frac
        return a

    def m(self, r):
        """
        :param epsilon:
        :param delta:
        :param r:
        :return:  The r'th uncentered moment of the SHASH distribution parameterized by epsilon and delta. Given by Jones et al.
        """
        frac1 = tt.as_tensor_variable(1 / np.power(2, r))
        acc = tt.as_tensor_variable(0)
        for i in range(r + 1):
            combs = spp.comb(r, i)
            flip = np.power(-1, i)
            ex = np.exp((r - 2 * i) * self.epsilon / self.delta)
            # This is the reason we can not sample delta/kurtosis using NUTS; the gradient of P is unknown to pymc3
            # TODO write a class that inherits theano.Op and do the gradient in there :)
            p = self.P((r - 2 * i) / self.delta)
            acc += combs * flip * ex * p
        return frac1 * acc

class SHASHo(SHASH):
    """
    This is the shash where the location and scale parameters have simply been applied as an linear transformation
    directly on the original shash.
    """

    def __init__(self, mu, sigma, epsilon, delta, **kwargs):
        super().__init__(epsilon, delta, **kwargs)
        self.mu = tt.as_tensor_variable(mu)
        self.sigma = tt.as_tensor_variable(sigma)

    def random(self, point=None, size=None):
        mu, sigma, epsilon, delta = draw_values([self.mu, self.sigma, self.epsilon, self.delta],
                                                point=point, size=size)

        def _random(mu, sigma, epsilon, delta, size=None):
            samples_transformed = np.sinh((np.arcsinh(np.random.randn(*size)) + epsilon) / delta) * sigma + mu
            return samples_transformed

        return generate_samples(_random, mu=mu, sigma=sigma, epsilon=epsilon, delta=delta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        sigma = self.sigma + tt.np.finfo(np.float32).eps
        epsilon = self.epsilon
        delta = self.delta + tt.np.finfo(np.float32).eps

        value_transformed = (value - mu) / sigma

        this_S = self.S( value_transformed)
        this_S_sqr = tt.sqr(this_S)
        this_C_sqr = 1+this_S_sqr
        frac1 = -tt.log(tt.constant(2 * tt.np.pi))/2
        frac2 = tt.log(delta) + tt.log(this_C_sqr)/2 - tt.log(
            1 + tt.sqr(value_transformed)) / 2
        exp = -this_S_sqr / 2
        change_of_variable = -tt.log(sigma)

        return bound(frac1 + frac2 + exp + change_of_variable, sigma > 0, delta > 0)


class SHASHo2(SHASH):
    """
    This is the shash where we apply the reparameterization provided in section 4.3 in Jones et al.
    """

    def __init__(self, mu, sigma, epsilon, delta, **kwargs):
        super().__init__(epsilon, delta, **kwargs)
        self.mu = tt.as_tensor_variable(mu)
        self.sigma = tt.as_tensor_variable(sigma)

    def random(self, point=None, size=None):
        mu, sigma, epsilon, delta = draw_values(
            [self.mu, self.sigma, self.epsilon, self.delta],
            point=point, size=size)
        sigma_d = sigma / delta

        def _random(mu, sigma, epsilon, delta, size=None):
            samples_transformed = np.sinh(
                (np.arcsinh(np.random.randn(*size)) + epsilon) / delta) * sigma_d + mu
            return samples_transformed

        return generate_samples(_random, mu=mu, sigma=sigma_d, epsilon=epsilon, delta=delta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        sigma = self.sigma + tt.np.finfo(np.float32).eps
        epsilon = self.epsilon
        delta = self.delta + tt.np.finfo(np.float32).eps
        sigma_d = sigma / delta


        # Here a double change of variables is applied
        value_transformed = ((value - mu) / sigma_d)

        this_S = self.S(value_transformed)
        this_S_sqr = tt.sqr(this_S)
        this_C = tt.sqrt(1+this_S_sqr)
        frac1 = -tt.log(tt.sqrt(tt.constant(2 * tt.np.pi)))
        frac2 = tt.log(delta) + tt.log(this_C) - tt.log(
            1 + tt.sqr(value_transformed)) / 2
        exp = -this_S_sqr / 2
        change_of_variable = -tt.log(sigma_d)

        # the change of variables is accounted for in the density by division and multiplication (adding and subtracting for logp)
        return bound(frac1 + frac2 + exp + change_of_variable, delta > 0, sigma > 0)

class SHASHb(SHASH):
    """
    This is the shash where the location and scale parameters been applied as an linear transformation on the shash
    distribution which was corrected for mean and variance.
    """

    def __init__(self, mu, sigma, epsilon, delta, **kwargs):
        super().__init__(epsilon, delta, **kwargs)
        self.mu = tt.as_tensor_variable(mu)
        self.sigma = tt.as_tensor_variable(sigma)

    def random(self, point=None, size=None):
        mu, sigma, epsilon, delta = draw_values(
            [self.mu, self.sigma, self.epsilon, self.delta],
            point=point, size=size)
        mean = (tt.sinh(epsilon/delta)*self.P(1/delta)).eval()
        var =  ((tt.cosh(2*epsilon/delta)*self.P(2/delta)-1)/2).eval() - mean**2

        def _random(mean, var, mu, sigma, epsilon, delta, size=None):
            samples_transformed = ((np.sinh(
                (np.arcsinh(np.random.randn(*size)) + epsilon) / delta) - mean) / np.sqrt(var)) * sigma + mu
            return samples_transformed

        return generate_samples(_random, mean=mean, var=var, mu=mu, sigma=sigma, epsilon=epsilon, delta=delta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        sigma = self.sigma + tt.np.finfo(np.float32).eps
        epsilon = self.epsilon
        delta = self.delta + tt.np.finfo(np.float32).eps
        mean = tt.sinh(epsilon/delta)*self.P(1/delta)
        var = (tt.cosh(2*epsilon/delta)*self.P(2/delta)-1)/2 - tt.sqr(mean)

        # Here a double change of variables is applied
        value_transformed = ((value - mu) / sigma) * tt.sqrt(var) + mean

        this_S = self.S(value_transformed)
        this_S_sqr = tt.sqr(this_S)
        this_C_sqr = 1+this_S_sqr
        frac1 = -tt.log(tt.constant(2 * tt.np.pi))/2
        frac2 = tt.log(delta) + tt.log(this_C_sqr)/2 - tt.log(1 + tt.sqr(value_transformed)) / 2
        exp = -this_S_sqr / 2
        change_of_variable = tt.log(var)/2 - tt.log(sigma)

        # the change of variables is accounted for in the density by division and multiplication (addition and subtraction in the log domain)
        return bound(frac1 + frac2 + exp + change_of_variable, delta > 0, sigma > 0, var > 0)
