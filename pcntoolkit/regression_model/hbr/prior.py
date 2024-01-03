from __future__ import annotations
from typing import List, Tuple
import pymc as pm
import numpy as np


class DistWrapper:
    def __init__(self, hbr, dims: Tuple[str] = ()):
        self.model = hbr.model
        self.has_covariate_dim = False if not dims else "covariates" in dims
        self.has_random_effect = False if not dims else (len(dims) - 1*self.has_covariate_dim) > 0


    def __getitem__(self, item):
        with self.model:
            if self.has_random_effect:
                if self.has_covariate_dim:
                    return self.dist[*item]
                else:
                    return self.dist[*item, None]
            else:
                return np.repeat(self.dist[None, :], item[0].shape[0], axis=0)
            
    def get_samples(self, data):
        return self[data.batch_effect_indices]

    @property
    def shape(self):
        return self.dist.shape

class Prior(DistWrapper):
    def __init__(self, hbr, name: str, dims: Tuple[str] = ()):
        super().__init__(hbr, dims)
        if dims == ():
            dims = None
            shape = (1,)
        else:
            shape = None
            if type(dims) is str:
                dims = (dims,)

        self.distmap = {'Normal': pm.Normal,
                   'HalfNormal': pm.HalfNormal,
                   'Uniform': pm.Uniform}
        dist = self.distmap[getattr(hbr.conf, f"{name}_dist")]
        params = getattr(hbr.conf, f"{name}_dist_params")
        with self.model:
            self.dist = dist(name, *params, shape=shape, dims=dims)


class DeterministicNode(DistWrapper):
    def __init__(self, hbr, name, fn, dims):
        super().__init__(hbr, dims)

        with hbr.model:
            self.dist = pm.Deterministic(name, fn(), dims=dims)


class LinearNode(DistWrapper):
    def __init__(self, hbr, slope, intercept):
        super().__init__(hbr, None)
        self.slope: DistWrapper = slope
        self.intercept: DistWrapper = intercept

    def get_samples(self, data):
        slope = self.slope.get_samples(data)
        print(f"{slope.shape.eval()=}")
        print(f"{data.pm_X.shape.eval()=}")
        product = slope * data.pm_X
        print(f"{product.shape.eval()=}")
        linear_part = pm.math.sum(product, axis=1, keepdims=True)
        intercept = self.intercept.get_samples(data)
        print(f"{intercept.shape.eval()=}")

        result = linear_part + intercept
        print(f"{result.shape.eval()=}")
        return result
    
    def __getitem__(self, item):
        raise NotImplementedError("LinearNode does not support __getitem__")
    
    @property
    def shape(self):
        raise NotImplementedError("LinearNode does not support __shape__")
