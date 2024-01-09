import os
import numpy as np

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_conf import NormConf
from pcntoolkit.normative_model.norm_factory import create_normative_model
from pcntoolkit.regression_model.blr.blr_conf import BLRConf
from pcntoolkit.regression_model.hbr.hbr import HBR
from pcntoolkit.regression_model.hbr.hbr_conf import HBRConf
from pcntoolkit.regression_model.hbr.hbr_data import HBRData


def main():
    args = {'n_samples': 1000,'n_tune': 1000,'n_cores': 1,'likelihood': 'Normal', 'linear_mu': True, 'linear_sigma':False}
    hbr = HBR.from_args(args)

    n_datapoints = 1000
    n_covariates = 2    
    n_batch_effects = 2

    X = np.random.randn(n_datapoints, n_covariates)
    y = np.random.randn(n_datapoints)
    batch_effects = np.random.choice([0, 1], size=(n_datapoints, n_batch_effects))


    data = HBRData(X, y, batch_effects)

    # hbr.create_pymc_model(data)
    # hbr.model.to_graphviz().render('test', view=True)

    hbr.fit(data)

    # n_datapoints = 1500
    # n_covariates = 2    
    # n_batch_effects = 2

    # X = np.random.randn(n_datapoints, n_covariates)
    # y = np.random.randn(n_datapoints)
    # batch_effects = np.random.choice([0, 1], size=(n_datapoints, n_batch_effects))

    # data = HBRData(X, y, batch_effects)

    # hbr.model.set_data('X', data.X, coords={'datapoints': np.arange(n_datapoints)})
    # hbr.model.set_data('y', data.y)
    # for i in range(n_batch_effects):
    #     hbr.model.set_data(data.batch_effect_dims[i], data.batch_effects[:, i])

    # hbr.model.to_graphviz().render('test2', view=True)

    # hbr.fit(data)


if __name__ == "__main__":
    main()
