import pytest
import numpy as np

from pcntoolkit.regression_model.hbr.hbr_data import HBRData

@pytest.mark.parametrize('n_datapoints,n_covariates,n_batch_effects,n_values_per_batch_effect,has_y',[(1000,2,0,(0,),True),(1000,2,2,(2,3),True),(1000,3,1,(3,),True),(1000,4,3,(2,3,4),True),                                                                                              (1000,2,0,(0,),False),(1000,2,2,(2,3),False),(1000,3,1,(3,),False),(1000,4,3,(2,3,4),False)])
def test_create_data(n_datapoints,n_covariates,n_batch_effects,n_values_per_batch_effect, has_y):
    X = np.random.randn(n_datapoints,n_covariates)
    if has_y:
        y = np.random.randn(n_datapoints)
    else:
        y = None

    if n_batch_effects == 0:
        batch_effects = None
    else:
        batch_effects_list = []
        for i in range(n_batch_effects):
            batch_effects_list.append(np.random.choice(n_values_per_batch_effect[i],size=(n_datapoints,1)))
        batch_effects = np.concatenate(batch_effects_list,axis=1)

    data = HBRData(X,y,batch_effects)
    assert data.n_datapoints == n_datapoints
    assert data.n_covariates == n_covariates
    assert data.X.shape == (n_datapoints,n_covariates)
    assert data.y.shape == (n_datapoints,1)
    assert data.batch_effects.shape == (n_datapoints,max(1, n_batch_effects))
