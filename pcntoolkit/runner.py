import os
import warnings
from copy import deepcopy
from typing import Any, Callable, Generator

# mp.set_start_method("spawn")
# from multiprocess import Pool
from joblib import Parallel, delayed

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase


class Runner:
    def __init__(
        self,
        normative_model: NormBase,
        cross_validate: bool = False,
        cv_folds: int = 5,
        parallelize: bool = False,
        cluster_type: str = "local",
        n_jobs: int = 1,
    ):
        self.normative_model = normative_model
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self.parallelize = parallelize
        self.cluster_type = cluster_type
        self.n_jobs = n_jobs
        self.pool = None

        if self.cross_validate and self.cv_folds <= 1:
            raise ValueError(
                "If cross-validation is enabled, cv_folds must be greater than 1"
            )
        if not self.normative_model.norm_conf.savemodel:
            warnings.warn("Model saving is disabled. Results will not be saved.")

    def fit(self, data: NormData) -> None:
        fn = self.get_fit_chunk_fn()
        self.submit_jobs(fn, data)

    def get_fit_chunk_fn(self) -> Callable:
        if self.cross_validate:
            return self.kfold_fit_chunk_fn
        else:
            return self.fit_chunk_fn

    def kfold_fit_chunk_fn(self, chunk: NormData):
        conf = self.normative_model.norm_conf
        for i_fold, (train_data, _) in enumerate(chunk.kfold_split(self.cv_folds)):
            fold_norm_model: NormBase = deepcopy(self.normative_model)
            fold_norm_model.norm_conf.set_log_dir(
                os.path.join(conf.log_dir, "folds", f"fold_{i_fold}")
            )
            fold_norm_model.norm_conf.set_save_dir(
                os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
            )
            fold_norm_model.fit(train_data)

    def fit_chunk_fn(self, chunk: NormData):
        self.normative_model.fit(chunk)

    def submit_jobs(self, fn: Callable, data: NormData) -> None:
        if self.parallelize:
            chunks = data.chunk(self.n_jobs)
            Parallel(n_jobs=self.n_jobs)(delayed(fn)(chunk) for chunk in chunks)
            # pool = Pool(self.n_jobs)
            # print(fn)
            # pool.map_async(fn, chunks)
        else:
            fn(data)

    def predict(self, data: NormData) -> NormData:
        pass
        return data

    def fit_predict(self, data: NormData) -> NormData:
        pass
        return data

    def get_chunks(self, data: NormData) -> Generator[NormData, Any, Any]:
        if self.parallelize:
            yield from data.chunk(self.n_jobs)
        else:
            yield data
