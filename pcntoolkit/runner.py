import os
import warnings
from copy import deepcopy
from typing import Any, Callable, Generator

import multiprocess as mp
from multiprocess import Pool

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase

mp.set_start_method("spawn")


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
            conf = self.normative_model.norm_conf
            def fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    # Create a new normative model and update the configuration
                    fold_norm_model: NormBase = deepcopy(self.normative_model)
                    fold_norm_model.norm_conf.set_log_dir(
                        os.path.join(conf.log_dir, "folds", f"fold_{i_fold}")
                    )
                    fold_norm_model.norm_conf.set_save_dir(
                        os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
                    )
                    fold_norm_model.fit(train_data)

            return fn
        else:
            def fn(chunk: NormData):
                self.normative_model.fit(chunk)
            return fn

    def submit_jobs(self, fn: Callable, data: NormData) -> None:
        if self.parallelize:
            chunks = data.chunk(self.n_jobs)
            pool = Pool(self.n_jobs)
            pool.map_async(fn, chunks)
        else:
            fn(data)

    # def run_fit_chunk_fns(self, fit_chunk_fns: list[Callable]) -> None:
    #     logging.info(f"Starting to process {len(fit_chunk_fns)} chunks")
    #     if not self.parallelize:
    #         for fn in fit_chunk_fns:
    #             fn()
    #     else:
    #         if self.cluster_type == "local":
    #             self.run_local_jobs(fit_chunk_fns)
    #         else:
    #             raise ValueError(f"Cluster type {self.cluster_type} not supported")

    def run_local_jobs(self, fit_chunk_fns: list[Callable]) -> None:
        p = Pool(self.n_jobs)
        p.map_async(lambda fn: fn(), fit_chunk_fns)
        # result.wait()

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
