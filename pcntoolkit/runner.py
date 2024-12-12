import os
import subprocess
import sys
import warnings
from copy import deepcopy
from typing import Any, Callable, Generator, Optional

import xarray as xr

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
        job_type: str = "local",
        n_jobs: int = 1,
        python_path: Optional[str] = None,
    ):
        self.normative_model = normative_model
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self.parallelize = parallelize
        self.job_type = job_type
        self.n_jobs = n_jobs
        self.pool = None
        
        # Get Python path if not provided
        if not python_path:
            # Option 1: Get the interpreter path
            self.python_path = sys.executable
            print("No python path specified. Using interpreter path of current process:", self.python_path)
        else:
            self.python_path = python_path

        if self.cross_validate and self.cv_folds <= 1:
            raise ValueError(
                "If cross-validation is enabled, cv_folds must be greater than 1"
            )
        if not self.normative_model.norm_conf.savemodel:
            warnings.warn("Model saving is disabled. Results will not be saved.")

    def fit(self, data: NormData) -> None:
        fn = self.get_fit_chunk_fn()
        self.submit_fit_jobs(fn, data)

    def fit_predict(self, fit_data: NormData, predict_data: Optional[NormData] = None) -> None:
        fn = self.get_fit_predict_chunk_fn()
        self.submit_fit_predict_jobs(fn, fit_data, predict_data)

    def get_fit_chunk_fn(self) -> Callable:
        if self.cross_validate:
            def kfold_fit_chunk_fn(chunk: NormData):
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
            return kfold_fit_chunk_fn
        else:
            def fit_chunk_fn(chunk: NormData):
                self.normative_model.fit(chunk)
            return fit_chunk_fn
        
    def get_fit_predict_chunk_fn(self) -> Callable:
        if self.cross_validate:
            def kfold_fit_predict_chunk_fn(all_data: NormData, unused_predict_data: Optional[NormData] = None):
                if unused_predict_data is not None:
                    warnings.warn("predict_data is not used in kfold cross-validation")
                for i_fold, (fit_data, predict_data) in enumerate(all_data.kfold_split(self.cv_folds)):
                    conf = self.normative_model.norm_conf
                    pid = os.getpid()
                    y:xr.DataArray = fit_data.y
                    print(f"I am {pid} and I am fitting fold {i_fold}, on train data with mean: {y.to_numpy().mean(axis=0)}")
                    # fold_norm_model: NormBase = deepcopy(self.normative_model)
                    # fold_norm_model.norm_conf.set_log_dir(
                    #     os.path.join(conf.log_dir, "folds", f"fold_{i_fold}")
                    # )
                    # fold_norm_model.norm_conf.set_save_dir(
                    #     os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
                    # )
                    # fold_norm_model.fit_predict(fit_data, predict_data)
                    # measures = predict_data.measures
                    # print(measures)
            return kfold_fit_predict_chunk_fn
        else:
            def fit_predict_chunk_fn(fit_data: NormData, predict_data: Optional[NormData]):
                if predict_data is None:
                    raise ValueError("predict_data is required for fit_predict without cross-validation")
                pid = os.getpid()
                print(f"I am {pid} and I am fitting on train data with mean: {fit_data.y.to_numpy().mean(axis=0)}")
                # self.normative_model.fit_predict(fit_data, predict_data)
                # measures = predict_data.measures
                # print(measures)
            return fit_predict_chunk_fn

    def submit_fit_jobs(self, fn: Callable, data: NormData) -> None:
        if self.parallelize:
            if self.job_type == "local":
                chunks = data.chunk(self.n_jobs)
                Parallel(n_jobs=self.n_jobs)(delayed(fn)(chunk) for chunk in chunks)
            elif self.job_type == "slurm":
                # Run a simple python script with the python path to verify that the correct python is used
                try:
                    # Method 1: Using subprocess.run (recommended for modern Python)
                    result = subprocess.run(
                        [self.python_path, "-c", "print('Hello from subprocess')"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    print("Command output:", result.stdout)
                    
                    # Method 2: Using subprocess.Popen (more control over process)
                    # process = subprocess.Popen(
                    #     [self.python_path, "-c", "print('Hello from subprocess')"],
                    #     stdout=subprocess.PIPE,
                    #     stderr=subprocess.PIPE,
                    #     text=True
                    # )
                    # stdout, stderr = process.communicate()
                    # print("Command output:", stdout)
                    # if stderr:
                    #     print("Errors:", stderr)
                    
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with exit code {e.returncode}")
                    print("Error output:", e.stderr)
                except Exception as e:
                    print(f"Failed to run command: {str(e)}")
        else:
            fn(data)

    def submit_fit_predict_jobs(self, fn: Callable, fit_data: NormData, predict_data: Optional[NormData] = None) -> None:
        if self.parallelize:
            fit_chunks = fit_data.chunk(self.n_jobs)
            if predict_data is not None:
                predict_chunks = predict_data.chunk(self.n_jobs)
            else:
                predict_chunks = [None] * self.n_jobs
            Parallel(n_jobs=self.n_jobs)(delayed(fn)(fit_chunk, predict_chunk) for fit_chunk, predict_chunk in zip(fit_chunks, predict_chunks))
        else:
            fn(fit_data, predict_data)

    def predict(self, data: NormData) -> NormData:
        pass
        return data

    def get_chunks(self, data: NormData) -> Generator[NormData, Any, Any]:
        if self.parallelize:
            yield from data.chunk(self.n_jobs)
        else:
            yield data

    def wrap_in_slurm_job(self, fn: Callable) -> Callable:
        