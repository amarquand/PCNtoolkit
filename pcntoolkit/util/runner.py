import os
import re
import subprocess
import sys
import warnings
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional

import cloudpickle as pickle

# mp.set_start_method("spawn")
# from multiprocess import Pool
from joblib import Parallel, delayed

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
from pcntoolkit.normative_model.norm_factory import load_normative_model
from pcntoolkit.util.job_observer import JobObserver
from pcntoolkit.util.output import Errors, Messages, Output, Warnings


class Runner:
    cross_validate: bool = False
    cv_folds: int = 5
    parallelize: bool = False
    job_type: str = "local"
    n_jobs: int = 1
    n_cores: int = 1
    python_path: Optional[str] = None
    time_limit_str: str | int = "00:05:00"
    time_limit_seconds: int = 300
    memory: str = "5gb"
    log_dir: str = ""
    temp_dir: str = ""

    def __init__(
        self,
        cross_validate: bool = False,
        cv_folds: int = 5,
        parallelize: bool = False,
        job_type: str = "local",
        n_jobs: int = 1,
        n_cores: int = 1,
        python_path: Optional[str] = None,
        time_limit: str | int = "00:05:00",
        memory: str = "5GB",
        log_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self.parallelize = parallelize
        self.job_type = job_type
        self.n_jobs = n_jobs
        self.pool = None
        self.n_cores = n_cores
        if isinstance(time_limit, str):
            self.time_limit_str = time_limit
            try:
                self.time_limit_seconds = sum(
                    [
                        int(v) * 60**i
                        for i, v in enumerate(reversed(self.time_limit_str.split(":")))
                    ]
                )
            except Exception:
                Output.error(
                    Errors.ERROR_PARSING_TIME_LIMIT,
                    time_limit_str=self.time_limit_str,
                )
        elif isinstance(time_limit, int):
            self.time_limit_seconds = time_limit
            s = self.time_limit_seconds
            self.time_limit_str = f"{str(s//3600)}:{str((s//60)%60).rjust(2,"0")}:{str(s%60).rjust(2,"0")}"
        else:
            Output.error(Errors.ERROR_PARSING_TIME_LIMIT, time_limit_str=time_limit)

        self.memory = memory
        self.active_job_ids: Dict[str, str] = {}

        # Get Python path if not provided
        if not python_path:
            # Option 1: Get the interpreter path
            self.python_path = sys.executable
            Output.print(
                Messages.NO_PYTHON_PATH_SPECIFIED, python_path=self.python_path
            )
        else:
            self.python_path = python_path

        if log_dir is None:
            self.log_dir = os.path.abspath("logs")
            Output.print(Messages.NO_LOG_DIR_SPECIFIED, log_dir=self.log_dir)
        else:
            self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        if temp_dir is None:
            self.temp_dir = os.path.abspath("temp")
            Output.print(Messages.NO_TEMP_DIR_SPECIFIED, temp_dir=self.temp_dir)
        else:
            self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.cross_validate and self.cv_folds <= 1:
            Output.error(Errors.ERROR_CROSS_VALIDATION_FOLDS, cv_folds=self.cv_folds)

    def fit(
        self, model: NormBase, data: NormData, save_dir: Optional[str] = None
    ) -> None:
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        fn = self.get_fit_chunk_fn(model)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def fit_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: Optional[NormData] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        fn = self.get_fit_predict_chunk_fn(model)
        self.submit_jobs(
            fn,
            first_data_source=fit_data,
            second_data_source=predict_data,
            mode="binary",
        )
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def predict(
        self, model: NormBase, data: NormData, save_dir: Optional[str] = None
    ) -> None:
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        fn = self.get_predict_chunk_fn(model)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def transfer(
        self, model: NormBase, data: NormData, save_dir: Optional[str] = None
    ) -> None:
        self.save_dir = (
            save_dir if save_dir is not None else model.norm_conf.save_dir + "_transfer"
        )
        fn = self.get_transfer_chunk_fn(model)
        self.submit_jobs(fn, data, mode="unary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def transfer_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: NormData,
        save_dir: Optional[str] = None,
    ) -> None:
        self.save_dir = (
            save_dir if save_dir is not None else model.norm_conf.save_dir + "_transfer"
        )
        fn = self.get_transfer_predict_chunk_fn(model)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def extend(
        self, model: NormBase, data: NormData, save_dir: Optional[str] = None
    ) -> None:
        self.save_dir = (
            save_dir if save_dir is not None else model.norm_conf.save_dir + "_extend"
        )
        fn = self.get_extend_chunk_fn(model)
        self.submit_jobs(fn, data, mode="unary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def extend_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: NormData,
        save_dir: Optional[str] = None,
    ) -> None:
        self.save_dir = (
            save_dir if save_dir is not None else model.norm_conf.save_dir + "_extend"
        )
        fn = self.get_extend_predict_chunk_fn(model)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        self.job_observer = JobObserver(self.active_job_ids)
        self.job_observer.wait_for_jobs()

    def get_fit_chunk_fn(self, model: NormBase) -> Callable:
        """Returns a callable that fits a chunk of data"""
        if self.cross_validate:

            def kfold_fit_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    fold_norm_model: NormBase = deepcopy(model)
                    fold_norm_model.norm_conf.set_save_dir(
                        os.path.join(
                            model.norm_conf.save_dir, "folds", f"fold_{i_fold}"
                        )
                    )
                    fold_norm_model.fit(train_data)
                    fold_norm_model.save()

            return kfold_fit_chunk_fn
        else:

            def fit_chunk_fn(chunk: NormData):
                model.fit(chunk)

            return fit_chunk_fn

    def get_fit_predict_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_fit_predict_chunk_fn(
                all_data: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    Output.warning(
                        Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
                    )
                for i_fold, (fit_data, predict_data) in enumerate(
                    all_data.kfold_split(self.cv_folds)
                ):
                    fold_norm_model: NormBase = deepcopy(model)
                    fold_norm_model.norm_conf.set_save_dir(
                        os.path.join(
                            model.norm_conf.save_dir, "folds", f"fold_{i_fold}"
                        )
                    )
                    fold_norm_model.fit_predict(fit_data, predict_data)
                    fold_norm_model.save()
                    fold_norm_model.save_results(predict_data)

            return kfold_fit_predict_chunk_fn
        else:

            def fit_predict_chunk_fn(
                fit_data: NormData, predict_data: Optional[NormData]
            ):
                if predict_data is None:
                    Output.error(
                        Errors.ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION
                    )

                assert predict_data is not None  # Make the linter happy
                model.fit_predict(fit_data, predict_data)
                model.save()
                model.save_results(predict_data)

            return fit_predict_chunk_fn

    def get_predict_chunk_fn(self, model: NormBase) -> Callable:
        """Loads each fold model and predicts on the corresponding fold of data. Model n is used to predict on fold n."""
        if self.cross_validate:

            def kfold_predict_chunk_fn(chunk: NormData):
                conf = model.norm_conf
                for i_fold, (_, predict_data) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    fold_model = load_normative_model(
                        os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
                    )
                    fold_model.predict(predict_data)
                    fold_model.save_results(predict_data)

            return kfold_predict_chunk_fn
        else:

            def predict_chunk_fn(chunk: NormData):
                model.predict(chunk)
                model.save_results(chunk)

            return predict_chunk_fn

    def get_transfer_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_transfer_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    transfered_model = model.transfer(
                        train_data,
                        save_dir=os.path.join(self.save_dir, "folds", f"fold_{i_fold}"),
                    )
                    transfered_model.save()

            return kfold_transfer_chunk_fn
        else:

            def transfer_chunk_fn(data: NormData):
                transfered_model = model.transfer(data, save_dir=self.save_dir)
                transfered_model.save()

            return transfer_chunk_fn

    def get_transfer_predict_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_transfer_predict_chunk_fn(
                chunk: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    Output.warning(
                        Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
                    )
                for i_fold, (train_data, predict_data) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    transfered_model = model.transfer_predict(
                        train_data,
                        predict_data,
                        save_dir=os.path.join(self.save_dir, "folds", f"fold_{i_fold}"),
                    )
                    transfered_model.save()
                    transfered_model.save_results(predict_data)

            return kfold_transfer_predict_chunk_fn
        else:

            def transfer_predict_chunk_fn(train_data: NormData, predict_data: NormData):
                if predict_data is None:
                    raise ValueError(
                        "predict_data is required for transfer_predict without cross-validation"
                    )
                model.transfer_predict(train_data, predict_data, save_dir=self.save_dir)
                model.save_results(predict_data)

            return transfer_predict_chunk_fn

    def get_extend_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_extend_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    extended_model = model.extend(
                        train_data,
                        save_dir=os.path.join(self.save_dir, "folds", f"fold_{i_fold}"),
                    )
                    extended_model.save()

            return kfold_extend_chunk_fn
        else:

            def extend_chunk_fn(data: NormData):
                extended_model = model.extend(data, save_dir=self.save_dir)
                extended_model.save()

            return extend_chunk_fn

    def get_extend_predict_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_extend_predict_chunk_fn(
                chunk: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    warnings.warn("predict_data is not used in kfold cross-validation")
                for i_fold, (train_data, predict_data) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    extended_model = model.extend_predict(
                        train_data,
                        predict_data,
                        save_dir=os.path.join(self.save_dir, "folds", f"fold_{i_fold}"),
                    )
                    extended_model.save()
                    extended_model.save_results(predict_data)

            return kfold_extend_predict_chunk_fn
        else:

            def extend_predict_chunk_fn(train_data: NormData, predict_data: NormData):
                if predict_data is None:
                    raise ValueError(
                        "predict_data is required for extend_predict without cross-validation"
                    )
                extended_model = model.extend_predict(
                    train_data, predict_data, save_dir=self.save_dir
                )
                extended_model.save()
                extended_model.save_results(predict_data)

            return extend_predict_chunk_fn

    def save_callable_and_data(
        self,
        job_name: int | str,
        fn: Callable,
        chunk: NormData | tuple[NormData, NormData | None],
    ) -> tuple[str, str]:
        python_callable_path = os.path.join(
            self.temp_dir, f"python_callable_{job_name}.pkl"
        )
        data_path = os.path.join(self.temp_dir, f"data_{job_name}.pkl")
        os.makedirs(os.path.dirname(self.temp_dir), exist_ok=True)
        with open(python_callable_path, "wb") as f:
            pickle.dump(fn, f)
        with open(data_path, "wb") as f:
            pickle.dump(chunk, f)
        return python_callable_path, data_path

    def submit_jobs(
        self,
        fn: Callable,
        first_data_source: NormData,
        second_data_source: Optional[NormData] = None,
        mode: Literal["unary", "binary"] = "unary",
    ) -> None:
        """Submit jobs to the job scheduler.

        The predict_data argument is optional, and if it is not provided, None is passed to the function.

        Parameters
        ----------
        fn : Callable
            Function to call. It should take two arguments.
        fit_data : NormData
            Data to fit the model on
        predict_data : Optional[NormData], optional
            Data to predict on, by default None
        """
        if self.parallelize:
            first_chunks = first_data_source.chunk(self.n_jobs)

            if second_data_source is not None:
                second_chunks = second_data_source.chunk(self.n_jobs)
            else:
                second_chunks = [None] * self.n_jobs

            if self.job_type == "local":
                delayed_functions = []
                for first_chunk, second_chunk in zip(first_chunks, second_chunks):
                    if mode == "unary":
                        delayed_functions.append(delayed(fn)(first_chunk))
                    elif mode == "binary":
                        delayed_functions.append(delayed(fn)(first_chunk, second_chunk))
                Parallel(n_jobs=self.n_jobs, timeout=self.time_limit_seconds)(
                    delayed_functions
                )

            else:
                for i, (first_chunk, second_chunk) in enumerate(
                    zip(first_chunks, second_chunks)
                ):
                    if mode == "unary":
                        chunk_tuple = first_chunk
                    elif mode == "binary":
                        chunk_tuple = (first_chunk, second_chunk)
                    python_callable_path, data_path = self.save_callable_and_data(
                        i, fn, chunk_tuple
                    )
                    if self.job_type == "slurm":
                        command = self.wrap_in_slurm_job(
                            i, python_callable_path, data_path
                        )
                    elif self.job_type == "torque":
                        command = self.wrap_in_torque_job(
                            i, python_callable_path, data_path
                        )
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    stdout, stderr = process.communicate()
                    job_id = re.search(r"Submitted batch job (\d+)", stdout)
                    if job_id:
                        self.active_job_ids[f"job_{i}"] = job_id.group(1)
                    elif stderr:
                        Output.error(
                            Errors.ERROR_SUBMITTING_JOB, job_id=i, stderr=stderr
                        )
        else:
            if mode == "unary":
                chunk_tuple = (first_data_source,)
            elif mode == "binary":
                chunk_tuple = (first_data_source, second_data_source)
            fn(*chunk_tuple)

    def wrap_in_slurm_job(
        self, job_name: int | str, python_callable_path: str, data_path: str
    ) -> list[str]:
        job_path = os.path.join(self.temp_dir, f"job_{job_name}.sh")
        current_file_path = os.path.abspath(__file__)
        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/bash
                    
#SBATCH --partition=batch
#SBATCH --job-name=normative_{job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={self.n_cores}
#SBATCH --time={self.time_limit_str}
#SBATCH --mem={self.memory}
#SBATCH --error={os.path.join(self.log_dir, f"{job_name}.err")}
#SBATCH --output={os.path.join(self.log_dir, f"{job_name}.out")}
#SBATCH --mail-type=FAIL

{self.python_path} {current_file_path} {python_callable_path} {data_path}
"""
            )
        return ["sbatch", job_path]

    def wrap_in_torque_job(
        self, job_name: int | str, python_callable_path: str, data_path: str
    ) -> list[str]:
        job_path = os.path.join(self.temp_dir, f"job_{job_name}.sh")
        current_file_path = os.path.abspath(__file__)

        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/sh

#PBS -N normative_{job_name}
#PBS -l nodes=1:ppn={self.n_cores}
#PBS -l walltime={self.time_limit_str}
#PBS -l mem={self.memory}
#PBS -o {os.path.join(self.log_dir, f"{job_name}.out")}
#PBS -e {os.path.join(self.log_dir, f"{job_name}.err")}
#PBS -m a

{self.python_path} {current_file_path} {python_callable_path} {data_path}
"""
            )

        return ["qsub", job_path]

    def load_model(self, fold_index: Optional[int] = 0) -> NormBase:
        if self.cross_validate:
            path = os.path.join(self.save_dir, "folds", f"fold_{fold_index}")
            return load_normative_model(path)
        else:
            return load_normative_model(self.save_dir)

    @classmethod
    def from_args(cls, args: dict) -> "Runner":
        filtered_args = {
            k: v for k, v in args.items() if k in list(cls.__dict__.keys())
        }
        return cls(**filtered_args)


def load_and_execute(args):
    with open(args[0], "rb") as executable_path:
        fn = pickle.load(executable_path)
    with open(args[1], "rb") as data_path:
        data = pickle.load(data_path)
    if isinstance(data, tuple):
        fn(data[0], data[1])
    else:
        fn(data)


if __name__ == "__main__":
    load_and_execute(sys.argv[1:])
