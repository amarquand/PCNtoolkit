import json
import os
import re
import subprocess
import sys
import uuid
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional

import cloudpickle as pickle

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
        save_dir: Optional[str] = None,
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
                self.time_limit_seconds = sum([int(v) * 60**i for i, v in enumerate(reversed(self.time_limit_str.split(":")))])
            except Exception:
                raise Output.error(
                    Errors.ERROR_PARSING_TIME_LIMIT,
                    time_limit_str=self.time_limit_str,
                )
        elif isinstance(time_limit, int):
            self.time_limit_seconds = time_limit
            s = self.time_limit_seconds
            self.time_limit_str = f"{str(s//3600)}:{str((s//60)%60).rjust(2,"0")}:{str(s%60).rjust(2,"0")}"
        else:
            raise Output.error(Errors.ERROR_PARSING_TIME_LIMIT, time_limit_str=time_limit)

        self.memory = memory
        self.all_jobs: Dict[str, str] = {}
        self.job_commands: Dict[str, list[str]] = {}  # Save the commands for re-submission
        self.active_jobs: Dict[str, str] = {}
        self.failed_jobs: Dict[str, str] = {}

        # Get Python path if not provided
        if not python_path:
            # Option 1: Get the interpreter path
            self.python_path = sys.executable
            Output.print(Messages.NO_PYTHON_PATH_SPECIFIED, python_path=self.python_path)
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
            raise Output.error(Errors.ERROR_CROSS_VALIDATION_FOLDS, cv_folds=self.cv_folds)

    def fit(self, model: NormBase, data: NormData, save_dir: Optional[str] = None, observe: bool = True) -> None:
        """
        Fit a normative model on a dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to fit.
        data : NormData
            The data to fit the model on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.

        Returns
        -------
        None
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        self.set_unique_temp_and_log_dir()
        fn = self.get_fit_chunk_fn(model)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        if observe:
            # If we want to observer the jobs, we wait for them, and then get the failed jobs after it's done
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            self.load_model(into=model)
        else:
            # Else, we save the runner state after all jobs are submitted
            # The load function will check for failed jobs
            self.save()

    def set_unique_temp_and_log_dir(self):
        my_new_uuid = str(uuid.uuid4())
        self.temp_dir = os.path.join(self.temp_dir, my_new_uuid)
        self.log_dir = os.path.join(self.log_dir, my_new_uuid)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def fit_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: Optional[NormData] = None,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> None:
        """
        Fit a normative model on a dataset and predict on another dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to fit.
        fit_data : NormData
            The data to fit the model on.
        predict_data : Optional[NormData], optional
            The data to predict on. If None, the function will predict on the fit_data.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish, then load the model into the model object
            If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.

        Returns
        -------
        None
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        self.set_unique_temp_and_log_dir()
        fn = self.get_fit_predict_chunk_fn(model)
        self.submit_jobs(
            fn,
            first_data_source=fit_data,
            second_data_source=predict_data,
            mode="binary",
        )
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            self.load_model(into=model)
        else:
            self.save()

    def predict(self, model: NormBase, data: NormData, save_dir: Optional[str] = None, observe: bool = True) -> None:
        """
        Predict on a dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to predict on.
        data : NormData
            The data to predict on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish.

        Returns
        -------
        None
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir
        self.set_unique_temp_and_log_dir()
        fn = self.get_predict_chunk_fn(model)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
        else:
            self.save()

    def transfer(self, model: NormBase, data: NormData, save_dir: Optional[str] = None, observe: bool = True) -> NormBase | None:
        """
        Transfer a normative model to a new dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to transfer.
        data : NormData
            The data to transfer the model to.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then return the transfered model.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | None
            The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir + "_transfer"
        self.set_unique_temp_and_log_dir()
        fn = self.get_transfer_chunk_fn(model)
        self.submit_jobs(fn, data, mode="unary")
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            return self.load_model()
        else:
            self.save()

    def transfer_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> NormBase | None:
        """
        Transfer a normative model to a new dataset and predict on another dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to transfer.
        fit_data : NormData
            The data to transfer the model to.
        predict_data : NormData
            The data to predict on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | No   ne
            The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir + "_transfer"
        self.set_unique_temp_and_log_dir()
        fn = self.get_transfer_predict_chunk_fn(model)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            return self.load_model()
        else:
            self.save()

    def extend(self, model: NormBase, data: NormData, save_dir: Optional[str] = None, observe: bool = True) -> NormBase | None:
        """
        Extend a normative model on a dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to extend.
        data : NormData
            The data to extend the model on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | None
            The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir + "_extend"
        self.set_unique_temp_and_log_dir()
        fn = self.get_extend_chunk_fn(model)
        self.submit_jobs(fn, data, mode="unary")
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            return self.load_model()
        else:
            self.save()

    def extend_predict(
        self,
        model: NormBase,
        fit_data: NormData,
        predict_data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> NormBase | None:
        """
        Extend a normative model on a dataset and predict on another dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to extend.
        fit_data : NormData
            The data to extend the model on.
        predict_data : NormData
            The data to predict on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | None
            The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        self.save_dir = save_dir if save_dir is not None else model.norm_conf.save_dir + "_extend"
        self.set_unique_temp_and_log_dir()
        fn = self.get_extend_predict_chunk_fn(model)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        if observe:
            self.job_observer = JobObserver(self.all_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            return self.load_model()
        else:
            self.save()

    def get_fit_chunk_fn(self, model: NormBase) -> Callable:
        """Returns a callable that fits a model on a chunk of data"""
        if self.cross_validate:

            def kfold_fit_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(chunk.kfold_split(self.cv_folds)):
                    fold_norm_model: NormBase = deepcopy(model)
                    fold_norm_model.norm_conf.set_save_dir(os.path.join(model.norm_conf.save_dir, "folds", f"fold_{i_fold}"))
                    fold_norm_model.fit(train_data)
                    fold_norm_model.save()

            return kfold_fit_chunk_fn
        else:

            def fit_chunk_fn(chunk: NormData):
                model.fit(chunk)

            return fit_chunk_fn

    def get_fit_predict_chunk_fn(self, model: NormBase) -> Callable:
        """Returns a callable that fits a model on a chunk of data and predicts on another chunk of data"""
        if self.cross_validate:

            def kfold_fit_predict_chunk_fn(all_data: NormData, unused_predict_data: Optional[NormData] = None):
                if unused_predict_data is not None:
                    Output.warning(Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION)
                for i_fold, (fit_data, predict_data) in enumerate(all_data.kfold_split(self.cv_folds)):
                    fold_norm_model: NormBase = deepcopy(model)
                    fold_norm_model.norm_conf.set_save_dir(os.path.join(model.norm_conf.save_dir, "folds", f"fold_{i_fold}"))
                    fold_norm_model.fit_predict(fit_data, predict_data)
                    fold_norm_model.save()
                    fold_norm_model.save_results(predict_data)

            return kfold_fit_predict_chunk_fn
        else:

            def fit_predict_chunk_fn(fit_data: NormData, predict_data: Optional[NormData]):
                if predict_data is None:
                    raise Output.error(Errors.ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION)

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
                for i_fold, (_, predict_data) in enumerate(chunk.kfold_split(self.cv_folds)):
                    fold_model = load_normative_model(os.path.join(conf.save_dir, "folds", f"fold_{i_fold}"))
                    fold_model.predict(predict_data)
                    fold_model.save_results(predict_data)

            return kfold_predict_chunk_fn
        else:

            def predict_chunk_fn(chunk: NormData):
                model.predict(chunk)
                model.save_results(chunk)

            return predict_chunk_fn

    def get_transfer_chunk_fn(self, model: NormBase) -> Callable:
        """Returns a callable that transfers a model on a chunk of data"""
        if self.cross_validate:

            def kfold_transfer_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(chunk.kfold_split(self.cv_folds)):
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

            def kfold_transfer_predict_chunk_fn(chunk: NormData, unused_predict_data: Optional[NormData] = None):
                if unused_predict_data is not None:
                    Output.warning(Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION)
                for i_fold, (train_data, predict_data) in enumerate(chunk.kfold_split(self.cv_folds)):
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
                    raise Output.error(Errors.ERROR_PREDICT_DATA_REQUIRED)
                model.transfer_predict(train_data, predict_data, save_dir=self.save_dir)
                model.save_results(predict_data)

            return transfer_predict_chunk_fn

    def get_extend_chunk_fn(self, model: NormBase) -> Callable:
        if self.cross_validate:

            def kfold_extend_chunk_fn(chunk: NormData):
                for i_fold, (train_data, _) in enumerate(chunk.kfold_split(self.cv_folds)):
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

            def kfold_extend_predict_chunk_fn(chunk: NormData, unused_predict_data: Optional[NormData] = None):
                if unused_predict_data is not None:
                    Output.warning(Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION)
                for i_fold, (train_data, predict_data) in enumerate(chunk.kfold_split(self.cv_folds)):
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
                    raise Output.error(Errors.ERROR_PREDICT_DATA_REQUIRED)
                extended_model = model.extend_predict(train_data, predict_data, save_dir=self.save_dir)
                extended_model.save()
                extended_model.save_results(predict_data)

            return extend_predict_chunk_fn

    def save_callable_and_data(
        self,
        job_name: int | str,
        fn: Callable,
        chunk: NormData | tuple[NormData, NormData | None],
    ) -> tuple[str, str]:
        python_callable_path = os.path.join(self.temp_dir, f"python_callable_{job_name}.pkl")
        data_path = os.path.join(self.temp_dir, f"data_{job_name}.pkl")
        os.makedirs(os.path.dirname(self.temp_dir), exist_ok=True)
        with open(python_callable_path, "wb") as f:
            pickle.dump(fn, f)
        with open(data_path, "wb") as f:
            pickle.dump(chunk, f)
        return python_callable_path, data_path

    def check_job_status(self, job_name: str) -> tuple[bool, bool, Optional[str]]:
        """Check if a job has failed by looking for success file.

        Returns
        -------
        tuple[bool, bool, Optional[str]]
            (is_running, finished_with_error, error_message)
            If job is still running, returns (True, False, None)
            If job finished successfully, returns (False, False, None)
            If job failed, returns (False, True, error_message)
        """
        job_id = self.all_jobs[job_name]
        if self.job_type == "local":
            try:
                os.kill(int(job_id), 0)  # Check if process still running
                return True, False, None  # Still running
            except ProcessLookupError:
                # Process finished, check for success file
                success_file = os.path.join(self.log_dir, f"{job_name}.success")
                if os.path.exists(success_file):
                    return False, False, None  # Finished successfully
                else:
                    # Job finished but no success file - read error output
                    error_file = os.path.join(self.log_dir, f"{job_name}.err")
                    if os.path.exists(error_file):
                        with open(error_file, "r") as f:
                            return False, True, f.read().strip()
                    return False, True, "Job failed without error output"
        else:
            # For cluster jobs, first check if job is still in queue/running
            is_running = False
            if self.job_type == "slurm":
                result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
                is_running = job_id in result.stdout
            elif self.job_type == "torque":
                result = subprocess.run(["qstat", job_id], capture_output=True, text=True)
                is_running = result.returncode == 0
            if is_running:
                return True, False, None  # Still running

            # Job not running, check for success file
            success_file = os.path.join(self.log_dir, f"{job_name}.success")
            if os.path.exists(success_file):
                return False, False, None  # Finished successfully

            # Job finished but no success file - read error output
            error_file = os.path.join(self.log_dir, f"{job_name}.err")
            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    return False, True, f.read().strip()
            return False, True, "Job failed without error output"

    def check_jobs_status(self) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Check all jobs in active_job_ids for errors.

        Returns
        -------
        tuple[Dict[str, str], Dict[str, str], Dict[str, str]]
            A tuple containing:
            - A dictionary mapping job names to job IDs for running jobs
            - A dictionary mapping job names to error messages for failed jobs
            - A dictionary mapping job names to job IDs for finished jobs
        """
        running_jobs = {}
        finished_jobs = {}
        failed_jobs = {}
        for job_name, job_id in self.all_jobs.items():
            is_running, finished_with_error, error_msg = self.check_job_status(job_name)
            if not is_running:
                if finished_with_error:
                    failed_jobs[job_name] = error_msg
                else:
                    finished_jobs[job_name] = job_id
            elif is_running:
                running_jobs[job_name] = job_id
        return running_jobs, finished_jobs, failed_jobs

    def run_command_with_subprocess(self, cmd: list[str]):
        """Run using subprocess"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        return process

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
            second_chunks = [None] * self.n_jobs if second_data_source is None else second_data_source.chunk(self.n_jobs)

            self.all_jobs.clear()
            self.job_commands.clear()

            if self.job_type == "local":
                # Create all job files first
                jobs_to_run = []
                for i, (first_chunk, second_chunk) in enumerate(zip(first_chunks, second_chunks)):
                    job_name = f"job_{i}"
                    if mode == "unary":
                        chunk_tuple = first_chunk
                    else:
                        chunk_tuple = (first_chunk, second_chunk)

                    python_callable_path, data_path = self.save_callable_and_data(job_name, fn, chunk_tuple)
                    command = self.wrap_in_local_job(job_name, python_callable_path, data_path)
                    jobs_to_run.append((job_name, command))
                    self.job_commands[job_name] = command

                # Wait for all jobs to finish
                all_processes = []
                import multiprocessing as mp
                with mp.Pool(processes=self.n_jobs) as pool:
                    results = pool.map(subprocess.Popen, [command for _, command in jobs_to_run])
                    # all_processes.append(process)  # noqa: F821
                # for process in all_processes:
                #     process.wait()
                #     stdout, stderr = process.communicate()
                #     print(stdout, stderr)
                import time
                time.sleep(1000)

            else:
                for i, (first_chunk, second_chunk) in enumerate(zip(first_chunks, second_chunks)):
                    job_name = f"job_{i}"
                    if mode == "unary": 
                        chunk_tuple = first_chunk
                    else:
                        chunk_tuple = (first_chunk, second_chunk)

                    python_callable_path, data_path = self.save_callable_and_data(job_name, fn, chunk_tuple)

                    if self.job_type == "slurm":
                        command = self.wrap_in_slurm_job(job_name, python_callable_path, data_path)
                    elif self.job_type == "torque":
                        command = self.wrap_in_torque_job(job_name, python_callable_path, data_path)

                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    stdout, stderr = process.communicate()
                    job_id = re.search(r"Submitted batch job (\d+)", stdout)
                    if job_id:
                        self.all_jobs[job_name] = job_id.group(1)
                        self.job_commands[job_name] = command
                    elif stderr:
                        raise Output.error(Errors.ERROR_SUBMITTING_JOB, job_id=job_name, stderr=stderr)

        else:
            if mode == "unary":
                chunk_tuple = (first_data_source,)
            elif mode == "binary":
                chunk_tuple = (first_data_source, second_data_source)
            fn(*chunk_tuple)

    def wrap_in_slurm_job(self, job_name: int | str, python_callable_path: str, data_path: str) -> list[str]:
        job_path = os.path.join(self.temp_dir, f"{job_name}.sh")
        current_file_path = os.path.abspath(__file__)
        out_file = os.path.join(self.log_dir, f"{job_name}.out")
        err_file = os.path.join(self.log_dir, f"{job_name}.err")
        success_file = os.path.join(self.log_dir, f"{job_name}.success")

        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/bash
    
#SBATCH --job-name=normative_{job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={self.n_cores}
#SBATCH --time={self.time_limit_str}
#SBATCH --mem={self.memory}
#SBATCH --error={err_file}
#SBATCH --output={out_file}

{self.python_path} {current_file_path} {python_callable_path} {data_path}
exit_code=$?
if [ $exit_code -eq 0 ]; then
    touch {success_file}
fi
exit $exit_code
"""
            )
        return ["sbatch", job_path]

    def wrap_in_torque_job(self, job_name: int | str, python_callable_path: str, data_path: str) -> list[str]:
        job_path = os.path.join(self.temp_dir, f"{job_name}.sh")
        current_file_path = os.path.abspath(__file__)
        out_file = os.path.join(self.log_dir, f"{job_name}.out")
        err_file = os.path.join(self.log_dir, f"{job_name}.err")
        success_file = os.path.join(self.log_dir, f"{job_name}.success")

        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/sh

#PBS -N normative_{job_name}
#PBS -l nodes=1:ppn={self.n_cores}
#PBS -l walltime={self.time_limit_str}
#PBS -l mem={self.memory}
#PBS -o {out_file}
#PBS -e {err_file}
#PBS -m a

{self.python_path} {current_file_path} {python_callable_path} {data_path}
exit_code=$?
if [ $exit_code -eq 0 ]; then
    touch {success_file}
fi
exit $exit_code
"""
            )

        return ["qsub", job_path]

    def wrap_in_local_job(self, job_name: int | str, python_callable_path: str, data_path: str) -> list[str]:
        job_path = os.path.join(self.temp_dir, f"{job_name}.sh")
        current_file_path = os.path.abspath(__file__)
        success_file = os.path.join(self.log_dir, f"{job_name}.success")
        out_file = os.path.join(self.log_dir, f"{job_name}.out")
        err_file = os.path.join(self.log_dir, f"{job_name}.err")

        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/bash
# Execute Python script with unbuffered output
PYTHONUNBUFFERED=1 {self.python_path} -m pcntoolkit.util.run_job {python_callable_path} {data_path} > {out_file} 2> {err_file}
exit_code=$?

if [ $exit_code -eq 0 ]; then
    touch {success_file}
fi
exit $exit_code
"""
            )
        os.chmod(job_path, 0o755)  # Make script executable
        return ["bash", job_path]

    def load_model(self, fold_index: Optional[int] = 0, into: NormBase | None = None) -> NormBase:
        if self.cross_validate:
            path = os.path.join(self.save_dir, "folds", f"fold_{fold_index}")
            return load_normative_model(path, into=into)
        else:
            return load_normative_model(self.save_dir, into=into)

    def save(self) -> None:
        """Save the runner state to a JSON file in the save directory."""
        runner_state = {
            "all_jobs": self.all_jobs,
            "log_dir": self.log_dir,
            "temp_dir": self.temp_dir,
            "job_type": self.job_type,
            "n_jobs": self.n_jobs,
            "save_dir": self.save_dir,
            "job_commands": self.job_commands,
            "active_jobs": self.active_jobs,
            "failed_jobs": self.failed_jobs,
        }

        runner_file = os.path.join(self.temp_dir, "runner_state.json")
        Output.print(Messages.SAVING_RUNNER_STATE, runner_file=runner_file)
        with open(runner_file, "w") as f:
            json.dump(runner_state, f, indent=4)

    @classmethod
    def load_from_state(cls, runner_file: str) -> "Runner":
        """Load a runner from a saved state.

        Parameters
        ----------
        runner_file : str
            Path to the runner state file

        Returns
        -------
        Runner
            A runner instance with the saved state
        """
        Output.print(Messages.LOADING_RUNNER_STATE, runner_file=runner_file)
        with open(runner_file, "r") as f:
            state = json.load(f)

        runner = cls(job_type=state["job_type"], n_jobs=state["n_jobs"], log_dir=state["log_dir"], temp_dir=state["temp_dir"])
        runner.save_dir = state["save_dir"]
        runner.job_commands = state["job_commands"]
        runner.all_jobs = state["all_jobs"]
        runner.active_jobs, runner.finished_jobs, runner.failed_jobs = runner.check_jobs_status()

        Output.print(
            Messages.RUNNER_LOADED,
            n_active_jobs=len(runner.active_jobs),
            n_finished_jobs=len(runner.finished_jobs),
            n_failed_jobs=len(runner.failed_jobs),
        )
        return runner

    @classmethod
    def from_args(cls, args: dict) -> "Runner":
        filtered_args = {k: v for k, v in args.items() if k in list(cls.__dict__.keys())}
        return cls(**filtered_args)

    def re_submit_failed_jobs(self, observe: bool = True) -> None:
        for job_name, command in self.job_commands.items():
            if job_name in self.failed_jobs:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
                )
                if self.job_type == "local":
                    self.active_jobs[job_name] = str(process.pid)
                    self.job_commands[job_name] = command
                else:
                    stdout, stderr = process.communicate()
                    job_id = re.search(r"Submitted batch job (\d+)", stdout)
                    if job_id:
                        self.all_jobs[job_name] = job_id.group(1)
                        self.job_commands[job_name] = command
                    elif stderr:
                        raise Output.error(Errors.ERROR_SUBMITTING_JOB, job_id=job_name, stderr=stderr)

        self.failed_jobs.clear()
        if observe:
            self.job_observer = JobObserver(self.active_jobs, self.job_type, self.log_dir)
            self.job_observer.wait_for_jobs()
            self.active_jobs, self.finished_jobs, self.failed_jobs = self.check_jobs_status()

        self.save()


def load_and_execute(args):
    print("Loading and executing")
    with open(args[0], "rb") as executable_path:
        fn = pickle.load(executable_path)
    with open(args[1], "rb") as data_path:
        data = pickle.load(data_path)
    if isinstance(data, tuple):
        fn(data[0], data[1])
    else:
        fn(data)


if __name__ == "__main__":
    print("Running the runner")
    load_and_execute(sys.argv[1:])
