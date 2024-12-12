import os
import subprocess
import sys
import warnings
from copy import deepcopy
from typing import Any, Callable, Generator, Optional, Dict, List
import re
from dataclasses import dataclass
from datetime import datetime
import time

import xarray as xr

# mp.set_start_method("spawn")
# from multiprocess import Pool
from joblib import Parallel, delayed

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model.norm_base import NormBase
import dill

@dataclass
class JobStatus:
    job_id: str
    name: str
    state: str
    time: str
    nodes: str

class Runner:
    def __init__(
        self,
        normative_model: NormBase,
        cross_validate: bool = False,
        cv_folds: int = 5,
        parallelize: bool = False,
        job_type: str = "local",
        n_jobs: int = 1,
        n_cores: int = 1,
        python_path: Optional[str] = None,
        walltime: str = "00:05:00",
        memory: str = "5GB",
        log_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        self.normative_model = normative_model
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self.parallelize = parallelize
        self.job_type = job_type
        self.n_jobs = n_jobs
        self.pool = None
        self.n_cores = n_cores
        self.walltime = walltime
        self.memory = memory
        self.active_job_ids: Dict[str, str] = {}
        
        # Get Python path if not provided
        if not python_path:
            # Option 1: Get the interpreter path
            self.python_path = sys.executable
            print("No python path specified. Using interpreter path of current process:", self.python_path)
        else:
            self.python_path = python_path

        if log_dir is None:
            self.log_dir = os.path.abspath("logs")
            print(f"No log directory specified. Using default log directory: {self.log_dir}")
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        if temp_dir is None:
            self.temp_dir = os.path.abspath("temp")
            print(f"No temp directory specified. Using default temp directory: {self.temp_dir}")
        else:
            self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.cross_validate and self.cv_folds <= 1:
            raise ValueError(
                "If cross-validation is enabled, cv_folds must be greater than 1"
            )
        if not self.normative_model.norm_conf.savemodel:
            warnings.warn("Model saving is disabled. Results will not be saved.")


    def fit(self, data: NormData) -> None:
        fn = self.get_fit_chunk_fn()
        self.submit_fit_jobs(fn, data)
        self.wait_for_jobs()

    def fit_predict(self, fit_data: NormData, predict_data: Optional[NormData] = None) -> None:
        fn = self.get_fit_predict_chunk_fn()
        self.submit_fit_predict_jobs(fn, fit_data, predict_data)
        self.wait_for_jobs()

    def get_fit_chunk_fn(self) -> Callable:
        """ Returns a callable that fits a chunk of data """
        if self.cross_validate:
            def kfold_fit_chunk_fn(chunk: NormData):
                conf = self.normative_model.norm_conf
                for i_fold, (train_data, _) in enumerate(chunk.kfold_split(self.cv_folds)):
                    conf = self.normative_model.norm_conf
                    pid = os.getpid()
                    y:xr.DataArray = train_data.y
                    print(f"I am {pid} and I am fitting fold {i_fold}, on train data with mean: {y.to_numpy().mean(axis=0)}")
                    fold_norm_model: NormBase = deepcopy(self.normative_model)
                    fold_norm_model.norm_conf.set_save_dir(
                        os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
                    )
                    fold_norm_model.fit(train_data)
            return kfold_fit_chunk_fn
        else:
            def fit_chunk_fn(chunk: NormData):
                conf = self.normative_model.norm_conf
                pid = os.getpid()
                y:xr.DataArray = chunk.y
                print(f"I am {pid} and I am fitting, on train data with mean: {y.to_numpy().mean(axis=0)}")
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
                    print(f"I am {pid} and I am fitting and predicting fold {i_fold}, on train data with mean: {y.to_numpy().mean(axis=0)}")
                    fold_norm_model: NormBase = deepcopy(self.normative_model)
                    fold_norm_model.norm_conf.set_save_dir(
                        os.path.join(conf.save_dir, "folds", f"fold_{i_fold}")
                    )
                    fold_norm_model.fit_predict(fit_data, predict_data)
                    measures = predict_data.measures
                    print(measures)
            return kfold_fit_predict_chunk_fn
        else:
            def fit_predict_chunk_fn(fit_data: NormData, predict_data: Optional[NormData]):
                if predict_data is None:
                    raise ValueError("predict_data is required for fit_predict without cross-validation")
                pid = os.getpid()
                print(f"I am {pid} and I am fitting on train data with mean: {fit_data.y.to_numpy().mean(axis=0)}")
                self.normative_model.fit_predict(fit_data, predict_data)
                measures = predict_data.measures
                print(measures)
            return fit_predict_chunk_fn

    def submit_fit_jobs(self, fn: Callable, data: NormData) -> None:
        if self.parallelize:
            if self.job_type == "local":
                chunks = data.chunk(self.n_jobs)
                Parallel(n_jobs=self.n_jobs)(delayed(fn)(chunk) for chunk in chunks)
            elif self.job_type == "slurm":
                chunks = data.chunk(self.n_jobs)
                for i, chunk in enumerate(chunks):
                    job_path = self.wrap_in_slurm_job(i, fn, chunk)
                    # Capture the output of sbatch to get job ID
                    process = subprocess.Popen(
                        ["sbatch", job_path], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = process.communicate()
                    
                    # Parse job ID from output (typically "Submitted batch job 123456")
                    job_id = re.search(r"Submitted batch job (\d+)", stdout)
                    if job_id:
                        self.active_job_ids[f"job_{i}"] = job_id.group(1)
                    elif stderr:
                        print(f"Error submitting job {i}: {stderr}")
        else:
            fn(data)


    def submit_fit_predict_jobs(self, fn: Callable, fit_data: NormData, predict_data: Optional[NormData] = None) -> None:
        if self.parallelize:
            fit_chunks = fit_data.chunk(self.n_jobs)
            if predict_data is not None:
                predict_chunks = predict_data.chunk(self.n_jobs)
            else:
                predict_chunks = [None] * self.n_jobs
                
            if self.job_type == "local":
                Parallel(n_jobs=self.n_jobs)(delayed(fn)(fit_chunk, predict_chunk) 
                    for fit_chunk, predict_chunk in zip(fit_chunks, predict_chunks))
            elif self.job_type == "slurm":
                for i, (fit_chunk, predict_chunk) in enumerate(zip(fit_chunks, predict_chunks)):
                    job_path = self.wrap_in_slurm_job(i, fn, (fit_chunk, predict_chunk))
                    # Use Popen instead of run
                    process = subprocess.Popen(
                        ["sbatch", job_path], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = process.communicate()
                    
                    # Parse job ID from output (typically "Submitted batch job 123456")
                    job_id = re.search(r"Submitted batch job (\d+)", stdout)
                    if job_id:
                        self.active_job_ids[f"job_{i}"] = job_id.group(1)
                    elif stderr:
                        print(f"Error submitting job {i}: {stderr}")
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

    def wrap_in_slurm_job(self, job_name: int | str, fn: Callable , chunk: NormData | tuple[NormData, NormData]) -> Callable:
        # Save all the necessary objects to the temp directory
        executable_path = os.path.join(self.temp_dir, f"slurm_executable_{job_name}.pkl")
        job_path = os.path.join(self.temp_dir, f"slurm_job_{job_name}.sh")
        data_path = os.path.join(self.temp_dir, f"slurm_data_{job_name}.pkl")
        os.makedirs(os.path.dirname(executable_path), exist_ok=True)
        with open(executable_path, "wb") as f:
            dill.dump(fn, f)
        with open(data_path, "wb") as f:
            dill.dump(chunk, f)

        # Get the current file path (we use this script as the entrypoint)
        current_file_path = os.path.abspath(__file__)

        # Write the job script
        with open(job_path, "w") as f:
            f.write(f"""#!/bin/bash
                    
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --time={self.walltime}
#SBATCH --mail-type=FAIL
#SBATCH --partition=batch
#SBATCH --mem={self.memory}
#SBATCH --cpus-per-task={self.n_cores}
#SBATCH --output={os.path.join(self.log_dir, f"{job_name}.out")}
#SBATCH --error={os.path.join(self.log_dir, f"{job_name}.err")}

{self.python_path} {current_file_path} {executable_path} {data_path}
""")
        return job_path
        
    def get_job_statuses(self) -> List[JobStatus]:
        """Get status of all tracked jobs."""
        if not self.active_job_ids:
            return []

        # Get all job statuses at once
        process = subprocess.Popen(
            ["squeue", "--format=%i|%j|%T|%M|%N", "--noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        statuses = []
        if process.returncode is None or process.returncode == 0:
            # Split output into lines and filter out empty lines
            lines = [line.strip() for line in stdout.split('\n') if line.strip()]
            for line in lines:
                try:
                    job_id, name, state, time, nodes = line.split('|')
                    if job_id in list(self.active_job_ids.values()):
                        statuses.append(JobStatus(
                            job_id=job_id,
                            name=name,
                            state=state,
                            time=time,
                            nodes=nodes
                    ))
                except ValueError as e:
                    print(f"Error parsing job status line: {line}")
                    print(f"Error details: {e}")
        elif stderr:
            print(f"Error getting job statuses: {stderr}")

        return statuses


    def wait_for_jobs(self, check_interval=5):
        """Wait for all submitted jobs to complete.
        
        Args:
            check_interval (int): Time in seconds between job status checks
        """
        # Detect if we're running in a Jupyter notebook
        in_notebook = True
        try:
            shell = get_ipython().__class__.__name__
            if shell not in ['ZMQInteractiveShell', 'Shell']:
                in_notebook = False
        except NameError:
            in_notebook = False
            
        if in_notebook:
            # Notebook-friendly display without ANSI codes
            from IPython.display import clear_output
            
            while self.active_job_ids:
                clear_output(wait=True)
                print("Job Status Monitor:")
                print("-" * 60)
                print("Job ID     Name     State     Time     Nodes")
                print("-" * 60)
                
                # Get and display current statuses
                statuses = self.get_job_statuses()
                for status in statuses:
                    print("{:<10} {:<8} {:<9} {:<8} {:<8}".format(
                        status.job_id, status.name, status.state, status.time, status.nodes
                    ))
                
                # Check for completed jobs
                completed_jobs = []
                for job_name, job_id in list(self.active_job_ids.items()):
                    if not any(s.job_id == job_id for s in statuses) or \
                       any(s.job_id == job_id and s.state in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                        completed_jobs.append(job_name)
                
                # Remove completed jobs
                for job_name in completed_jobs:
                    del self.active_job_ids[job_name]
                    
                if self.active_job_ids:
                    time.sleep(check_interval)
            
            print("\nAll jobs completed!")
            
        else:
            # Terminal display with ANSI codes
            # Keep existing terminal implementation
            print("\033[2K\r", end="")  # Clear current line
            print("Job Status Monitor:")
            print("-" * 60)
            print("Job ID     Name     State     Time     Nodes")
            print("-" * 60)
            
            prev_lines = 0
            
            while self.active_job_ids:
                print(f"\033[{prev_lines + 5}A", end="")
                
                statuses = self.get_job_statuses()
                
                for status in statuses:
                    print("\033[2K\r{:<10} {:<8} {:<9} {:<8} {:<8}".format(
                        status.job_id, status.name, status.state, status.time, status.nodes
                    ))
                
                prev_lines = len(statuses)
                
                completed_jobs = []
                for job_name, job_id in list(self.active_job_ids.items()):
                    if not any(s.job_id == job_id for s in statuses) or \
                       any(s.job_id == job_id and s.state in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                        completed_jobs.append(job_name)
                
                for job_name in completed_jobs:
                    del self.active_job_ids[job_name]
                    
                if self.active_job_ids:
                    time.sleep(check_interval)
            
            print("\033[2K\r\nAll jobs completed!")

def load_and_execute(args):
    print(args)
    with open(args[0], "rb") as executable_path:
        fn = dill.load(executable_path)
    with open(args[1], "rb") as data_path:
        data = dill.load(data_path)
    if isinstance(data, tuple):
        fn(data[0], data[1])
    else:
        fn(data)

if __name__ == "__main__":
    load_and_execute(sys.argv[1:])