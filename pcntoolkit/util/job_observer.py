import copy
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List

from pcntoolkit.util.output import Messages, Output, Warnings


@dataclass
class JobStatus:
    job_id: str
    name: str
    state: str
    time: str
    nodes: str
    success_file_exists: bool = False


class JobObserver:
    def __init__(self, active_job_ids: Dict[str, str], job_type: str = "local", log_dir: str = "logs"):
        self.active_job_ids = copy.deepcopy(active_job_ids)
        self.job_type = job_type
        self.log_dir = log_dir
        # Reverse mapping from job_id to job_name for looking up success files
        self.job_id_to_name = {v: k for k, v in active_job_ids.items()}

    def check_success_file(self, job_name: str) -> bool:
        """Check if a success file exists for the given job name."""
        success_file = os.path.join(self.log_dir, f"{job_name}.success")
        return os.path.exists(success_file)

    def get_job_statuses(self) -> List[JobStatus]:
        """Get status of all tracked jobs."""
        if not self.active_job_ids:
            return []

        # Get all job statuses at once
        if self.job_type == "local":
            process = subprocess.Popen(
                ["ps", "-e", "-o", "pid,comm,stat,etime"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        elif self.job_type == "slurm":
            process = subprocess.Popen(
                ["squeue", "--format=%i|%j|%T|%M|%N", "--noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            raise ValueError(f"Invalid job type: {self.job_type}")

        stdout, stderr = process.communicate()

        statuses = []
        if process.returncode is None or process.returncode == 0:
            # Split output into lines and filter out empty lines
            lines = [line.strip() for line in stdout.split("\n") if line.strip()]
            for line in lines:
                try:
                    if self.job_type == "local":
                        job_id = line.split(maxsplit=1)[0]
                        if job_id in list(self.active_job_ids.values()):
                            job_id, name, state, time = line.split()
                            state = self.map_local_to_slurm_state(state)
                            job_name = self.job_id_to_name.get(job_id)
                            success_exists = (job_name is not None) and self.check_success_file(job_name)
                            statuses.append(
                                JobStatus(
                                    job_id=job_id,
                                    name=name,
                                    state=state,
                                    time=time,
                                    nodes="",
                                    success_file_exists=success_exists,
                                )
                            )
                    else:
                        job_id, name, state, time, nodes = line.split("|")
                        if job_id in list(self.active_job_ids.values()):
                            job_name = self.job_id_to_name.get(job_id)
                            success_exists = (job_name is not None) and self.check_success_file(job_name)
                            statuses.append(
                                JobStatus(
                                    job_id=job_id,
                                    name=name,
                                    state=state,
                                    time=time,
                                    nodes=nodes,
                                    success_file_exists=success_exists,
                                )
                            )
                except ValueError as e:
                    Output.warning(Warnings.ERROR_PARSING_JOB_STATUS_LINE, line=line, error=e)
        elif stderr:
            Output.warning(Warnings.ERROR_GETTING_JOB_STATUSES, stderr=stderr)

        # For jobs not found in process list, check if they have success files
        active_job_ids_set = set(self.active_job_ids.values())
        found_job_ids = {status.job_id for status in statuses}
        for job_id in active_job_ids_set - found_job_ids:
            job_name = self.job_id_to_name.get(job_id)
            if job_name:
                success_exists = self.check_success_file(job_name)
                state = "COMPLETED" if success_exists else "FAILED"
                statuses.append(
                    JobStatus(
                        job_id=job_id,
                        name=job_name,
                        state=state,
                        time="",
                        nodes="",
                        success_file_exists=success_exists,
                    )
                )

        return statuses

    def map_local_to_slurm_state(self, local_state: str) -> str:
        base_state = local_state[0].upper()

        state_map = {
            "R": "RUNNING",  # Running or runnable
            "S": "RUNNING",  # Interruptible sleep
            "D": "RUNNING",  # Uninterruptible sleep
            "Z": "FAILED",  # Zombie
            "T": "SUSPENDED",  # Stopped
            "X": "FAILED",  # Dead
            "I": "PENDING",  # Idle kernel thread
        }
        # Assume that if the job is not in the state map, it is completed (sorry)
        return state_map.get(base_state, "COMPLETED")

    def wait_for_jobs(self, check_interval=5):
        """Wait for all submitted jobs to complete.

        Args:
            check_interval (int): Time in seconds between job status checks
        """
        # Detect if we're running in a Jupyter notebook
        in_notebook = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell not in ["ZMQInteractiveShell", "Shell"]:
                in_notebook = False
        except NameError:
            in_notebook = False

        if in_notebook:
            self.wait_for_jobs_notebook(check_interval)
        else:
            self.wait_for_jobs_terminal(check_interval)

    def wait_for_jobs_notebook(self, check_interval=5):
        # Notebook-friendly display without ANSI codes
        from IPython.display import clear_output

        while self.active_job_ids:
            clear_output(wait=True)
            Output.print(Messages.JOB_STATUS_MONITOR)
            # Get and display current statuses
            statuses = self.get_job_statuses()
            for status in statuses:
                Output.print(
                    Messages.JOB_STATUS_LINE,
                    status.job_id,
                    status.name,
                    status.state,
                    status.time,
                    status.nodes,
                )

            # Check for completed jobs
            completed_jobs = []
            for job_name, job_id in list(self.active_job_ids.items()):
                matching_statuses = [s for s in statuses if s.job_id == job_id]
                if not matching_statuses:
                    # Job not found in process list and no success file
                    if not self.check_success_file(job_name):
                        completed_jobs.append(job_name)
                elif any(
                    s.state in ["COMPLETED", "FAILED", "CANCELLED"] or (s.state == "COMPLETED" and not s.success_file_exists)
                    for s in matching_statuses
                ):
                    completed_jobs.append(job_name)

            # Remove completed jobs
            for job_name in completed_jobs:
                del self.active_job_ids[job_name]

            if self.active_job_ids:
                time.sleep(check_interval)

        Output.print(Messages.ALL_JOBS_COMPLETED)

    def wait_for_jobs_terminal(self, check_interval=5):
        # Terminal display with ANSI codes
        # Keep existing terminal implementation
        show_pid = Output.get_show_pid()
        Output.set_show_pid(False)
        Output.print(Messages.JOB_STATUS_MONITOR)

        prev_lines = 0

        while self.active_job_ids:
            statuses = self.get_job_statuses()

            for status in statuses:
                Output.print(
                    Messages.JOB_STATUS_LINE,
                    status.job_id,
                    status.name,
                    status.state,
                    status.time,
                    status.nodes,
                )

            prev_lines = len(statuses)

            completed_jobs = []
            for job_name, job_id in list(self.active_job_ids.items()):
                if not any(s.job_id == job_id for s in statuses) or any(
                    s.job_id == job_id and s.state in ["COMPLETED", "FAILED", "CANCELLED"] for s in statuses
                ):
                    completed_jobs.append(job_name)

            for job_name in completed_jobs:
                del self.active_job_ids[job_name]

            if self.active_job_ids:
                time.sleep(check_interval)
            print(f"\033[{prev_lines}A", end="")

        Output.set_show_pid(show_pid)
        Output.print(Messages.ALL_JOBS_COMPLETED)
