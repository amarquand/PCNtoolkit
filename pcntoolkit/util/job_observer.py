import copy
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


class JobObserver:
    def __init__(self, active_job_ids: Dict[str, str], job_type: str = "local"):
        self.active_job_ids = copy.deepcopy(active_job_ids)
        self.job_type = job_type

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
                            statuses.append(
                                JobStatus(
                                    job_id=job_id,
                                    name=name,
                                    state=state,
                                    time=time,
                                    nodes="",
                                )
                            )
                    else:
                        job_id, name, state, time, nodes = line.split("|")
                        if job_id in list(self.active_job_ids.values()):
                            statuses.append(
                                JobStatus(
                                    job_id=job_id,
                                    name=name,
                                    state=state,
                                    time=time,
                                    nodes=nodes,
                                )
                            )
                except ValueError as e:
                    Output.warning(Warnings.ERROR_PARSING_JOB_STATUS_LINE, line=line, error=e)
        elif stderr:
            Output.warning(Warnings.ERROR_GETTING_JOB_STATUSES, stderr=stderr)

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
                if not any(s.job_id == job_id for s in statuses) or any(
                    s.job_id == job_id and s.state in ["COMPLETED", "FAILED", "CANCELLED"] for s in statuses
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
