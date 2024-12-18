import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class JobStatus:
    job_id: str
    name: str
    state: str
    time: str
    nodes: str


class JobObserver:
    def __init__(self, active_job_ids: Dict[str, str]):
        self.active_job_ids = active_job_ids

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
            shell = get_ipython().__class__.__name__ # type: ignore
            if shell not in ['ZMQInteractiveShell', 'Shell']:
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
        

    def wait_for_jobs_terminal(self, check_interval=5):
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
