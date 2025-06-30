pcntoolkit.util.job_observer
============================

.. py:module:: pcntoolkit.util.job_observer


Classes
-------

.. autoapisummary::

   pcntoolkit.util.job_observer.JobObserver
   pcntoolkit.util.job_observer.JobStatus


Module Contents
---------------

.. py:class:: JobObserver(active_job_ids: Dict[str, str], job_type: str = 'local', log_dir: str = 'logs', task_id: str = '')

   .. py:method:: check_success_file(job_name: str) -> bool

      Check if a success file exists for the given job name.



   .. py:method:: get_job_statuses() -> List[JobStatus]

      Get status of all tracked jobs.



   .. py:method:: show_job_status_monitor(in_notebook, statuses)


   .. py:method:: wait_for_jobs(check_interval=1)

      Wait for all submitted jobs to complete.

      Args:
          check_interval (int): Time in seconds between job status checks



   .. py:attribute:: active_job_ids


   .. py:attribute:: all_job_ids


   .. py:attribute:: job_id_to_name


   .. py:attribute:: job_type
      :value: 'local'



   .. py:attribute:: log_dir
      :value: 'logs'



   .. py:attribute:: task_id
      :value: ''



.. py:class:: JobStatus

   .. py:attribute:: job_id
      :type:  str


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: nodes
      :type:  str


   .. py:attribute:: state
      :type:  str


   .. py:attribute:: success_file_exists
      :type:  bool
      :value: False



   .. py:attribute:: time
      :type:  str


