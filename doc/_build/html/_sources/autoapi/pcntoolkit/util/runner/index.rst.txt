pcntoolkit.util.runner
======================

.. py:module:: pcntoolkit.util.runner


Classes
-------

.. autoapisummary::

   pcntoolkit.util.runner.Runner


Functions
---------

.. autoapisummary::

   pcntoolkit.util.runner.load_and_execute


Module Contents
---------------

.. py:class:: Runner(parallelize: bool = False, job_type: Literal['torque', 'slurm'] = 'slurm', n_batches: int | None = None, batch_size: int | None = None, n_cores: int = 1, time_limit: str | int = '00:05:00', memory: str = '5GB', max_retries: int = 3, random_sleep_scale: float = 0.1, environment: Optional[str] = None, cross_validate: bool = False, cv_folds: int = 5, preamble: str = 'module load anaconda3', log_dir: Optional[str] = None, temp_dir: Optional[str] = None)

   
   Initialize the runner.

   :param parallelize: Whether to parallelize the jobs.
   :type parallelize: :py:class:`bool`, *optional*
   :param job_type: The type of job to use.
   :type job_type: :py:class:`Literal[```"torque"``, ``"slurm"``:py:class:`]`, *optional*
   :param n_batches: The number of jobs to run in parallel.
   :type n_batches: :py:class:`int`, *optional*
   :param n_cores: The number of cores to use for each job.
   :type n_cores: :py:class:`int`, *optional*
   :param time_limit: The time limit for each job.
   :type time_limit: :py:class:`str | int`, *optional*
   :param memory: The memory to use for each job.
   :type memory: :py:class:`str`, *optional*
   :param max_retries: The maximum number of retries for each job.
   :type max_retries: :py:class:`int`, *optional*
   :param environment: The environment to use for each job.
   :type environment: :py:class:`str`, *optional*
   :param cross_validate: Whether to cross-validate the model.
   :type cross_validate: :py:class:`bool`, *optional*
   :param cv_folds: The number of folds to use for cross-validation.
   :type cv_folds: :py:class:`int`, *optional*
   :param preamble: The preamble to use for each job.
   :type preamble: :py:class:`str`, *optional*
   :param log_dir: The directory to save the logs to.
   :type log_dir: :py:class:`str`, *optional*
   :param temp_dir: The directory to save the temporary files to.
   :type temp_dir: :py:class:`str`, *optional*


   .. py:method:: check_job_status(job_name: str) -> tuple[bool, bool, Optional[str]]

      Check if a job has failed by looking for success file.

      :returns: (is_running, finished_with_error, error_message)
                If job is still running, returns (True, False, None)
                If job finished successfully, returns (False, False, None)
                If job failed, returns (False, True, error_message)
      :rtype: :py:class:`tuple[bool`, :py:class:`bool`, :py:class:`Optional[str]]`



   .. py:method:: check_jobs_status() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]

      Check all jobs in active_job_ids for errors.

      :returns: A tuple containing:
                - A dictionary mapping job names to job IDs for running jobs
                - A dictionary mapping job names to error messages for failed jobs
                - A dictionary mapping job names to job IDs for finished jobs
      :rtype: :py:class:`tuple[Dict[str`, :py:class:`str]`, :py:class:`Dict[str`, :py:class:`str]`, :py:class:`Dict[str`, :py:class:`str]]`



   .. py:method:: create_temp_and_log_dir()


   .. py:method:: extend(model: pcntoolkit.normative_model.NormativeModel, data: pcntoolkit.dataio.norm_data.NormData, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Extend a normative model on a dataset.

      :param model: The normative model to extend.
      :type model: :py:class:`NormativeModel`
      :param data: The data to extend the model on.
      :type data: :py:class:`NormData`
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
                      If false, the function will dispatch the jobs and return.
      :type observe: :py:class:`bool`, *optional*

      :returns: The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormativeModel | None`



   .. py:method:: extend_predict(model: pcntoolkit.normative_model.NormativeModel, fit_data: pcntoolkit.dataio.norm_data.NormData, predict_data: Optional[pcntoolkit.dataio.norm_data.NormData] = None, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Extend a normative model on a dataset and predict on another dataset.

      :param model: The normative model to extend.
      :type model: :py:class:`NormativeModel`
      :param fit_data: The data to extend the model on.
      :type fit_data: :py:class:`NormData`
      :param predict_data: The data to predict on. Can be None if cross-validation is used.
      :type predict_data: :py:class:`Optional[NormData]`, *optional*
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
                      If false, the function will dispatch the jobs and return.
      :type observe: :py:class:`bool`, *optional*

      :returns: The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormativeModel | None`



   .. py:method:: fit(model: pcntoolkit.normative_model.NormativeModel, data: pcntoolkit.dataio.norm_data.NormData, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Fit a normative model on a dataset.

      :param model: The normative model to fit.
      :type model: :py:class:`NormBase`
      :param data: The data to fit the model on.
      :type data: :py:class:`NormData`
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
                      If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.
      :type observe: :py:class:`bool`, *optional*

      :returns: The fitted model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormativeModel | None`



   .. py:method:: fit_predict(model: pcntoolkit.normative_model.NormativeModel, fit_data: pcntoolkit.dataio.norm_data.NormData, predict_data: Optional[pcntoolkit.dataio.norm_data.NormData] = None, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Fit a normative model on a dataset and predict on another dataset.

      :param model: The normative model to fit.
      :type model: :py:class:`NormativeModel`
      :param fit_data: The data to fit the model on.
      :type fit_data: :py:class:`NormData`
      :param predict_data: The data to predict on. Can be None if cross-validation is used.
      :type predict_data: :py:class:`Optional[NormData]`, *optional*
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish, then load the model into the model object
                      If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.
      :type observe: :py:class:`bool`, *optional*

      :returns: The fitted and model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormativeModel | None`



   .. py:method:: from_args(args: dict) -> Runner
      :classmethod:



   .. py:method:: get_all_job_file_paths(job_name)


   .. py:method:: get_data_path(job_name)


   .. py:method:: get_extend_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable


   .. py:method:: get_extend_predict_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable


   .. py:method:: get_fit_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable

      Returns a callable that fits a model on a chunk of data



   .. py:method:: get_fit_predict_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable

      Returns a callable that fits a model on a chunk of data and predicts on another chunk of data



   .. py:method:: get_predict_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable

      Loads each fold model and predicts on the corresponding fold of data. Model n is used to predict on fold n.



   .. py:method:: get_python_callable_path(job_name)


   .. py:method:: get_transfer_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable

      Returns a callable that transfers a model on a chunk of data



   .. py:method:: get_transfer_predict_chunk_fn(model: pcntoolkit.normative_model.NormativeModel, save_dir: str) -> Callable


   .. py:method:: load_from_state(runner_file: str) -> Runner
      :classmethod:


      Load a runner from a saved state.

      :param runner_file: Path to the runner state file
      :type runner_file: :py:class:`str`

      :returns: A runner instance with the saved state
      :rtype: :py:class:`Runner`



   .. py:method:: load_model(fold_index: Optional[int] = 0, into: pcntoolkit.normative_model.NormativeModel | None = None) -> pcntoolkit.normative_model.NormativeModel


   .. py:method:: predict(model: pcntoolkit.normative_model.NormativeModel, data: pcntoolkit.dataio.norm_data.NormData, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Predict on a dataset.

      :param model: The normative model to predict on.
      :type model: :py:class:`NormativeModel`
      :param data: The data to predict on.
      :type data: :py:class:`NormData`
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish.
      :type observe: :py:class:`bool`, *optional*

      :rtype: :py:class:`None. If you want` to :py:class:`load the model`, :py:class:`use the runner.load_model function.`



   .. py:method:: re_submit_failed_jobs(observe: bool = True) -> None


   .. py:method:: save() -> None

      Save the runner state to a JSON file in the save directory.



   .. py:method:: save_callable_and_data(job_name: int | str, fn: Callable, chunk: tuple[pcntoolkit.dataio.norm_data.NormData] | tuple[pcntoolkit.dataio.norm_data.NormData, pcntoolkit.dataio.norm_data.NormData | None]) -> tuple[str, str]


   .. py:method:: set_task_id(task_name: str, model: pcntoolkit.normative_model.NormativeModel, data: pcntoolkit.dataio.norm_data.NormData)


   .. py:method:: submit_jobs(fn: Callable, first_data_source: pcntoolkit.dataio.norm_data.NormData, second_data_source: Optional[pcntoolkit.dataio.norm_data.NormData] = None, mode: Literal['unary', 'binary'] = 'unary') -> None

      Submit jobs to the job scheduler.

      The predict_data argument is optional, and if it is not provided, None is passed to the function.

      :param fn: Function to call. It should take two arguments.
      :type fn: :py:class:`Callable`
      :param fit_data: Data to fit the model on
      :type fit_data: :py:class:`NormData`
      :param predict_data: Data to predict on, by default None
      :type predict_data: :py:class:`Optional[NormData]`, *optional*



   .. py:method:: transfer(model: pcntoolkit.normative_model.NormativeModel, data: pcntoolkit.dataio.norm_data.NormData, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Transfer a normative model to a new dataset.

      :param model: The normative model to transfer.
      :type model: :py:class:`NormativeModel`
      :param data: The data to transfer the model to.
      :type data: :py:class:`NormData`
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish and then return the transfered model.
                      If false, the function will dispatch the jobs and return.
      :type observe: :py:class:`bool`, *optional*

      :returns: The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormBase | None`



   .. py:method:: transfer_predict(model: pcntoolkit.normative_model.NormativeModel, fit_data: pcntoolkit.dataio.norm_data.NormData, predict_data: Optional[pcntoolkit.dataio.norm_data.NormData] = None, save_dir: Optional[str] = None, observe: bool = True) -> pcntoolkit.normative_model.NormativeModel | None

      Transfer a normative model to a new dataset and predict on another dataset.

      :param model: The normative model to transfer.
      :type model: :py:class:`NormativeModel`
      :param fit_data: The data to transfer the model to.
      :type fit_data: :py:class:`NormData`
      :param predict_data: The data to predict on. Can be None if cross-validation is used.
      :type predict_data: :py:class:`Optional[NormData]`, *optional*
      :param save_dir: The directory to save the model to. If None, the model will be saved in the model's save directory.
      :type save_dir: :py:class:`Optional[str]`, *optional*
      :param observe: Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
                      If false, the function will dispatch the jobs and return.
      :type observe: :py:class:`bool`, *optional*

      :returns: The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
      :rtype: :py:class:`NormBase | No   ne`



   .. py:method:: wait_or_finish(observe: bool, into: pcntoolkit.normative_model.NormativeModel | None = None) -> pcntoolkit.normative_model.NormativeModel | None


   .. py:method:: wrap_in_job(job_name, python_callable_path, data_path)


   .. py:method:: wrap_in_slurm_job(job_name: int | str, python_callable_path: str, data_path: str) -> list[str]


   .. py:method:: wrap_in_torque_job(job_name: int | str, python_callable_path: str, data_path: str) -> list[str]


   .. py:attribute:: active_jobs
      :type:  Dict[str, str]


   .. py:attribute:: batch_size
      :type:  int | None
      :value: 2



   .. py:attribute:: cross_validate
      :type:  bool
      :value: False



   .. py:attribute:: cv_folds
      :type:  int
      :value: 5



   .. py:attribute:: environment
      :type:  str
      :value: None



   .. py:attribute:: failed_jobs
      :type:  Dict[str, str]


   .. py:attribute:: job_commands
      :type:  Dict[str, list[str]]


   .. py:attribute:: job_observer
      :value: None



   .. py:attribute:: job_type
      :type:  str
      :value: 'local'



   .. py:attribute:: log_dir
      :type:  str
      :value: ''



   .. py:attribute:: max_retries
      :type:  int
      :value: 3



   .. py:attribute:: memory
      :type:  str
      :value: '5gb'



   .. py:attribute:: n_batches
      :type:  int | None
      :value: None



   .. py:attribute:: n_cores
      :type:  int
      :value: 1



   .. py:attribute:: parallelize
      :type:  bool
      :value: True



   .. py:attribute:: preamble
      :type:  str
      :value: 'module load anaconda3'



   .. py:attribute:: random_sleep_scale
      :type:  float
      :value: 0.1



   .. py:attribute:: save_dir
      :value: ''



   .. py:attribute:: task_id
      :value: ''



   .. py:attribute:: temp_dir
      :type:  str
      :value: ''



   .. py:attribute:: time_limit_seconds
      :type:  int
      :value: 300



   .. py:attribute:: time_limit_str
      :type:  str | int
      :value: '00:05:00'



   .. py:attribute:: unique_log_dir
      :value: ''



   .. py:attribute:: unique_temp_dir
      :value: ''



.. py:function:: load_and_execute(args)

   Load a callable and data from a pickle file and execute it.

   :param args: A list of arguments. The first argument is the path to the callable. The second argument is the path to the data. The third argument is the max number of retries.  The fourth argument is the scale of the random sleep.
   :type args: :py:class:`list[str]`


