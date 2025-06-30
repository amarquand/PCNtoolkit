pcntoolkit.util.output
======================

.. py:module:: pcntoolkit.util.output


Classes
-------

.. autoapisummary::

   pcntoolkit.util.output.Errors
   pcntoolkit.util.output.Messages
   pcntoolkit.util.output.Output
   pcntoolkit.util.output.Warnings


Module Contents
---------------

.. py:class:: Errors

   .. py:attribute:: BLR_ERROR_NO_DESIGN_MATRIX_CREATED
      :value: 'No design matrix created'



   .. py:attribute:: BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH
      :value: 'Hyperparameter vector invalid length'



   .. py:attribute:: BLR_MODEL_NOT_FITTED
      :value: 'Model must be fitted before computing log probabilities'



   .. py:attribute:: BLR_X_NOT_PROVIDED
      :value: 'X is not provided'



   .. py:attribute:: ENSURE_POSITIVE_DISTRIBUTION
      :value: 'Distribution for {name} needs to be positive.'



   .. py:attribute:: ERROR_ARGUMENT_SPECIFIED_TWICE
      :value: 'Argument {key} is specified twice.'



   .. py:attribute:: ERROR_BASIS_FUNCTION_NOT_FITTED
      :value: 'Basis function is not fitted. Please fit the basis function first.'



   .. py:attribute:: ERROR_BATCH_EFFECTS_NOT_LIST
      :value: 'Items of the batch_effect dict be a list or a string, not {batch_effect_type}'



   .. py:attribute:: ERROR_BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH
      :value: 'Hyperparameter vector invalid length'



   .. py:attribute:: ERROR_BLR_PENALTY_NOT_RECOGNIZED
      :value: "Requested penalty ({penalty}) not recognized, choose between 'L1' or 'L2'."



   .. py:attribute:: ERROR_BLR_TRANSFER_NOT_IMPLEMENTED
      :value: 'BLR transfer not implemented'



   .. py:attribute:: ERROR_BLR_VAR_X_NOT_PROVIDED
      :value: 'Variance of covariates (var_X) is required for models with variance.'



   .. py:attribute:: ERROR_BLR_WARPS_NOT_PROVIDED
      :value: 'A list of warp functions is required'



   .. py:attribute:: ERROR_CDF_SHAPE
      :value: 'CDF shape {cdf_shape} does not match data shape {data_shape}'



   .. py:attribute:: ERROR_CROSS_VALIDATION_FOLDS
      :value: 'If cross-validation is enabled, cv_folds must be greater than 1'



   .. py:attribute:: ERROR_DATA_MUST_BE_1D
      :value: 'Data must be a 1D array or a N-dimensional array with a single column'



   .. py:attribute:: ERROR_ENVIRONMENT_NOT_FOUND
      :value: 'Environment {environment} not found. Please specify the path to the python environment using...



   .. py:attribute:: ERROR_FILE_NOT_FOUND
      :value: 'File not found: {path}'



   .. py:attribute:: ERROR_HBRDATA_X_NOT_PROVIDED
      :value: 'X must be provided'



   .. py:attribute:: ERROR_HBR_COULD_NOT_LOAD_IDATA
      :value: 'Could not load idata from {path}'



   .. py:attribute:: ERROR_HBR_FITTED_BUT_NO_IDATA
      :value: 'HBR model is fitted but does not have idata. This should not happen.'



   .. py:attribute:: ERROR_HBR_Y_NOT_PROVIDED
      :value: 'y must be provided for z-score computation'



   .. py:attribute:: ERROR_MODEL_NOT_FITTED
      :value: 'Model needs to be fitted before it can be transferred'



   .. py:attribute:: ERROR_MULTIPLE_COVARIATE_DIMS
      :value: 'Multiple covariate dimensions found: {covariate_dims}'



   .. py:attribute:: ERROR_NO_ENVIRONMENT_SPECIFIED
      :value: 'No python environment specified. Please specify the path to the python environment using the...



   .. py:attribute:: ERROR_PARSING_TIME_LIMIT
      :value: 'Cannot parse {time_limit_str} as time limit'



   .. py:attribute:: ERROR_PREDICT_DATA_NOT_SUPPORTED_FOR_CROSS_VALIDATION
      :value: 'Predict with cross-validation is not supported. Please use fit_predict instead.'



   .. py:attribute:: ERROR_PREDICT_DATA_REQUIRED
      :value: 'Predict data is required for fit_predict without cross-validation'



   .. py:attribute:: ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION
      :value: 'Predict data is required for fit_predict without cross-validation'



   .. py:attribute:: ERROR_SCALER_NOT_FITTED
      :value: 'Scaler must be fitted before {method}'



   .. py:attribute:: ERROR_SCALER_TYPE_NOT_FOUND
      :value: "Dictionary must contain 'scaler_type' key"



   .. py:attribute:: ERROR_SOURCE_ARRAY_NOT_FOUND
      :value: 'Source array {source_array_name} does not exist in the data.'



   .. py:attribute:: ERROR_SUBMITTING_JOB
      :value: 'Error submitting job {job_id}: {stderr}'



   .. py:attribute:: ERROR_UNKNOWN_CLASS
      :value: 'Unknown class {class_name}'



   .. py:attribute:: ERROR_UNKNOWN_DISTRIBUTION
      :value: 'Unknown distribution ({dist_name})'



   .. py:attribute:: ERROR_UNKNOWN_FUNCTION
      :value: 'Unknown function {func}'



   .. py:attribute:: ERROR_UNKNOWN_FUNCTION_FOR_CLASS
      :value: 'Unknown function {func} for class {class_name}'



   .. py:attribute:: ERROR_UNKNOWN_LIKELIHOOD
      :value: 'Unsupported likelihood ({likelihood})'



   .. py:attribute:: ERROR_UNKNOWN_MAPPING
      :value: 'Unknown mapping ({mapping})'



   .. py:attribute:: ERROR_UNKNOWN_SCALER_TYPE
      :value: 'Undefined scaler type: {scaler_type}'



   .. py:attribute:: ERROR_WARP_STRING_INVALID
      :value: 'Invalid warp string: {warp_string}'



   .. py:attribute:: ERROR_Y_NOT_FOUND
      :value: 'y not found in data'



   .. py:attribute:: ERROR_ZSCORES_INVERSE
      :value: 'Z-scores inverse has invalid shape: Z_shape={Z_shape}, X_shape={X_shape}'



   .. py:attribute:: HBR_MODEL_NOT_FITTED
      :value: 'HBR model is not fitted'



   .. py:attribute:: INVALID_ENVIRONMENT
      :value: 'The python environment {environment} is invalid because it has no /bin/python file. Please...



   .. py:attribute:: NORMATIVE_MODEL_CONFIGURATION_PROBLEMS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """The following problems have been detected in the normative model configuration:
         {problems}"""

      .. raw:: html

         </details>




   .. py:attribute:: NO_FLOAT_DATA_TYPE
      :value: 'Only float data types currently handled, not {data_type}'



   .. py:attribute:: OFFSETS_NOT_1D
      :value: 'Offsets must be a 1-d array or list'



   .. py:attribute:: OFFSET_NOT_VALID
      :value: 'Invalid list of offsets provided'



   .. py:attribute:: REGRESSION_MODEL_CONFIGURATION_PROBLEMS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """The following problems have been detected in the regression model configuration:
         {problems}"""

      .. raw:: html

         </details>




   .. py:attribute:: SAMPLE_BATCH_EFFECTS
      :value: 'Cannot sample {n_samples} batch effects, because some batch effects have more levels than the...



   .. py:attribute:: UNKNOWN_FILE_TYPE
      :value: 'Unknown file type: {filename}'



   .. py:attribute:: WB_COMMAND_FAILED
      :value: 'wb_command failed with error: {error}'



   .. py:attribute:: WB_COMMAND_NOT_FOUND
      :value: 'wb_command not found in PATH'



.. py:class:: Messages

   .. py:attribute:: BLR_HYPERPARAMETERS_HAVE_NOT_CHANGED
      :value: 'Hyperparameters have not changed, exiting'



   .. py:attribute:: BLR_RESTARTING_ESTIMATION_AT_HYP
      :value: 'Restarting estimation at hyp = {hyp}, due to: {e}'



   .. py:attribute:: COMPUTING_CENTILES
      :value: 'Computing centiles for {n_models} response variables.'



   .. py:attribute:: COMPUTING_CENTILES_MODEL
      :value: 'Computing centiles for {model_name}.'



   .. py:attribute:: COMPUTING_LOGP
      :value: 'Computing log-probabilities for {n_models} response variables.'



   .. py:attribute:: COMPUTING_LOGP_MODEL
      :value: 'Computing log-probabilities for {model_name}.'



   .. py:attribute:: COMPUTING_YHAT
      :value: 'Computing yhat for {n_models} response variables.'



   .. py:attribute:: COMPUTING_YHAT_MODEL
      :value: 'Computing yhat for {model_name}.'



   .. py:attribute:: COMPUTING_ZSCORES
      :value: 'Computing z-scores for {n_models} response variables.'



   .. py:attribute:: COMPUTING_ZSCORES_MODEL
      :value: 'Computing z-scores for {model_name}.'



   .. py:attribute:: DATASET_CREATED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Dataset "{name}" created.
             - {n_observations} observations
             - {n_subjects} unique subjects
             - {n_covariates} covariates
             - {n_response_vars} response variables
             - {n_batch_effects} batch effects:
             {batch_effects}
             """

      .. raw:: html

         </details>




   .. py:attribute:: EXECUTING_CALLABLE
      :value: 'Executing callable, attempt {attempt} of {total}.'



   .. py:attribute:: EXECUTION_FAILED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Execution of callable failed, attempt {attempt} of {total} with error: 
         {error}"""

      .. raw:: html

         </details>




   .. py:attribute:: EXECUTION_SUCCESSFUL
      :value: 'Execution of callable successful, attempt {attempt} of {total}.'



   .. py:attribute:: EXTRACTING_CIFTI_SURFACE_DATA
      :value: 'Extracting cifti surface data to {outstem} ...'



   .. py:attribute:: EXTRACTING_CIFTI_VOLUME_DATA
      :value: 'Extracting cifti volume data to {niiname} ...'



   .. py:attribute:: FITTING_AND_PREDICTING_MODEL
      :value: 'Fitting and predicting model for {model_name}.'



   .. py:attribute:: FITTING_AND_PREDICTING_MODELS
      :value: 'Fitting and predicting {n_models} response variables.'



   .. py:attribute:: FITTING_MODEL
      :value: 'Fitting model for {model_name}.'



   .. py:attribute:: FITTING_MODELS
      :value: 'Fitting models on {n_models} response variables.'



   .. py:attribute:: GENERATING_MASK_AUTOMATICALLY
      :value: 'Generating mask automatically ...'



   .. py:attribute:: HARMONIZING_DATA
      :value: 'Harmonizing data on {n_models} response variables.'



   .. py:attribute:: HARMONIZING_DATA_MODEL
      :value: 'Harmonizing data for {model_name}.'



   .. py:attribute:: JOB_STATUS_LINE
      :value: '{:<11} {:<9} {:<10} {:<9} {:<14}'



   .. py:attribute:: JOB_STATUS_MONITOR
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
         ---------------------------------------------------------
                       PCNtoolkit Job Status Monitor ®
         ---------------------------------------------------------
         Task ID: {task_id}
         ---------------------------------------------------------
         Job ID      Name          State      Time      Nodes
         ---------------------------------------------------------
         """

      .. raw:: html

         </details>




   .. py:attribute:: JOB_STATUS_SUMMARY
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
         ---------------------------------------------------------
         Total active jobs: {total_active_jobs}
         Total completed jobs: {total_completed_jobs}
         Total failed jobs: {total_failed_jobs}
         ---------------------------------------------------------
         """

      .. raw:: html

         </details>




   .. py:attribute:: LOADING_CALLABLE
      :value: 'Loading callable from {path}.'



   .. py:attribute:: LOADING_DATA
      :value: 'Loading data from {path}.'



   .. py:attribute:: LOADING_ROI_MASK
      :value: 'Loading ROI mask ...'



   .. py:attribute:: LOADING_RUNNER_STATE
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Loading runner state from:
         	{runner_file}"""

      .. raw:: html

         </details>




   .. py:attribute:: LOG_DIR_CREATED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Log directory created:
         	{log_dir}"""

      .. raw:: html

         </details>




   .. py:attribute:: NORMATIVE_MODEL_CONFIGURATION_VALID
      :value: 'Configuration of normative model is valid.'



   .. py:attribute:: NO_LOG_DIR_SPECIFIED
      :value: 'No log directory specified. Using default log directory: {log_dir}'



   .. py:attribute:: NO_MORE_RUNNING_JOBS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
         ---------------------------------------------------------
         No more running jobs!
         ---------------------------------------------------------
         """

      .. raw:: html

         </details>




   .. py:attribute:: NO_PYTHON_PATH_SPECIFIED
      :value: 'No python path specified. Using interpreter path of current process: {python_path}'



   .. py:attribute:: NO_TEMP_DIR_SPECIFIED
      :value: 'No temporary directory specified. Using default temporary directory: {temp_dir}'



   .. py:attribute:: PREDICTING_MODEL
      :value: 'Making predictions on {model_name}.'



   .. py:attribute:: PREDICTING_MODELS
      :value: 'Making predictions on {n_models} response variables.'



   .. py:attribute:: REGRESSION_MODEL_CONFIGURATION_VALID
      :value: 'Configuration of regression model is valid.'



   .. py:attribute:: RUNNER_LOADED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Runner loaded
         ---------------------------------------------------------Active jobs: {n_active_jobs}
         Finished jobs: {n_finished_jobs}
         Failed jobs: {n_failed_jobs}
         ---------------------------------------------------------"""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_CENTILES
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving centiles to:
         	{save_dir}."""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_MODEL
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving model to:
         	{save_dir}."""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_RESULTS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving results to:
         	{save_dir}."""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_RUNNER_STATE
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving runner state to:
         	{runner_file}"""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_STATISTICS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving statistics to:
         	{save_dir}."""

      .. raw:: html

         </details>




   .. py:attribute:: SAVING_ZSCORES
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Saving z-scores to:
         	{save_dir}."""

      .. raw:: html

         </details>




   .. py:attribute:: SYNTHESIZING_DATA
      :value: 'Synthesizing data for {n_models} response variables.'



   .. py:attribute:: SYNTHESIZING_DATA_MODEL
      :value: 'Synthesizing data for {model_name}.'



   .. py:attribute:: TASK_ID_CREATED
      :value: 'Task ID created: {task_id}'



   .. py:attribute:: TEMP_DIR_CREATED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Temporary directory created:
         	{temp_dir}"""

      .. raw:: html

         </details>




   .. py:attribute:: TRANSFERRING_MODEL
      :value: 'Transferring model for {model_name}.'



   .. py:attribute:: TRANSFERRING_MODELS
      :value: 'Transferring models on {n_models} response variables.'



.. py:class:: Output

   .. py:method:: error(message: str, *args, **kwargs) -> str
      :classmethod:


      Print error message



   .. py:method:: get_show_messages() -> bool
      :classmethod:



   .. py:method:: get_show_pid() -> bool
      :classmethod:



   .. py:method:: get_show_timestamp() -> bool
      :classmethod:



   .. py:method:: get_show_warnings() -> bool
      :classmethod:



   .. py:method:: print(message: str, *args, **kwargs) -> None
      :classmethod:


      Print message only if show_messages mode is enabled



   .. py:method:: set_show_messages(value: bool) -> None
      :classmethod:



   .. py:method:: set_show_pid(value: bool) -> None
      :classmethod:



   .. py:method:: set_show_timestamp(value: bool) -> None
      :classmethod:



   .. py:method:: set_show_warnings(value: bool) -> None
      :classmethod:



   .. py:method:: warning(message: str, *args, **kwargs) -> None
      :classmethod:


      Print warning message only if show_warnings mode is enabled



.. py:class:: Warnings

   .. py:attribute:: BLR_BATCH_EFFECTS_NOT_PROVIDED
      :value: 'batch_effects is not provided, setting self.batch_effects to zeros'



   .. py:attribute:: BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """Estimation of posterior distribution failed due to: 
         {error}"""

      .. raw:: html

         </details>




   .. py:attribute:: BLR_VAR_X_NOT_PROVIDED
      :value: 'var_X is not provided, setting self.var_X to zeros'



   .. py:attribute:: BLR_Y_NOT_PROVIDED
      :value: 'y is not provided, setting self.y to zeros'



   .. py:attribute:: CENTILES_ALREADY_COMPUTED_FOR_CENTILES
      :value: 'Centiles are already computed for {dataset_name} for centiles {centiles}, skipping computation....



   .. py:attribute:: DATA_ALREADY_SCALED
      :value: 'Data is already scaled, skipping scaling back to original scale.'



   .. py:attribute:: DATA_NOT_SCALED
      :value: 'Data is not scaled, skipping scaling back to original scale.'



   .. py:attribute:: DIR_DOES_NOT_EXIST
      :value: '{dir_attr_str} ({dir_attr}) does not exist, creating it for you.'



   .. py:attribute:: ERROR_GETTING_JOB_STATUSES
      :value: 'Error getting job statuses: {stderr}'



   .. py:attribute:: ERROR_PARSING_JOB_STATUS_LINE
      :value: 'Error parsing job status line: {line} - {error}'



   .. py:attribute:: ERROR_SUBMITTING_JOB
      :value: 'Error submitting job {job_id}: {stderr}'



   .. py:attribute:: EXTRA_COVARIATES
      :value: 'The dataset {dataset_name} has too many covariates: {covariates}'



   .. py:attribute:: EXTRA_RESPONSE_VARS
      :value: 'The dataset {dataset_name} has too many response variables: {response_vars}'



   .. py:attribute:: HBR_BATCH_EFFECTS_NOT_PROVIDED
      :value: 'batch_effects is not provided, setting self.batch_effects to zeros'



   .. py:attribute:: LOAD_CIFTI_GENERIC_EXCEPTION
      :value: 'A general exception occurred while loading CIFTI file: {}'



   .. py:attribute:: LOAD_NIFTI_GENERIC_EXCEPTION
      :value: 'A general exception occurred while loading NIFTI file: {}'



   .. py:attribute:: MISSING_COVARIATES
      :value: 'The dataset {dataset_name} is missing the following covariates: {covariates}'



   .. py:attribute:: MULTIPLE_BATCH_EFFECT_SUMMARY
      :value: "Multiple batch effect dimensions found. The summary printout currently uses the first dimension...



   .. py:attribute:: MULTIPLE_JOBS_FOUND_FOR_JOB_ID
      :value: 'Multiple jobs found for job ID {job_id}: {job_name}. Please check the job statuses and try again.'



   .. py:attribute:: NO_COVARIATES
      :value: 'No covariates provided for dataset {dataset_name}.'



   .. py:attribute:: NO_RESPONSE_VARS
      :value: 'No response variables provided for dataset {dataset_name}. Please provide a list of response...



   .. py:attribute:: PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
      :value: 'Predict data not used in k-fold cross-validation'



   .. py:attribute:: REMOVE_NAN_SET_TO_FALSE
      :value: 'Warning: remove_NAN is set to False. Missing (NaN) values may cause errors during model...



   .. py:attribute:: RESPONSE_VAR_NOT_FOUND
      :value: 'Response variable {response_var} not found in dataset {dataset_name}. Setting to NaN.'



   .. py:attribute:: SUBJECT_ID_MULTIPLE_COLUMNS
      :value: 'Subject ID file contains multiple columns. Using the first column for subject IDs.'



   .. py:attribute:: SUBJECT_ID_UNEXPECTED_SHAPE
      :value: 'Subject ID data has an unexpected shape. Expected 1D array or 2D array with one column. Using...



   .. py:attribute:: SYNTHESIZE_N_SAMPLES_IGNORED
      :value: 'n_samples is ignored because data is provided.'



   .. py:attribute:: THRIVELINES_ALREADY_COMPUTED_FOR
      :value: 'Thrivelines are already computed for {dataset_name} for offsets {offsets}, skipping...



   .. py:attribute:: UNKNOWN_BATCH_EFFECTS
      :value: 'The dataset {dataset_name} has unknown batch effects: {batch_effects}'



