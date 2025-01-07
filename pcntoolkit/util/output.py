import os
import warnings


class Messages:
    FITTING_MODELS = "Fitting models on {n_models} response variables."
    FITTING_MODEL = "Fitting model for {model_name}."
    PREDICTING_MODELS = "Making predictions on {n_models} response variables."
    PREDICTING_MODEL = "Making predictions on {model_name}."
    FITTING_AND_PREDICTING_MODELS = "Fitting and predicting {n_models} response variables."
    FITTING_AND_PREDICTING_MODEL = "Fitting and predicting model for {model_name}."
    SAVING_MODEL = "Saving model to {save_dir}."
    SAVING_RESULTS = "Saving results to {save_dir}."
    TRANSFERRING_MODELS = "Transferring models on {n_models} response variables."
    TRANSFERRING_MODEL = "Transferring model for {model_name}."
    COMPUTING_CENTILES = "Computing centiles for {n_models} response variables."
    COMPUTING_CENTILES_MODEL = "Computing centiles for {model_name}."
    COMPUTING_ZSCORES = "Computing z-scores for {n_models} response variables."
    COMPUTING_ZSCORES_MODEL = "Computing z-scores for {model_name}."
    COMPUTING_LOGP = "Computing log-probabilities for {n_models} response variables."
    COMPUTING_LOGP_MODEL = "Computing log-probabilities for {model_name}."
    NORMATIVE_MODEL_CONFIGURATION_VALID = "Configuration of normative model is valid."
    LOADING_ROI_MASK = "Loading ROI mask ..."
    GENERATING_MASK_AUTOMATICALLY = "Generating mask automatically ..."
    EXTRACTING_CIFTI_SURFACE_DATA = "Extracting cifti surface data to {outstem} ..."
    EXTRACTING_CIFTI_VOLUME_DATA = "Extracting cifti volume data to {niiname} ..."
    BLR_RESTARTING_ESTIMATION_AT_HYP = "Restarting estimation at hyp = {hyp}, due to: {e}"
    BLR_HYPERPARAMETERS_HAVE_NOT_CHANGED = "Hyperparameters have not changed, exiting"
    REGRESSION_MODEL_CONFIGURATION_VALID = "Configuration of regression model is valid."
    JOB_STATUS_MONITOR = """
Job Status Monitor:
--------------------------------------------
Job ID     Name     State     Time     Nodes
--------------------------------------------
"""
    ALL_JOBS_COMPLETED = """
All jobs completed!
"""
    JOB_STATUS_LINE = "{:<10} {:<8} {:<9} {:<8} {:<8}"
    NO_PYTHON_PATH_SPECIFIED = "No python path specified. Using interpreter path of current process: {python_path}"
    NO_LOG_DIR_SPECIFIED = "No log directory specified. Using default log directory: {log_dir}"
    NO_TEMP_DIR_SPECIFIED = "No temporary directory specified. Using default temporary directory: {temp_dir}"


class Warnings:
    MISSING_COVARIATES = "The dataset {dataset_name} is missing the following covariates: {covariates}"
    EXTRA_COVARIATES = "The dataset {dataset_name} has too many covariates: {covariates}"
    EXTRA_RESPONSE_VARS = "The dataset {dataset_name} has too many response variables: {response_vars}"
    UNKNOWN_BATCH_EFFECTS = "The dataset {dataset_name} has unknown batch effects: {batch_effects}"
    BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED = "Estimation of posterior distribution failed due to: \n{error}"
    ERROR_GETTING_JOB_STATUSES = "Error getting job statuses: {stderr}"
    ERROR_PARSING_JOB_STATUS_LINE = "Error parsing job status line: {line}"
    PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION = "Predict data not used in k-fold cross-validation"


class Errors:
    SAMPLE_BATCH_EFFECTS = (
        "Cannot sample {n_samples} batch effects, because some batch effects have more levels than the number of samples."
    )
    NORMATIVE_MODEL_CONFIGURATION_PROBLEMS = (
        "The following problems have been detected in the normative model configuration:\n{problems}"
    )
    REGRESSION_MODEL_CONFIGURATION_PROBLEMS = (
        "The following problems have been detected in the regression model configuration:\n{problems}"
    )
    UNKNOWN_FILE_TYPE = "Unknown file type: {filename}"
    NO_FLOAT_DATA_TYPE = "Only float data types currently handled, not {data_type}"
    BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH = "Hyperparameter vector invalid length"
    BLR_MODEL_NOT_FITTED = "Model must be fitted before computing log probabilities"
    ERROR_PARSING_TIME_LIMIT = "Cannot parse {time_limit_str} as time limit"
    ERROR_CROSS_VALIDATION_FOLDS = "If cross-validation is enabled, cv_folds must be greater than 1"
    ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION = (
        "Predict data is required for fit_predict without cross-validation"
    )
    ERROR_SUBMITTING_JOB = "Error submitting job {job_id}: {stderr}"


class Output:
    _show_messages = True  # Default to showing output
    _show_warnings = True

    @classmethod
    def set_show_messages(cls, value: bool) -> None:
        cls._show_messages = value

    @classmethod
    def print(cls, message: str, *args, **kwargs) -> None:
        """Print message only if show_messages mode is enabled"""
        if cls._show_messages:
            print("Process: " + str(os.getpid()) + " - " + message.format(*args, **kwargs))

    @classmethod
    def warning(cls, message: str, *args, **kwargs) -> None:
        """Print warning message only if show_warnings mode is enabled"""
        if cls._show_warnings:
            warnings.warn("Process: " + str(os.getpid()) + " - " + message.format(*args, **kwargs))

    @classmethod
    def error(cls, message: str, *args, **kwargs) -> Exception:
        """Print error message"""
        raise ValueError("Process: " + str(os.getpid()) + " - " + message.format(*args, **kwargs))
