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
    ERROR_PARSING_JOB_STATUS_LINE = "Error parsing job status line: {line} - {error}"
    PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION = "Predict data not used in k-fold cross-validation"
    DIR_DOES_NOT_EXIST = "{dir_attr_str} ({dir_attr}) does not exist, creating it for you."
    BLR_Y_NOT_PROVIDED = "y is not provided, setting self.y to zeros"
    BLR_VAR_X_NOT_PROVIDED = "var_X is not provided, setting self.var_X to zeros"
    BLR_BATCH_EFFECTS_NOT_PROVIDED = "batch_effects is not provided, setting self.batch_effects to zeros"
    HBR_BATCH_EFFECTS_NOT_PROVIDED = BLR_BATCH_EFFECTS_NOT_PROVIDED


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
    ERROR_PREDICT_DATA_REQUIRED = "Predict data is required for fit_predict without cross-validation"
    ERROR_SUBMITTING_JOB = "Error submitting job {job_id}: {stderr}"
    BLR_X_NOT_PROVIDED = "X is not provided"
    ERROR_UNKNOWN_FUNCTION = "Unknown function {func}"
    ERROR_ARGUMENT_SPECIFIED_TWICE = "Argument {key} is specified twice."
    ERROR_UNKNOWN_FUNCTION_FOR_CLASS = "Unknown function {func} for class {class_name}"
    BLR_ERROR_NO_DESIGN_MATRIX_CREATED = "No design matrix created"
    ERROR_UNKNOWN_CLASS = "Unknown class {class_name}"
    ERROR_FILE_NOT_FOUND = "File not found: {path}"
    ERROR_MODEL_NOT_FITTED = "Model needs to be fitted before it can be transferred"
    ERROR_BLR_VAR_X_NOT_PROVIDED = "Variance of covariates (var_X) is required for models with variance."
    ERROR_BLR_PENALTY_NOT_RECOGNIZED = "Requested penalty ({penalty}) not recognized, choose between 'L1' or 'L2'."
    ERROR_BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH = "Hyperparameter vector invalid length"
    ERROR_BLR_WARPS_NOT_PROVIDED = "A list of warp functions is required"
    ERROR_HBRDATA_X_NOT_PROVIDED = "X must be provided"
    ERROR_UNKNOWN_LIKELIHOOD = "Unsupported likelihood ({likelihood})"
    ERROR_HBR_Y_NOT_PROVIDED = "y must be provided for z-score computation"
    ERROR_HBR_FITTED_BUT_NO_IDATA = "HBR model is fitted but does not have idata. This should not happen."
    ERROR_HBR_COULD_NOT_LOAD_IDATA = "Could not load idata from {path}"
    ERROR_UNKNOWN_MAPPING = "Unknown mapping ({mapping})"
    ERROR_UNKNOWN_DISTRIBUTION = "Unknown distribution ({dist_name})"
    ERROR_SOURCE_ARRAY_NOT_FOUND = "Source array {source_array_name} does not exist in the data."
    ERROR_BASIS_FUNCTION_NOT_FITTED = "Basis function is not fitted. Please fit the basis function first."
    ERROR_DATA_MUST_BE_1D = "Data must be a 1D array or a N-dimensional array with a single column"
    ERROR_BATCH_EFFECTS_NOT_LIST = "Items of the batch_effect dict be a list or a string, not {batch_effect_type}"
    ERROR_SCALER_TYPE_NOT_FOUND = "Dictionary must contain 'scaler_type' key"
    ERROR_UNKNOWN_SCALER_TYPE = "Undefined scaler type: {scaler_type}"
    ERROR_SCALER_NOT_FITTED = "Scaler must be fitted before {method}"
    ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION = (
        "Predict data is required for fit_predict without cross-validation"
    )


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
    def error(cls, message: str, *args, **kwargs) -> ValueError:
        """Print error message"""
        return ValueError("Process: " + str(os.getpid()) + " - " + message.format(*args, **kwargs))