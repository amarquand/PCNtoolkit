pcntoolkit.normative_model
==========================

.. py:module:: pcntoolkit.normative_model

.. autoapi-nested-parse::

   Module providing the NormativeModel class, which is the main class for building and using normative models.



Classes
-------

.. autoapisummary::

   pcntoolkit.normative_model.NormativeModel


Module Contents
---------------

.. py:class:: NormativeModel(template_regression_model: pcntoolkit.regression_model.regression_model.RegressionModel, savemodel: bool = True, evaluate_model: bool = True, saveresults: bool = True, saveplots: bool = True, save_dir: Optional[str] = None, inscaler: str = 'none', outscaler: str = 'none', name: Optional[str] = None)

   This class provides the foundation for building normative models, handling multiple
   response variables through separate regression models. It manages data preprocessing,
   model fitting, prediction, and evaluation.

   :param template_reg_model: Regression model used as a template to create all regression models.
   :type template_reg_model: :py:class:`RegressionModel`
   :param savemodel: Whether to save the model.
   :type savemodel: :py:class:`bool`
   :param evaluate_model: Whether to evaluate the model.
   :type evaluate_model: :py:class:`bool`
   :param saveresults: Whether to save the results.
   :type saveresults: :py:class:`bool`
   :param saveplots: Whether to save the plots.
   :type saveplots: :py:class:`bool`
   :param save_dir: Directory to save the model, results, and plots.
   :type save_dir: :py:class:`str`
   :param inscaler: Input (X/covariates) scaler to use.
   :type inscaler: :py:class:`str`
   :param outscaler: Output (Y/response_vars) scaler to use.
   :type outscaler: :py:class:`str`
   :param name: Name of the model
   :type name: :py:class:`str`


   .. py:method:: __getitem__(key: str) -> pcntoolkit.regression_model.regression_model.RegressionModel


   .. py:method:: __setitem__(key: str, value: pcntoolkit.regression_model.regression_model.RegressionModel) -> None


   .. py:method:: check_compatibility(data: pcntoolkit.dataio.norm_data.NormData) -> bool

      Check if the data is compatible with the model.

      :param data: Data to check compatibility with.
      :type data: :py:class:`NormData`

      :returns: True if compatible, False otherwise
      :rtype: :py:class:`bool`



   .. py:method:: compute_centiles(data: pcntoolkit.dataio.norm_data.NormData, centiles: Optional[List[float] | numpy.ndarray] = None, **kwargs) -> pcntoolkit.dataio.norm_data.NormData

      Computes the centiles for each response variable in the data.

      :param data: Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).
      :type data: :py:class:`NormData`
      :param centiles: The centiles to compute. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
      :type centiles: :py:class:`np.ndarray`, *optional*

      :returns: Prediction results containing:
                - Centiles: centiles of the response variables
      :rtype: :py:class:`NormData`



   .. py:method:: compute_correlation_matrix(data, bandwidth=5, covariate='age')


   .. py:method:: compute_logp(data: pcntoolkit.dataio.norm_data.NormData) -> pcntoolkit.dataio.norm_data.NormData

      Computes the log-probability of the data under the model.

      :param data: Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).
      :type data: :py:class:`NormData`

      :returns: Prediction results containing:
                - Logp: log-probability of the response variables per datapoint
      :rtype: :py:class:`NormData`



   .. py:method:: compute_thrivelines(data: pcntoolkit.dataio.norm_data.NormData, span: int = 5, step: int = 1, z_thrive: float = 0.0, covariate='age', **kwargs) -> pcntoolkit.dataio.norm_data.NormData

      Computes the thrivelines for each responsevar in the data



   .. py:method:: compute_yhat(data: pcntoolkit.dataio.norm_data.NormData) -> pcntoolkit.dataio.norm_data.NormData

      Computes the predicted values for each response variable in the data.



   .. py:method:: compute_zscores(data: pcntoolkit.dataio.norm_data.NormData) -> pcntoolkit.dataio.norm_data.NormData

      Computes Z-scores for each response variable using fitted regression models.

      :param data: Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).
      :type data: :py:class:`NormData`

      :returns: Prediction results containing:
                - Zscores: z-scores of the response variables
      :rtype: :py:class:`NormData`



   .. py:method:: evaluate(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Evaluates the model performance on the data.
      This method performs the following steps:
      1. Preprocesses the data

      5. Evaluates the model performance
      6. Postprocesses the data



   .. py:method:: extend(data: pcntoolkit.dataio.norm_data.NormData, save_dir: str | None = None, n_synth_samples: int | None = None) -> NormativeModel

      Extends the model to a new dataset.



   .. py:method:: extend_predict(extend_data: pcntoolkit.dataio.norm_data.NormData, predict_data: pcntoolkit.dataio.norm_data.NormData, save_dir: str | None = None, n_synth_samples: int | None = None) -> NormativeModel

      Extends the model to a new dataset and predicts the data.



   .. py:method:: extract_data(data: pcntoolkit.dataio.norm_data.NormData) -> Tuple[xarray.DataArray, xarray.DataArray, dict[str, dict[str, int]], xarray.DataArray, xarray.DataArray]


   .. py:method:: fit(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Fits a regression model for each response variable in the data.

      :param data: Training data containing covariates (X), batch effects (batch_effects), and response variables (Y).
                   Must be a valid NormData object with properly formatted dimensions:
                   - X: (n_samples, n_covariates)
                   - batch_effects: (n_samples, n_batch_effects)
                   - Y: (n_samples, n_response_vars)
      :type data: :py:class:`NormData`



   .. py:method:: fit_predict(fit_data: pcntoolkit.dataio.norm_data.NormData, predict_data: pcntoolkit.dataio.norm_data.NormData) -> pcntoolkit.dataio.norm_data.NormData

      Combines model.fit and model.predict in a single operation.



   .. py:method:: from_args(**kwargs) -> NormativeModel
      :classmethod:


      Create a new normative model from command line arguments.

      :param args: A dictionary of command line arguments.
      :type args: :py:class:`dict[str`, :py:class:`str]`

      :returns: An instance of a normative model.
      :rtype: :py:class:`NormBase`

      :raises ValueError: If the regression model specified in the arguments is unknown.



   .. py:method:: harmonize(data: pcntoolkit.dataio.norm_data.NormData, reference_batch_effect: dict[str, str] | None = None) -> pcntoolkit.dataio.norm_data.NormData

      Harmonizes the data to a reference batch effect. Harmonizes to the provided reference batch effect if provided,
      otherwise, harmonizes to the first batch effect alphabetically.

      :param data: Data to harmonize.
      :type data: :py:class:`NormData`
      :param reference_batch_effect: Reference batch effect.
      :type reference_batch_effect: :py:class:`dict[str`, :py:class:`str]`



   .. py:method:: load(path: str, into: NormativeModel | None = None) -> NormativeModel
      :classmethod:


      Load a normative model from a path.

      :param path: The path to the normative model.
      :type path: :py:class:`str`
      :param into: The normative model to load the data into. If None, a new normative model is created.
                   This is useful if you want to load a normative model into an existing normative model, for example in the runner.
      :type into: :py:class:`NormBase`, *optional*



   .. py:method:: map_batch_effects(batch_effects: xarray.DataArray) -> xarray.DataArray


   .. py:method:: model_specific_evaluation() -> None

      Save model-specific evaluation metrics.



   .. py:method:: postprocess(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Apply postprocessing to the data.

      Args:
          data (NormData): Data to postprocess.



   .. py:method:: predict(data: pcntoolkit.dataio.norm_data.NormData) -> pcntoolkit.dataio.norm_data.NormData

      Computes Z-scores, centiles, logp, yhat for each observation using fitted regression models.



   .. py:method:: preprocess(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Applies preprocessing transformations to the input data.

      Args:
          data (NormData): Data to preprocess.



   .. py:method:: register_batch_effects(data: pcntoolkit.dataio.norm_data.NormData) -> None


   .. py:method:: register_data_info(data: pcntoolkit.dataio.norm_data.NormData) -> None


   .. py:method:: sample_batch_effects(n_samples: int) -> xarray.DataArray

      Sample the batch effects from the estimated distribution.



   .. py:method:: sample_covariates(bes: xarray.DataArray, covariate_range_per_batch_effect: bool = False) -> xarray.DataArray

      Sample the covariates from the estimated distribution.

      Uses ranges of observed covariates matched with batch effects to create a representative sample



   .. py:method:: save(path: Optional[str] = None) -> None

      Save the model to a file.

      Args:
          path (str, optional): The path to save the model to. If None, the model is saved to the save_dir provided in the norm_conf.



   .. py:method:: scale_backward(data: pcntoolkit.dataio.norm_data.NormData) -> None

      Scales data back to its original scale using stored scalers.

      :param data: Data object containing arrays to be scaled back:
                   - X : array-like, shape (n_samples, n_covariates)
                       Covariate data to be scaled back
                   - y : array-like, shape (n_samples, n_response_vars), optional
                       Response variable data to be scaled back
      :type data: :py:class:`NormData`



   .. py:method:: scale_forward(data: pcntoolkit.dataio.norm_data.NormData, overwrite: bool = False) -> None

      Scales input data to standardized form using configured scalers.

      :param data: Data object containing arrays to be scaled:
                   - X : array-like, shape (n_samples, n_covariates)
                       Covariate data to be scaled
                   - y : array-like, shape (n_samples, n_response_vars), optional
                       Response variable data to be scaled
      :type data: :py:class:`NormData`
      :param overwrite: If True, creates new scalers even if they already exist.
                        If False, uses existing scalers when available.
      :type overwrite: :py:class:`bool`, *default* :py:obj:`False`



   .. py:method:: set_ensure_save_dirs()

      Ensures that the save directories for results and plots are created when they are not there yet (otherwise resulted in an error)



   .. py:method:: set_save_dir(save_dir: str) -> None

      Override the save_dir in the norm_conf.

      Args:
          save_dir (str): New save directory.



   .. py:method:: synthesize(data: pcntoolkit.dataio.norm_data.NormData | None = None, n_samples: int | None = None, covariate_range_per_batch_effect=False) -> pcntoolkit.dataio.norm_data.NormData

      Synthesize data from the model

      :param data: A NormData object with X and batch_effects. If provided, used to generate the synthetic data.
                   If the data has no batch_effects, batch_effects are sampled from the model.
                   If the data has no X, X is sampled from the model, using the provided or sampled batch_effects.
                   If neither X nor batch_effects are provided, the model is used to generate the synthetic data.
      :type data: :py:class:`NormData`, *optional*
      :param n_samples: Number of samples to synthesize. If this is None, the number of samples that were in the train data is used.
      :type n_samples: :py:class:`int`, *optional*
      :param covariate_range_per_batch_effect: If True, the covariate range is different for each batch effect.
      :type covariate_range_per_batch_effect: :py:class:`bool`, *optional*



   .. py:method:: to_dict()


   .. py:method:: transfer(transfer_data: pcntoolkit.dataio.norm_data.NormData, save_dir: str | None = None, **kwargs) -> NormativeModel

      Transfers the model to a new dataset.



   .. py:method:: transfer_predict(transfer_data: pcntoolkit.dataio.norm_data.NormData, predict_data: pcntoolkit.dataio.norm_data.NormData, save_dir: str | None = None, **kwargs) -> NormativeModel

      Transfers the model to a new dataset and predicts the data.



   .. py:property:: batch_effect_dims
      :type: list[str]


      Returns the batch effect dimensions.
      Returns:
          list[str]: The batch effect dimensions.


   .. py:attribute:: evaluate_model
      :type:  bool
      :value: True



   .. py:attribute:: evaluator


   .. py:property:: has_batch_effect
      :type: bool


      Returns whether the model has a batch effect.
      Returns:
          bool: True if the model has a batch effect, False otherwise. This currently looks at the template reg conf


   .. py:attribute:: inscaler
      :type:  str
      :value: 'none'



   .. py:attribute:: inscalers
      :type:  dict


   .. py:attribute:: is_fitted
      :type:  bool
      :value: False



   .. py:property:: n_fit_observations
      :type: int


      Returns the number of batch effects.
      Returns:
          int: The number of batch effects.


   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



   .. py:attribute:: outscaler
      :type:  str
      :value: 'none'



   .. py:attribute:: outscalers
      :type:  dict


   .. py:attribute:: regression_models
      :type:  dict[str, pcntoolkit.regression_model.regression_model.RegressionModel]


   .. py:attribute:: response_vars
      :type:  list[str]
      :value: None



   .. py:property:: save_dir
      :type: str



   .. py:attribute:: savemodel
      :type:  bool
      :value: True



   .. py:attribute:: saveplots
      :type:  bool
      :value: True



   .. py:attribute:: saveresults
      :type:  bool
      :value: True



   .. py:attribute:: template_regression_model
      :type:  pcntoolkit.regression_model.regression_model.RegressionModel


