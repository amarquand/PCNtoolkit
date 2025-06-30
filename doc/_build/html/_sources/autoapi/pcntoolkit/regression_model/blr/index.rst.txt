pcntoolkit.regression_model.blr
===============================

.. py:module:: pcntoolkit.regression_model.blr

.. autoapi-nested-parse::

   Bayesian Linear Regression (BLR) implementation.

   This module implements Bayesian Linear Regression with support for:
   - L1/L2 regularization
   - Automatic Relevance Determination (ARD)
   - Heteroskedastic noise modeling
   - Multiple optimization methods (CG, Powell, Nelder-Mead, L-BFGS-B)

   The implementation follows standard Bayesian formulation with Gaussian priors
   and supports both homoskedastic and heteroskedastic noise models.



Classes
-------

.. autoapisummary::

   pcntoolkit.regression_model.blr.BLR


Functions
---------

.. autoapisummary::

   pcntoolkit.regression_model.blr.create_design_matrix


Module Contents
---------------

.. py:class:: BLR(name: str = 'template', fixed_effect: bool = False, heteroskedastic: bool = False, fixed_effect_var: bool = False, warp_name: pcntoolkit.math_functions.warp.Optional[str] = None, warp_reparam: bool = False, basis_function_mean: pcntoolkit.math_functions.basis_function.BasisFunction = None, basis_function_var: pcntoolkit.math_functions.basis_function.BasisFunction = None, n_iter: int = 300, tol: float = 1e-05, ard: bool = False, optimizer: str = 'l-bfgs-b', l_bfgs_b_l: float = 0.1, l_bfgs_b_epsilon: float = 0.1, l_bfgs_b_norm: str = 'l2', hyp0: pcntoolkit.math_functions.warp.np.ndarray | None = None, is_fitted: bool = False, is_from_dict: bool = False)

   Bases: :py:obj:`pcntoolkit.regression_model.regression_model.RegressionModel`


   Bayesian Linear Regression model implementation.

   This class implements Bayesian Linear Regression with various features including
   automatic relevance determination (ARD), heteroskedastic noise modeling, and
   multiple optimization methods.

   This class implements Bayesian Linear Regression with various features including
   automatic relevance determination (ARD), heteroskedastic noise modeling, and
   multiple optimization methods.

   :param name: Unique identifier for the model instance
   :type name: :py:class:`str`
   :param fixed_effect: Whether to model a fixed effect in the mean, by default False
   :type fixed_effect: :py:class:`bool`, *optional*
   :param heteroskedastic: Whether to use heteroskedastic noise modeling, by default False
   :type heteroskedastic: :py:class:`bool`, *optional*
   :param fixed_effect_var: Whether to model a fixed effect in the variance, by default False
   :type fixed_effect_var: :py:class:`bool`, *optional*
   :param warp_name: Name of the warp function to use, by default None
   :type warp_name: :py:class:`str`, *optional*
   :param warp_reparam: Whether to use a reparameterized warp function, by default False
   :type warp_reparam: :py:class:`bool`, *optional*
   :param basis_function_mean: Basis function for the mean, by default None
   :type basis_function_mean: :py:class:`BasisFunction`, *optional*
   :param basis_function_var: Basis function for the variance, by default None
   :type basis_function_var: :py:class:`BasisFunction`, *optional*
   :param n_iter: Number of iterations for the optimization, by default 300
   :type n_iter: :py:class:`int`, *optional*
   :param tol: Tolerance for the optimization, by default 1e-5
   :type tol: :py:class:`float`, *optional*
   :param ard: Whether to use automatic relevance determination, by default False
   :type ard: :py:class:`bool`, *optional*
   :param optimizer: Optimizer to use for the optimization, by default "l-bfgs-b"
   :type optimizer: :py:class:`str`, *optional*
   :param l_bfgs_b_l: L-BFGS-B parameter, by default 0.1
   :type l_bfgs_b_l: :py:class:`float`, *optional*
   :param l_bfgs_b_epsilon: L-BFGS-B parameter, by default 0.1
   :type l_bfgs_b_epsilon: :py:class:`float`, *optional*
   :param l_bfgs_b_norm: L-BFGS-B parameter, by default "l2"
   :type l_bfgs_b_norm: :py:class:`str`, *optional*
   :param hyp0: Initial hyperparameters, by default None
   :type hyp0: :py:class:`np.ndarray`, *optional*
   :param is_fitted: Whether the model has been fitted, by default False
   :type is_fitted: :py:class:`bool`, *optional*
   :param is_from_dict: Whether the model was created from a dictionary, by default False
   :type is_from_dict: :py:class:`bool`, *optional*


   .. py:method:: Phi_Phi_var(X: pcntoolkit.math_functions.warp.np.ndarray, be: pcntoolkit.math_functions.warp.np.ndarray, be_maps: dict[str, dict[str, int]]) -> tuple[pcntoolkit.math_functions.warp.np.ndarray, pcntoolkit.math_functions.warp.np.ndarray]


   .. py:method:: backward(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Z: xarray.DataArray) -> xarray.DataArray

      Map Z values to Y space using BLR.

      :param X: Covariate data
      :type X: :py:class:`xr.DataArray`
      :param be: Batch effect data
      :type be: :py:class:`xr.`
      :param be_maps: Batch effect maps
      :type be_maps: :py:class:`dict[str`, :py:class:`dict[str`, :py:class:`int]]`
      :param Z: Z-score data
      :type Z: :py:class:`xr.DataArray`

      :returns: Z-values mapped to Y space
      :rtype: :py:class:`xr.DataArray`



   .. py:method:: dloglik(hyp: pcntoolkit.math_functions.warp.np.ndarray, X: pcntoolkit.math_functions.warp.np.ndarray, y: pcntoolkit.math_functions.warp.np.ndarray, var_X: pcntoolkit.math_functions.warp.np.ndarray) -> pcntoolkit.math_functions.warp.np.ndarray

      Function to compute derivatives



   .. py:method:: elemwise_logp(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> xarray.DataArray

      Compute log-probabilities for each observation in the data.

      :param X: Covariate data
      :type X: :py:class:`xr.DataArray`
      :param be: Batch effect data
      :type be: :py:class:`xr.DataArray`
      :param be_maps: Batch effect maps
      :type be_maps: :py:class:`dict[str`, :py:class:`dict[str`, :py:class:`int]]`
      :param Y: Response variable data
      :type Y: :py:class:`xr.DataArray`

      :returns: Log-probabilities of the data
      :rtype: :py:class:`xr.DataArray`



   .. py:method:: fit(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> None

      Fit the Bayesian Linear Regression model to the data.

      :param X: Covariate data
      :type X: :py:class:`xr.DataArray`
      :param be: Batch effect data
      :type be: :py:class:`xr.DataArray`
      :param be_maps: Batch effect maps
      :type be_maps: :py:class:`dict[str`, :py:class:`dict[str`, :py:class:`int]]`
      :param Y: Response variable data
      :type Y: :py:class:`xr.DataArray`

      :rtype: :py:obj:`None`



   .. py:method:: forward(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> xarray.DataArray

      Map Y values to Z space using BLR.

      :param X: Covariate data
      :type X: :py:class:`xr.DataArray`
      :param be: Batch effect data
      :type be: :py:class:`xr.DataArray`
      :param be_maps: Batch effect maps
      :type be_maps: :py:class:`dict[str`, :py:class:`dict[str`, :py:class:`int]]`
      :param Y: Response variable data
      :type Y: :py:class:`xr.DataArray`

      :returns: Z-values mapped to Y space
      :rtype: :py:class:`xr.DataArray`



   .. py:method:: from_args(name: str, args: dict) -> BLR
      :classmethod:


      Creates a configuration from command line arguments



   .. py:method:: from_dict(my_dict: dict, path: str | None = None) -> BLR
      :classmethod:


      Creates a configuration from a dictionary.



   .. py:method:: get_warp(warp: str | None) -> pcntoolkit.math_functions.warp.Optional[pcntoolkit.math_functions.warp.WarpBase]


   .. py:method:: init_hyp() -> pcntoolkit.math_functions.warp.np.ndarray

      Initialize model hyperparameters.

      :param data: Training data containing features and targets
      :type data: :py:class:`BLRData`

      :returns: Initialized hyperparameter vector
      :rtype: :py:class:`np.ndarray`



   .. py:method:: initialize_warp() -> None


   .. py:method:: loglik(hyp: pcntoolkit.math_functions.warp.np.ndarray, X: pcntoolkit.math_functions.warp.np.ndarray, y: pcntoolkit.math_functions.warp.np.ndarray, var_X: pcntoolkit.math_functions.warp.Optional[pcntoolkit.math_functions.warp.np.ndarray] = None) -> float

      Compute the negative log likelihood.

      :param hyp: Hyperparameter vector.
      :type hyp: :py:class:`np.ndarray`
      :param X: Covariates.
      :type X: :py:class:`np.ndarray`
      :param y: Responses.
      :type y: :py:class:`np.ndarray`
      :param var_X: Variance of covariates.
      :type var_X: :py:class:`np.ndarray`

      :returns: Negative log likelihood.
      :rtype: :py:class:`float`



   .. py:method:: model_specific_evaluation(path: str) -> None

      Save model-specific evaluation metrics.



   .. py:method:: parse_hyps(hyp: pcntoolkit.math_functions.warp.np.ndarray, Phi: pcntoolkit.math_functions.warp.np.ndarray, Phi_var: pcntoolkit.math_functions.warp.Optional[pcntoolkit.math_functions.warp.np.ndarray] = None) -> tuple[pcntoolkit.math_functions.warp.np.ndarray, pcntoolkit.math_functions.warp.np.ndarray, pcntoolkit.math_functions.warp.np.ndarray]

      Parse hyperparameters into model parameters.

      :param hyp: Hyperparameter vector.
      :type hyp: :py:class:`np.ndarray`
      :param Phi: Covariates.
      :type Phi: :py:class:`np.ndarray`
      :param Phi_var: Variance of covariates.
      :type Phi_var: :py:class:`np.ndarray (Optional)`

      :returns: Parsed alpha, beta and gamma parameters.
      :rtype: :py:class:`tuple[np.ndarray`, :py:class:`np.ndarray`, :py:class:`np.ndarray]`



   .. py:method:: penalized_loglik(hyp: pcntoolkit.math_functions.warp.np.ndarray, X: pcntoolkit.math_functions.warp.np.ndarray, y: pcntoolkit.math_functions.warp.np.ndarray, var_X: pcntoolkit.math_functions.warp.Optional[pcntoolkit.math_functions.warp.np.ndarray] = None, regularizer_strength: float = 0.1, norm: Literal['L1', 'L2'] = 'L1') -> float

      Compute the penalized log likelihood with L1 or L2 regularization.

      :param hyp: Hyperparameter vector
      :type hyp: :py:class:`np.ndarray`
      :param X: Feature matrix
      :type X: :py:class:`np.ndarray`
      :param y: Target vector
      :type y: :py:class:`np.ndarray`
      :param var_X: Variance of features
      :type var_X: :py:class:`np.ndarray`
      :param regularizer_strength: Regularization strength, by default 0.1
      :type regularizer_strength: :py:class:`float`, *optional*
      :param norm: Type of regularization norm, by default "L1"
      :type norm: ``{"L1", "L2"}``, *optional*

      :returns: Penalized negative log likelihood value
      :rtype: :py:class:`float`

      :raises ValueError: If norm is not "L1" or "L2"



   .. py:method:: post(hyp: pcntoolkit.math_functions.warp.np.ndarray, X: pcntoolkit.math_functions.warp.np.ndarray, y: pcntoolkit.math_functions.warp.np.ndarray, var_X: pcntoolkit.math_functions.warp.Optional[pcntoolkit.math_functions.warp.np.ndarray] = None) -> None

      Compute the posterior distribution.

      :param hyp: Hyperparameter vector.
      :type hyp: :py:class:`np.ndarray`
      :param X: Covariates.
      :type X: :py:class:`np.ndarray`
      :param y: Responses.
      :type y: :py:class:`np.ndarray`
      :param var_X: Variance of covariates.
      :type var_X: :py:class:`np.ndarray`



   .. py:method:: to_dict(path: str | None = None) -> dict

      Convert model instance to dictionary representation.

      Used for saving models to disk.

      :param path: Path to save any associated files, by default None
      :type path: :py:class:`str | None`, *optional*

      :returns: Dictionary containing model parameters and configuration
      :rtype: :py:class:`dict`



   .. py:method:: transfer(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray, **kwargs) -> BLR

      Transfer the model to a new dataset.

      :param X:
      :type X: :py:class:`xr.DataArray containing covariates`
      :param be:
      :type be: :py:class:`xr.DataArray containing batch effects`
      :param be_maps:
      :type be_maps: :py:class:`dictionary` of :py:class:`dictionaries mapping batch effect` to :py:class:`indices`
      :param Y:
      :type Y: :py:class:`xr.DataArray containing covariates`

      :returns: New instance of the regression model, transfered to the new dataset
      :rtype: :py:class:`RegressionModel`



   .. py:method:: ys_s2(X: pcntoolkit.math_functions.warp.np.ndarray, be: pcntoolkit.math_functions.warp.np.ndarray, be_maps: dict[str, dict[str, int]]) -> tuple[pcntoolkit.math_functions.warp.np.ndarray, pcntoolkit.math_functions.warp.np.ndarray]


   .. py:attribute:: ard
      :value: False



   .. py:attribute:: basis_function_mean


   .. py:attribute:: basis_function_var


   .. py:attribute:: fixed_effect
      :value: False



   .. py:attribute:: fixed_effect_var
      :value: False



   .. py:attribute:: gamma
      :type:  pcntoolkit.math_functions.warp.np.ndarray
      :value: None



   .. py:property:: has_batch_effect
      :type: bool


      Check if model includes batch effects.

      :returns: True if model includes batch effects, False otherwise
      :rtype: :py:class:`bool`


   .. py:attribute:: heteroskedastic
      :value: False



   .. py:attribute:: hyp
      :type:  pcntoolkit.math_functions.warp.np.ndarray
      :value: None



   .. py:attribute:: hyp0
      :value: None



   .. py:attribute:: l_bfgs_b_epsilon
      :value: 0.1



   .. py:attribute:: l_bfgs_b_l
      :value: 0.1



   .. py:attribute:: l_bfgs_b_norm
      :value: 'l2'



   .. py:attribute:: models_variance
      :value: False



   .. py:attribute:: n_iter
      :value: 300



   .. py:attribute:: optimizer
      :value: 'l-bfgs-b'



   .. py:attribute:: tol
      :value: 1e-05



   .. py:attribute:: warp_name
      :value: None



   .. py:attribute:: warp_reparam
      :value: False



.. py:function:: create_design_matrix(X: pcntoolkit.math_functions.warp.np.ndarray, be: pcntoolkit.math_functions.warp.np.ndarray, be_maps: dict[str, dict[str, int]], linear: bool = False, intercept: bool = False, fixed_effect: bool = False) -> pcntoolkit.math_functions.warp.np.ndarray

   Create design matrix for the model.

   :param data: Input data containing features and batch effects.
   :type data: :py:class:`NormData`
   :param linear: Include linear terms in the design matrix.
   :type linear: :py:class:`bool`, *default* :py:obj:`False`
   :param intercept: Include intercept term in the design matrix.
   :type intercept: :py:class:`bool`, *default* :py:obj:`False`
   :param fixed_effect: Include fixed effect for batch effects.
   :type fixed_effect: :py:class:`bool`, *default* :py:obj:`False`

   :returns: Design matrix combining all requested components.
   :rtype: :py:class:`np.ndarray`

   :raises ValueError: If no components are selected for the design matrix.


