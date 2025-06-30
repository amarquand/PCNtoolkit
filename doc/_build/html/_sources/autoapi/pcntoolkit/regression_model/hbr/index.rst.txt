pcntoolkit.regression_model.hbr
===============================

.. py:module:: pcntoolkit.regression_model.hbr


Classes
-------

.. autoapisummary::

   pcntoolkit.regression_model.hbr.HBR


Module Contents
---------------

.. py:class:: HBR(name: str = 'template', likelihood: pcntoolkit.math_functions.likelihood.Likelihood = get_default_normal_likelihood(), draws: int = 1500, tune: int = 500, cores: int = 4, chains: int = 4, nuts_sampler: str = 'nutpie', init: str = 'jitter+adapt_diag', progressbar: bool = True, is_fitted: bool = False, is_from_dict: bool = False)

   Bases: :py:obj:`pcntoolkit.regression_model.regression_model.RegressionModel`


   Hierarchical Bayesian Regression model implementation.

   This class implements a Bayesian hierarchical regression model using PyMC for
   posterior sampling. It supports multiple likelihood functions and provides
   methods for model fitting, prediction, and analysis.

   This class implements a Bayesian hierarchical regression model using PyMC for
   posterior sampling.

   :param name: Unique identifier for the model instance
   :type name: :py:class:`str`
   :param likelihood: Likelihood function to use for the model
   :type likelihood: :py:class:`Likelihood`
   :param draws: Number of samples to draw from the posterior distribution per chain, by default 1000
   :type draws: :py:class:`int`, *optional*
   :param tune: Number of tuning samples to draw from the posterior distribution per chain, by default 500
   :type tune: :py:class:`int`, *optional*
   :param cores: Number of cores to use for parallel sampling, by default 4
   :type cores: :py:class:`int`, *optional*
   :param chains: Number of chains to use for parallel sampling, by default 4
   :type chains: :py:class:`int`, *optional*
   :param nuts_sampler: NUTS sampler to use for parallel sampling, by default "nutpie"
   :type nuts_sampler: :py:class:`str`, *optional*
   :param init: Initialization method for the model, by default "jitter+adapt_diag"
   :type init: :py:class:`str`, *optional*
   :param progressbar: Whether to display a progress bar during sampling, by default True
   :type progressbar: :py:class:`bool`, *optional*
   :param is_fitted: Whether the model has been fitted, by default False
   :type is_fitted: :py:class:`bool`, *optional*
   :param is_from_dict: Whether the model was created from a dictionary, by default False
   :type is_from_dict: :py:class:`bool`, *optional*


   .. py:method:: backward(X, be, be_maps, Z) -> xarray.DataArray

      Map Z values to Y space using MCMC samples

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



   .. py:method:: compute_yhat(data, n_samples, responsevar, X, be, be_maps)


   .. py:method:: elemwise_logp(X, be, be_maps, Y) -> xarray.DataArray

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



   .. py:method:: extract_and_reshape(post_pred, observations, var_name: str) -> xarray.DataArray


   .. py:method:: fit(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> None

      Fit the model to training data using MCMC sampling.

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

      Map Y values to Z space using MCMC samples

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



   .. py:method:: from_args(name: str, args: Dict[str, Any]) -> HBR
      :classmethod:


      Create model instance from command line arguments.

      :param name: Name for new model instance
      :type name: :py:class:`str`
      :param args: Dictionary of command line arguments
      :type args: :py:class:`Dict[str`, :py:class:`Any]`

      :returns: New model instance
      :rtype: :py:class:`HBR`



   .. py:method:: from_dict(my_dict: Dict[str, Any], path: Optional[str] = None) -> HBR
      :classmethod:


      Create model instance from serialized dictionary.

      :param dict: Dictionary containing serialized model
      :type dict: :py:class:`Dict[str`, :py:class:`Any]`
      :param path: Path to load inference data from, by default None
      :type path: :py:class:`Optional[str]`, *optional*

      :returns: New model instance
      :rtype: :py:class:`HBR`



   .. py:method:: generic_MCMC_apply(X, be, be_maps, Y, fn, kwargs)

      Apply a generic function to likelihood parameters



   .. py:method:: has_batch_effect() -> bool

      Check if model includes batch effects.

      :returns: True if model includes batch effects, False otherwise
      :rtype: :py:class:`bool`



   .. py:method:: load_idata(path: str) -> None

      Load inference data from NetCDF file.

      :param path: Path to load inference data from. Should end in '.nc'
      :type path: :py:class:`str`

      :rtype: :py:obj:`None`

      :raises RuntimeError: If model is fitted but inference data cannot be loaded from path



   .. py:method:: model_specific_evaluation(path: str) -> None

      Save model-specific evaluation metrics.



   .. py:method:: save_idata(path: str) -> None

      Save inference data to NetCDF file.

      :param path: Path to save inference data to. Should end in '.nc'
      :type path: :py:class:`str`

      :rtype: :py:obj:`None`

      :raises RuntimeError: If model is fitted but does not have inference data



   .. py:method:: to_dict(path: Optional[str] = None) -> Dict[str, Any]

      Serialize model to dictionary format.

      :param path: Path to save inference data, by default None
      :type path: :py:class:`Optional[str]`, *optional*

      :returns: Dictionary containing serialized model
      :rtype: :py:class:`Dict[str`, :py:class:`Any]`



   .. py:method:: transfer(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray, **kwargs) -> HBR

              Perform transfer learning using existing model as prior.

              Parameters
              ----------
              hbrconf : HBRConf
                  Configuration for new model
              transferdata : HBRData
                  Data for transfer learning
              freedom : float
                  Parameter controlling influence of prior model (0-1)
      x
              Returns
              -------
              HBR
                  New model instance with transferred knowledge



   .. py:attribute:: chains
      :value: 4



   .. py:attribute:: cores
      :value: 4



   .. py:attribute:: draws
      :value: 1500



   .. py:attribute:: idata
      :type:  arviz.InferenceData
      :value: None



   .. py:attribute:: init
      :value: 'jitter+adapt_diag'



   .. py:attribute:: likelihood


   .. py:attribute:: nuts_sampler
      :value: 'nutpie'



   .. py:attribute:: progressbar
      :value: True



   .. py:attribute:: pymc_model
      :type:  pymc.Model
      :value: None



   .. py:attribute:: tune
      :value: 500



