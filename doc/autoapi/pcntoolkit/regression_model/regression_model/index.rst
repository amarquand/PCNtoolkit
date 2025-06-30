pcntoolkit.regression_model.regression_model
============================================

.. py:module:: pcntoolkit.regression_model.regression_model

.. autoapi-nested-parse::

   Abstract base class for regression models in the PCNToolkit.

   This module provides the base class for all regression models, defining the common
   interface and shared functionality that all regression implementations must follow.

   .. rubric:: Notes

   All regression model implementations should inherit from this base class and
   implement the abstract methods.



Classes
-------

.. autoapisummary::

   pcntoolkit.regression_model.regression_model.RegressionModel


Module Contents
---------------

.. py:class:: RegressionModel(name: str, is_fitted: bool = False, is_from_dict: bool = False)

   Bases: :py:obj:`abc.ABC`


   This class defines the interface for all regression models in the toolkit,
   providing common attributes and methods that must be implemented by concrete
   subclasses.

   :param name: Unique identifier for the regression model instance
   :type name: :py:class:`str`
   :param reg_conf: Configuration object containing regression model parameters
   :type reg_conf: :py:class:`RegConf`
   :param is_fitted: Flag indicating if the model has been fitted to data, by default False
   :type is_fitted: :py:class:`bool`, *optional*
   :param is_from_dict: Flag indicating if the model was instantiated from a dictionary, by default False
   :type is_from_dict: :py:class:`bool`, *optional*

   .. attribute:: is_fitted

      Indicates whether the model has been fitted to data

      :type: :py:class:`bool`


   .. py:method:: backward(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Z: xarray.DataArray) -> xarray.DataArray
      :abstractmethod:


      Compute points in feature space for given z-scores

      :param X:
      :type X: :py:class:`xr.DataArray containing covariates`
      :param be:
      :type be: :py:class:`xr.DataArray containing batch effects`
      :param be_maps:
      :type be_maps: :py:class:`dictionary` of :py:class:`dictionaries mapping batch effect` to :py:class:`indices`
      :param Y:
      :type Y: :py:class:`xr.DataArray containing covariates`

      :returns: Data with Y values derived from Z-scores
      :rtype: :py:class:`xr.DataArray`



   .. py:method:: compute_yhat(data, n_samples, responsevar, X, be, be_maps)


   .. py:method:: elemwise_logp(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> xarray.DataArray
      :abstractmethod:


      Compute the log-probability of the data under the model.




   .. py:method:: fit(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> None
      :abstractmethod:


      Fit the model to the data.

      :param X:
      :type X: :py:class:`xr.DataArray containing covariates`
      :param be:
      :type be: :py:class:`xr.DataArray containing batch effects`
      :param be_maps:
      :type be_maps: :py:class:`dictionary` of :py:class:`dictionaries mapping batch effect` to :py:class:`indices`
      :param Y:
      :type Y: :py:class:`xr.DataArray containing covariates`

      :rtype: :py:class:`Nothing`



   .. py:method:: forward(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> xarray.DataArray
      :abstractmethod:


      Compute Z-scores for provided Y values

      :param X:
      :type X: :py:class:`xr.DataArray containing covariates`
      :param be:
      :type be: :py:class:`xr.DataArray containing batch effects`
      :param be_maps:
      :type be_maps: :py:class:`dictionary` of :py:class:`dictionaries mapping batch effect` to :py:class:`indices`
      :param Y:
      :type Y: :py:class:`xr.DataArray containing covariates`

      :returns: Data with Z-scores derived from Y values
      :rtype: :py:class:`xr.DataArray`



   .. py:method:: from_args(name: str, args: dict) -> RegressionModel
      :classmethod:

      :abstractmethod:


      Create model instance from arguments dictionary.

      Used for instantiating models from the command line.

      :param name: Unique identifier for the model instance
      :type name: :py:class:`str`
      :param args: Dictionary of model parameters and configuration
      :type args: :py:class:`dict`

      :returns: New instance of the regression model
      :rtype: :py:class:`RegressionModel`

      :raises NotImplementedError: Must be implemented by concrete subclasses



   .. py:method:: from_dict(my_dict: dict, path: str) -> RegressionModel
      :classmethod:

      :abstractmethod:


      Create model instance from dictionary representation.

      Used for loading models from disk.

      :param dct: Dictionary containing model parameters and configuration
      :type dct: :py:class:`dict`
      :param path: Path to load any associated files
      :type path: :py:class:`str`

      :returns: New instance of the regression model
      :rtype: :py:class:`RegressionModel`

      :raises NotImplementedError: Must be implemented by concrete subclasses



   .. py:method:: model_specific_evaluation(path: str) -> None

      Save model-specific evaluation metrics.



   .. py:method:: to_dict(path: str | None = None) -> dict
      :abstractmethod:


      Convert model instance to dictionary representation.

      Used for saving models to disk.

      :param path: Path to save any associated files, by default None
      :type path: :py:class:`str | None`, *optional*

      :returns: Dictionary containing model parameters and configuration
      :rtype: :py:class:`dict`



   .. py:method:: transfer(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> RegressionModel
      :abstractmethod:


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



   .. py:property:: has_batch_effect
      :type: bool

      :abstractmethod:


      Check if model includes batch effects.

      :returns: True if model includes batch effects, False otherwise
      :rtype: :py:class:`bool`


   .. py:attribute:: is_fitted
      :type:  bool
      :value: False



   .. py:attribute:: is_from_dict
      :type:  bool
      :value: False



   .. py:property:: name
      :type: str


      Get the model's name.

      :returns: The unique identifier of the model
      :rtype: :py:class:`str`


   .. py:property:: regmodel_dict
      :type: dict



