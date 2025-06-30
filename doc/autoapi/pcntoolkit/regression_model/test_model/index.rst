pcntoolkit.regression_model.test_model
======================================

.. py:module:: pcntoolkit.regression_model.test_model


Classes
-------

.. autoapisummary::

   pcntoolkit.regression_model.test_model.TestModel


Module Contents
---------------

.. py:class:: TestModel(name: str, success_ratio: float = 1.0)

   Bases: :py:obj:`pcntoolkit.regression_model.regression_model.RegressionModel`


   Test model for regression model testing.

   Initialize the test model.

   Args:
       name: The name of the model.
       success_ratio: The ratio of successful fits.


   .. py:method:: backward(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Z: xarray.DataArray) -> xarray.DataArray

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



   .. py:method:: elemwise_logp(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> xarray.DataArray

      Compute the log-probability of the data under the model.




   .. py:method:: fit(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> None

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



   .. py:method:: from_args(name: str, args: dict) -> pcntoolkit.regression_model.regression_model.RegressionModel
      :classmethod:


      Create model instance from arguments dictionary.

      Used for instantiating models from the command line.

      :param name: Unique identifier for the model instance
      :type name: :py:class:`str`
      :param args: Dictionary of model parameters and configuration
      :type args: :py:class:`dict`

      :returns: New instance of the regression model
      :rtype: :py:class:`RegressionModel`

      :raises NotImplementedError: Must be implemented by concrete subclasses



   .. py:method:: from_dict(my_dict: dict, path: str) -> pcntoolkit.regression_model.regression_model.RegressionModel
      :classmethod:


      Create model instance from dictionary representation.

      Used for loading models from disk.

      :param dct: Dictionary containing model parameters and configuration
      :type dct: :py:class:`dict`
      :param path: Path to load any associated files
      :type path: :py:class:`str`

      :returns: New instance of the regression model
      :rtype: :py:class:`RegressionModel`

      :raises NotImplementedError: Must be implemented by concrete subclasses



   .. py:method:: to_dict(path: str | None = None) -> dict

      Convert model instance to dictionary representation.

      Used for saving models to disk.

      :param path: Path to save any associated files, by default None
      :type path: :py:class:`str | None`, *optional*

      :returns: Dictionary containing model parameters and configuration
      :rtype: :py:class:`dict`



   .. py:method:: transfer(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> pcntoolkit.regression_model.regression_model.RegressionModel

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


      Check if model includes batch effects.

      :returns: True if model includes batch effects, False otherwise
      :rtype: :py:class:`bool`


   .. py:attribute:: success_ratio
      :value: 1.0



