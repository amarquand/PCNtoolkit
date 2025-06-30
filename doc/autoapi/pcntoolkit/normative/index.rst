pcntoolkit.normative
====================

.. py:module:: pcntoolkit.normative

.. autoapi-nested-parse::

   Module providing entry points for fitting and predicting with normative models from the command line.



Functions
---------

.. autoapisummary::

   pcntoolkit.normative.entrypoint
   pcntoolkit.normative.fit
   pcntoolkit.normative.fit_predict
   pcntoolkit.normative.get_argparser
   pcntoolkit.normative.get_conf_dict_from_args
   pcntoolkit.normative.load_data
   pcntoolkit.normative.load_response_vars
   pcntoolkit.normative.load_test_data
   pcntoolkit.normative.main
   pcntoolkit.normative.make_synthetic_data
   pcntoolkit.normative.predict


Module Contents
---------------

.. py:function:: entrypoint(*args)

.. py:function:: fit(conf_dict: dict) -> None

   Fit a new normative model.

   :param conf_dict: Dictionary containing configuration options


.. py:function:: fit_predict(conf_dict: dict) -> None

   Fit a normative model and predict response variables.

   :param conf_dict: Dictionary containing configuration options


.. py:function:: get_argparser() -> argparse.ArgumentParser

   Get an argument parser for the normative modeling functions.

   Returns:
       argparse.ArgumentParser: The argument parser


.. py:function:: get_conf_dict_from_args() -> dict[str, str | int | float | bool]

   Parse the arguments and return a dictionary with the configuration options.

   Raises:
       ValueError: Raised if an argument is specified twice.

   Returns:
       dict[str, str | int | float | bool]: A dictionary with the configuration option, parsed to the correct type.


.. py:function:: load_data(conf_dict: dict) -> pcntoolkit.dataio.norm_data.NormData

   Load the data from the configuration dictionary.

   Returns:
       NormData: NormData object containing the data


.. py:function:: load_response_vars(datafile: str, maskfile: str | None = None, vol: bool = True) -> tuple[numpy.ndarray, numpy.ndarray | None]

   Load response variables from file. This will load the data and mask it if
   necessary. If the data is in ascii format it will be converted into a numpy
   array. If the data is in neuroimaging format it will be reshaped into a
   2D array (observations x variables) and a mask will be created if necessary.

   :param datafile: File containing the response variables
   :param maskfile: Mask file (nifti only)
   :param vol: If True, load the data as a 4D volume (nifti only)
   :returns Y: Response variables
   :returns volmask: Mask file (nifti only)


.. py:function:: load_test_data(conf_dict: dict) -> pcntoolkit.dataio.norm_data.NormData

   Load the test data from the file specified in the configuration dictionary.

   Args:
       conf_dict (dict): dictionary containing the configuration options

   Returns:
       NormData: NormData object containing the test data


.. py:function:: main(*args) -> None

   Main function to run the normative modeling functions.

   Raises:
       ValueError: If the function specified in the configuration dictionary is unknown.



.. py:function:: make_synthetic_data() -> None

   Create synthetic data for testing.


.. py:function:: predict(conf_dict: dict) -> None

   Predict response variables using a saved normative model.

   :param conf_dict: Dictionary containing configuration options


