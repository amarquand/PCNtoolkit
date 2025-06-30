pcntoolkit.dataio.norm_data
===========================

.. py:module:: pcntoolkit.dataio.norm_data

.. autoapi-nested-parse::

   norm_data module
   ===============

   This module provides functionalities for normalizing and converting different types of data into a NormData object.

   The NormData object is an xarray.Dataset that contains the data, covariates, batch effects, and response variables, and it
   is used by all the models in the toolkit.



Classes
-------

.. autoapisummary::

   pcntoolkit.dataio.norm_data.NormData


Module Contents
---------------

.. py:class:: NormData(name: str, data_vars: xarray.core.types.DataVars, coords: Mapping[Any, Any], attrs: Mapping[Any, Any] | None = None)

   Bases: :py:obj:`xarray.Dataset`


   A class for handling normative modeling data, extending xarray.Dataset.

   This class provides functionality for loading data for normative modeling.
   It supports various data formats.

   :param name: The name of the dataset
   :type name: :py:class:`str`
   :param data_vars: Data variables for the dataset
   :type data_vars: :py:class:`DataVars`
   :param coords: Coordinates for the dataset
   :type coords: :py:class:`Mapping[Any`, :py:class:`Any]`
   :param attrs: Additional attributes for the dataset, by default None
   :type attrs: :py:class:`Mapping[Any`, :py:class:`Any] | None`, *optional*

   .. attribute:: X

      Covariate data

      :type: :py:class:`xr.DataArray`

   .. attribute:: y

      Response variable data

      :type: :py:class:`xr.DataArray`

   .. attribute:: batch_effects

      Batch effect data

      :type: :py:class:`xr.DataArray`

   .. attribute:: Z

      Z-score data

      :type: :py:class:`xr.DataArray`

   .. attribute:: centiles

      Centile data

      :type: :py:class:`xr.DataArray`

   .. rubric:: Examples

   >>> data = NormData.from_dataframe("my_data", df, covariates, batch_effects, response_vars)
   >>> train_data, test_data = data.train_test_split([0.8, 0.2])

   Initialize a NormData object.

   :param name: The name of the dataset.
   :type name: :py:class:`str`
   :param data_vars: Data variables for the dataset.
   :type data_vars: :py:class:`DataVars`
   :param coords: Coordinates for the dataset.
   :type coords: :py:class:`Mapping[Any`, :py:class:`Any]`
   :param attrs: Additional attributes for the dataset, by default None.
   :type attrs: :py:class:`Mapping[Any`, :py:class:`Any] | None`, *optional*


   .. py:method:: batch_effects_split(batch_effects: Dict[str, List[str]], names: Optional[Tuple[str, str]]) -> Tuple[NormData, NormData]

      Split the data into two datasets, one with the specified batch effects and one without.



   .. py:method:: check_compatibility(other: NormData) -> bool

      Check if the data is compatible with another dataset.

      :param other: Another NormData instance to compare with.
      :type other: :py:class:`NormData`

      :returns: True if compatible, False otherwise
      :rtype: :py:class:`bool`



   .. py:method:: chunk(n_chunks: int) -> Generator[NormData]

      Split the data into n_chunks with roughly equal number of response variables

      :param n_chunks: The number of chunks to split the data into.
      :type n_chunks: :py:class:`int`

      :returns: A generator of NormData instances.
      :rtype: :py:class:`Generator[NormData]`



   .. py:method:: concatenate_string_arrays(*arrays: Any) -> numpy.ndarray

      Concatenate arrays of strings.

      :param arrays: A list of numpy arrays containing strings.
      :type arrays: :py:class:`List[np.ndarray]`

      :returns: A single concatenated numpy array of strings.
      :rtype: :py:class:`np.ndarray`



   .. py:method:: create_statistics_group() -> None

      Initializes a DataArray for statistics with NaN values.

      This method creates a DataArray with dimensions 'response_vars' and 'statistics',
      where 'response_vars' corresponds to the response variables in the dataset,
      and 'statistics' includes statistics such as Rho, RMSE, SMSE, EXPV, NLL, and ShapiroW.
      The DataArray is filled with NaN values initially.



   .. py:method:: from_bids(bids_folder, config_params) -> NormData
      :classmethod:


      Load a normative dataset from a BIDS dataset.

      :param bids_folder: Path to the BIDS folder.
      :type bids_folder: :py:class:`str`
      :param config_params: Configuration parameters for loading the dataset.
      :type config_params: :py:class:`dict`

      :returns: An instance of NormData.
      :rtype: :py:class:`NormData`



   .. py:method:: from_dataframe(name: str, dataframe: pandas.DataFrame, covariates: List[str] | None = None, batch_effects: List[str] | None = None, response_vars: List[str | LiteralString] | None = None, subject_ids: str | None = None, remove_Nan: bool = False, attrs: Mapping[str, Any] | None = None) -> NormData
      :classmethod:


      Load a normative dataset from a pandas DataFrame.

      :param name: The name you want to give to the dataset. Will be used to name saved results.
      :type name: :py:class:`str`
      :param dataframe: The pandas DataFrame to load.
      :type dataframe: :py:class:`pd.DataFrame`
      :param covariates: The list of column names to be used as covariates in the dataset.
      :type covariates: :py:class:`List[str]`
      :param batch_effects: The list of column names to be used as batch effects in the dataset.
      :type batch_effects: :py:class:`List[str]`
      :param response_vars: The list of column names to be used as response variables in the dataset.
      :type response_vars: :py:class:`List[str]`
      :param subject_ids: The name of the column containing the subject IDs
      :type subject_ids: :py:class:`str`
      :param attrs: Additional attributes for the dataset, by default None.
      :type attrs: :py:class:`Mapping[str`, :py:class:`Any] | None`, *optional*
      :param remove_Nan: Wheter or not to remove NAN values from the dataframe before creationg of the class object. By default False
      :type remove_Nan: :py:class:`bool`

      :returns: An instance of NormData.
      :rtype: :py:class:`NormData`



   .. py:method:: from_fsl(fsl_folder, config_params) -> NormData
      :classmethod:


      Load a normative dataset from a FSL file.

      :param fsl_folder: Path to the FSL folder.
      :type fsl_folder: :py:class:`str`
      :param config_params: Configuration parameters for loading the dataset.
      :type config_params: :py:class:`dict`

      :returns: An instance of NormData.
      :rtype: :py:class:`NormData`



   .. py:method:: from_ndarrays(name: str, X: numpy.ndarray, Y: numpy.ndarray, batch_effects: numpy.ndarray | None = None, subject_ids: numpy.ndarray | None = None, attrs: Mapping[str, Any] | None = None) -> NormData
      :classmethod:


      Create a NormData object from numpy arrays.

      :param name: The name of the dataset
      :type name: :py:class:`str`
      :param X: Covariate data of shape (n_samples, n_features)
      :type X: :py:class:`np.ndarray`
      :param y: Response variable data of shape (n_samples, n_responses)
      :type y: :py:class:`np.ndarray`
      :param batch_effects: Batch effect data of shape (n_samples, n_batch_effects)
      :type batch_effects: :py:class:`np.ndarray`
      :param attrs: Additional attributes for the dataset, by default None
      :type attrs: :py:class:`Mapping[str`, :py:class:`Any] | None`, *optional*

      :returns: A new NormData instance containing the provided data
      :rtype: :py:class:`NormData`

      .. rubric:: Notes

      Input arrays are automatically reshaped to 2D if they are 1D



   .. py:method:: from_paths(name: str, covariates_path: str, responses_path: str, batch_effects_path: str, **kwargs) -> NormData
      :classmethod:


      Load a normative dataset from a dictionary of paths.



   .. py:method:: from_xarray(name: str, xarray_dataset: xarray.Dataset) -> NormData
      :classmethod:


      Load a normative dataset from an xarray dataset.

      :param name: The name of the dataset.
      :type name: :py:class:`str`
      :param xarray_dataset: The xarray dataset to load.
      :type xarray_dataset: :py:class:`xr.Dataset`

      :returns: An instance of NormData.
      :rtype: :py:class:`NormData`



   .. py:method:: get_single_batch_effect() -> Dict[str, List[str]]

      Get a single batch effect for each dimension.

      :returns: A dictionary mapping each batch effect dimension to a list containing a single value.
      :rtype: :py:class:`Dict[str`, :py:class:`List[str]]`



   .. py:method:: get_statistics_df() -> pandas.DataFrame

      Get the statistics as a pandas DataFrame.



   .. py:method:: kfold_split(k: int) -> Generator[Tuple[NormData, NormData], Any, Any]

      Perform k-fold splitting of the data.

      :param k: The number of folds.
      :type k: :py:class:`int`

      :returns: A generator yielding training and testing NormData instances for each fold.
      :rtype: :py:class:`Generator[Tuple[NormData`, :py:class:`NormData]`, :py:class:`Any`, :py:class:`Any]`



   .. py:method:: load_centiles(save_dir) -> None


   .. py:method:: load_logp(save_dir) -> None


   .. py:method:: load_results(save_dir: str) -> None

      Loads the results (zscores, centiles, logp, statistics) back into the data

      Args:
          save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv



   .. py:method:: load_statistics(save_dir) -> None


   .. py:method:: load_zscores(save_dir) -> None


   .. py:method:: make_compatible(other: NormData)

      Ensures datasets are compatible by merging the batch effects maps




   .. py:method:: merge(other: NormData) -> NormData

      Merge two NormData objects.

      Drops all columns that are not present in both datasets.



   .. py:method:: register_batch_effects() -> None

      Create a mapping of batch effects to unique values.



   .. py:method:: save_centiles(save_dir: str) -> None


   .. py:method:: save_logp(save_dir: str) -> None


   .. py:method:: save_results(save_dir: str) -> None

      Saves the results (zscores, centiles, logp, statistics) to disk

      Args:
          save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv



   .. py:method:: save_statistics(save_dir: str) -> None


   .. py:method:: save_zscores(save_dir: str) -> None


   .. py:method:: scale_backward(inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None

      Scale the data backward using provided scalers.

      :param inscalers: Scalers for the covariate data.
      :type inscalers: :py:class:`Dict[str`, :py:class:`Any]`
      :param outscalers: Scalers for the response variable data.
      :type outscalers: :py:class:`Dict[str`, :py:class:`Any]`



   .. py:method:: scale_forward(inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None

      Scale the data forward in-place using provided scalers.

      :param inscalers: Scalers for the covariate data.
      :type inscalers: :py:class:`Dict[str`, :py:class:`Any]`
      :param outscalers: Scalers for the response variable data.
      :type outscalers: :py:class:`Dict[str`, :py:class:`Any]`



   .. py:method:: select_batch_effects(name, batch_effects: Dict[str, List[str]], invert: bool = False) -> NormData

      Select only the specified batch effects.

      :param batch_effects: A dictionary specifying which batch effects to select.
      :type batch_effects: :py:class:`Dict[str`, :py:class:`List[str]]`

      :returns: A NormData instance with the selected batch effects.
      :rtype: :py:class:`NormData`



   .. py:method:: to_dataframe(dim_order: Sequence[Hashable] | None = None) -> pandas.DataFrame

      Convert the NormData instance to a pandas DataFrame.

      :param dim_order: The order of dimensions for the DataFrame, by default None.
      :type dim_order: :py:class:`Sequence[Hashable] | None`, *optional*

      :returns: A DataFrame representation of the NormData instance.
      :rtype: :py:class:`pd.DataFrame`



   .. py:method:: train_test_split(splits: Tuple[float, Ellipsis] | List[float] | float = 0.8, split_names: Tuple[str, Ellipsis] | None = None, random_state: int = 42) -> Tuple[NormData, Ellipsis]

      Split the data into training and testing datasets.

      :param splits: A tuple (train_size, test_size), specifying the proportion of data for each split. Or a float specifying the proportion of data for the train set.
      :type splits: :py:class:`Tuple[float`, :py:class:`...] | List[float] | float`
      :param split_names: Names for the splits, by default None.
      :type split_names: :py:class:`Tuple[str`, :py:class:`...] | None`, *optional*
      :param random_state: Random state for splits, by default 42.
      :type random_state: int , *optional*

      :returns: A tuple containing the training and testing NormData instances.
      :rtype: :py:class:`Tuple[NormData`, :py:class:`...]`



   .. py:attribute:: __slots__
      :value: ('unique_batch_effects', 'batch_effect_counts', 'batch_effect_covariate_ranges',...



   .. py:property:: name
      :type: str


      Get the name of the dataset.

      :returns: The name of the dataset.
      :rtype: :py:class:`str`


   .. py:property:: response_var_list
      :type: xarray.DataArray



