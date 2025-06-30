pcntoolkit.math_functions.thrive
================================

.. py:module:: pcntoolkit.math_functions.thrive


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.thrive.design_matrix
   pcntoolkit.math_functions.thrive.fill_missing
   pcntoolkit.math_functions.thrive.fisher_transform
   pcntoolkit.math_functions.thrive.get_correlation_matrix
   pcntoolkit.math_functions.thrive.get_thrive_Z_X
   pcntoolkit.math_functions.thrive.offset_indices


Module Contents
---------------

.. py:function:: design_matrix(bandwidth: int, Sigma: numpy.ndarray) -> pandas.DataFrame

   Constructs a design matrix according to: Buuren, S. Evaluation and prediction of individual growth trajectories. Ann. Hum. Biol. 50, 247–257 (2023).

   Args:
       bandwidth (int): The bandwidth for which the covariance has been computed
       Sigma np.ndarray: Covariate matrix with possibly missing values. The 0'th column represents an age of 0.
   Returns:
       pd.DataFrame: A design matrix with regressors and predictors. The matrix may have missing values in the 'y' column.


.. py:function:: fill_missing(bandwidth: int, cors: numpy.ndarray) -> numpy.ndarray

   Fills in missing correlation values according to:

   Args:
       bandwidth (int): the bandwidth within which the indices are filled in
       cors (np.ndarray): possibly incomplete correlation matrix of shape [n_responsevars, n_ages, n_ages]

   Returns:
       np.ndarray: New matrix completed with predicted values


.. py:function:: fisher_transform(cor)

.. py:function:: get_correlation_matrix(data: pcntoolkit.dataio.norm_data.NormData, bandwidth: int, covariate_name='age')

   Compute correlations of Z scores between pairs of observations of the same subject at different ages

   Args:
       data (NormData): Data containing covariates, predicted Z-scores, batch effects and subject indices
       bandwidth (int): The age offset range within which correlations are computed
       covariate_name (str, optional): Covariate to use for grouping subjects. Defaults to "age".

   Returns:
       xr.DataArray: Correlations of shape [n_response_vars, n_ages, n_ages]


.. py:function:: get_thrive_Z_X(cors: xarray.DataArray, start_x: xarray.DataArray, start_z: xarray.DataArray, span: int, z_thrive=1.96)

.. py:function:: offset_indices(max_age: int, bandwidth: int)

   Generate pairs of indices that iterate over all the cells in the upper triangular region specified by the parameters.

   E.g:
   Offset_indices(3, 2) will yield (0,1) -> (0,2) -> (1,2) -> (1,3) -> (2,3)
   Which index these positions:
   _,0,1,_
   _,_,2,3
   _,_,_,4
   _,_,_,_

   Args:
       max_age (int): max age for which indices are generated (includes 0)
       bandwidth (int): the bandwidth within which the indices are computed

   Yields:
       (int, int): pairs of indices


