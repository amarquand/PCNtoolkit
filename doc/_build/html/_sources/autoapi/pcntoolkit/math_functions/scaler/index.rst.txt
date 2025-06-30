pcntoolkit.math_functions.scaler
================================

.. py:module:: pcntoolkit.math_functions.scaler

.. autoapi-nested-parse::

   Data scaling and normalization module for PCNToolkit.

   This module provides various scaling implementations for data preprocessing,
   including standardization, min-max scaling, robust scaling, and identity scaling.
   All scalers implement a common interface defined by the abstract base class Scaler.

   The module supports the following scaling operations:
       - Standardization (zero mean, unit variance)
       - Min-max scaling (to [0,1] range)
       - Robust min-max scaling (using percentiles)
       - Identity scaling (no transformation)

   Each scaler supports:
       - Fitting to training data
       - Transforming new data
       - Inverse transforming scaled data
       - Serialization to/from dictionaries
       - Optional outlier adjustment

   Available Classes
   ---------------
   Scaler : ABC
       Abstract base class defining the scaler interface
   StandardScaler
       Standardizes features to zero mean and unit variance
   MinMaxScaler
       Scales features to a fixed range [0, 1]
   RobustMinMaxScaler
       Scales features using robust statistics based on percentiles
   IdentityScaler
       Passes data through unchanged

   .. rubric:: Examples

   >>> from pcntoolkit.dataio.scaler import StandardScaler
   >>> import numpy as np
   >>> X = np.array([[1, 2], [3, 4], [5, 6]])
   >>> scaler = StandardScaler()
   >>> X_scaled = scaler.fit_transform(X)

   .. rubric:: Notes

   All scalers support both fitting to the entire dataset and transforming specific
   indices of features, allowing for flexible scaling strategies. The scalers can
   be serialized to dictionaries for saving/loading trained parameters.

   .. seealso::

      :py:obj:`pcntoolkit.normative_model`
          Uses scalers for data preprocessing
      
      :py:obj:`pcntoolkit.dataio.basis_expansions`
          Complementary data transformations



Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.scaler.IdentityScaler
   pcntoolkit.math_functions.scaler.MinMaxScaler
   pcntoolkit.math_functions.scaler.RobustMinMaxScaler
   pcntoolkit.math_functions.scaler.Scaler
   pcntoolkit.math_functions.scaler.StandardScaler


Module Contents
---------------

.. py:class:: IdentityScaler(adjust_outliers: bool = False)

   Bases: :py:obj:`Scaler`


   A scaler that returns the input unchanged.

   This scaler is useful as a placeholder when no scaling is desired
   but a scaler object is required by the API.

   :param adjust_outliers: Has no effect for this scaler, included for API compatibility
   :type adjust_outliers: :py:class:`bool`, *optional*

   .. rubric:: Examples

   >>> import numpy as np
   >>> from pcntoolkit.dataio.scaler import IdentityScaler
   >>> X = np.array([[1, 2], [3, 4]])
   >>> scaler = IdentityScaler()
   >>> X_scaled = scaler.fit_transform(X)
   >>> np.array_equal(X, X_scaled)
   True


   .. py:method:: fit(X: numpy.typing.NDArray) -> None

      Compute the parameters needed for scaling.

      :param X: Training data to fit the scaler on, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`



   .. py:method:: from_dict(my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> IdentityScaler
      :classmethod:


      Create a scaler instance from a dictionary.

      :param my_dict: Dictionary containing scaler parameters. Must include 'scaler_type' key.
      :type my_dict: :py:class:`Dict[str`, :py:class:`Union[bool`, :py:class:`str`, :py:class:`float`, :py:class:`List[float]]]`

      :returns: Instance of the appropriate scaler subclass
      :rtype: :py:class:`Scaler`

      :raises ValueError: If scaler_type is missing or invalid



   .. py:method:: inverse_transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Inverse transform scaled data back to original scale.

      :param X: Data to inverse transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to inverse transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Inverse transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:method:: to_dict() -> Dict[str, Union[bool, str, float, List[float]]]

      Convert scaler instance to dictionary.



   .. py:method:: transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Transform the data using the fitted scaler.

      :param X: Data to transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



.. py:class:: MinMaxScaler(adjust_outliers: bool = False)

   Bases: :py:obj:`Scaler`


   Scale features to a fixed range (0, 1).

   Transforms features by scaling each feature to a given range (default [0, 1]):
   X_scaled = (X - X_min) / (X_max - X_min)

   :param adjust_outliers: Whether to clip transformed values to [0, 1], by default True
   :type adjust_outliers: :py:class:`bool`, *optional*

   .. attribute:: min

      Minimum value for each feature from training data

      :type: :py:class:`Optional[NDArray]`

   .. attribute:: max

      Maximum value for each feature from training data

      :type: :py:class:`Optional[NDArray]`

   .. rubric:: Examples

   >>> import numpy as np
   >>> from pcntoolkit.dataio.scaler import MinMaxScaler
   >>> X = np.array([[1, 2], [3, 4], [5, 6]])
   >>> scaler = MinMaxScaler()
   >>> X_scaled = scaler.fit_transform(X)
   >>> print(X_scaled.min(axis=0))  # [0, 0]
   >>> print(X_scaled.max(axis=0))  # [1, 1]


   .. py:method:: fit(X: numpy.typing.NDArray) -> None

      Compute the parameters needed for scaling.

      :param X: Training data to fit the scaler on, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`



   .. py:method:: from_dict(my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> MinMaxScaler
      :classmethod:


      Create a scaler instance from a dictionary.

      :param my_dict: Dictionary containing scaler parameters. Must include 'scaler_type' key.
      :type my_dict: :py:class:`Dict[str`, :py:class:`Union[bool`, :py:class:`str`, :py:class:`float`, :py:class:`List[float]]]`

      :returns: Instance of the appropriate scaler subclass
      :rtype: :py:class:`Scaler`

      :raises ValueError: If scaler_type is missing or invalid



   .. py:method:: inverse_transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Inverse transform scaled data back to original scale.

      :param X: Data to inverse transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to inverse transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Inverse transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:method:: to_dict() -> Dict[str, Union[bool, str, float, List[float]]]

      Convert scaler instance to dictionary.



   .. py:method:: transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Transform the data using the fitted scaler.

      :param X: Data to transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:attribute:: max
      :type:  Optional[numpy.typing.NDArray]
      :value: None



   .. py:attribute:: min
      :type:  Optional[numpy.typing.NDArray]
      :value: None



.. py:class:: RobustMinMaxScaler(adjust_outliers: bool = False, tail: float = 0.05)

   Bases: :py:obj:`MinMaxScaler`


   Scale features using robust statistics based on percentiles.

   Similar to MinMaxScaler but uses percentile-based statistics to be
   robust to outliers.

   :param adjust_outliers: Whether to clip transformed values to [0, 1], by default True
   :type adjust_outliers: :py:class:`bool`, *optional*
   :param tail: The percentile to use for computing robust min/max, by default 0.05
                (5th and 95th percentiles)
   :type tail: :py:class:`float`, *optional*

   .. attribute:: min

      Robust minimum for each feature from training data

      :type: :py:class:`Optional[NDArray]`

   .. attribute:: max

      Robust maximum for each feature from training data

      :type: :py:class:`Optional[NDArray]`

   .. attribute:: tail

      The percentile value used for robust statistics

      :type: :py:class:`float`

   .. rubric:: Examples

   >>> import numpy as np
   >>> from pcntoolkit.dataio.scaler import RobustMinMaxScaler
   >>> X = np.array([[1, 2], [3, 4], [100, 6]])  # with outlier
   >>> scaler = RobustMinMaxScaler(tail=0.1)
   >>> X_scaled = scaler.fit_transform(X)


   .. py:method:: fit(X: numpy.typing.NDArray) -> None

      Compute the parameters needed for scaling.

      :param X: Training data to fit the scaler on, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`



   .. py:method:: from_dict(my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> RobustMinMaxScaler
      :classmethod:


      Create a scaler instance from a dictionary.

      :param my_dict: Dictionary containing scaler parameters. Must include 'scaler_type' key.
      :type my_dict: :py:class:`Dict[str`, :py:class:`Union[bool`, :py:class:`str`, :py:class:`float`, :py:class:`List[float]]]`

      :returns: Instance of the appropriate scaler subclass
      :rtype: :py:class:`Scaler`

      :raises ValueError: If scaler_type is missing or invalid



   .. py:method:: to_dict() -> Dict[str, Union[bool, str, float, List[float]]]

      Convert scaler instance to dictionary.



   .. py:attribute:: tail
      :value: 0.05



.. py:class:: Scaler(adjust_outliers: bool = False)

   Bases: :py:obj:`abc.ABC`


   Abstract base class for data scaling operations.

   This class defines the interface for all scaling operations in PCNToolkit.
   Concrete implementations must implement fit, transform, inverse_transform,
   and to_dict methods.

   :param adjust_outliers: Whether to clip values to valid ranges, by default True
   :type adjust_outliers: :py:class:`bool`, *optional*

   .. rubric:: Notes

   All scaling operations support both fitting to the entire dataset and
   transforming specific indices of features, allowing for flexible scaling
   strategies.


   .. py:method:: fit(X: numpy.typing.NDArray) -> None
      :abstractmethod:


      Compute the parameters needed for scaling.

      :param X: Training data to fit the scaler on, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`



   .. py:method:: fit_transform(X: numpy.typing.NDArray) -> numpy.typing.NDArray

      Fit the scaler and transform the data in one step.

      :param X: Data to fit and transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`

      :returns: Transformed data
      :rtype: :py:class:`NDArray`



   .. py:method:: from_dict(my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> Scaler
      :classmethod:


      Create a scaler instance from a dictionary.

      :param my_dict: Dictionary containing scaler parameters. Must include 'scaler_type' key.
      :type my_dict: :py:class:`Dict[str`, :py:class:`Union[bool`, :py:class:`str`, :py:class:`float`, :py:class:`List[float]]]`

      :returns: Instance of the appropriate scaler subclass
      :rtype: :py:class:`Scaler`

      :raises ValueError: If scaler_type is missing or invalid



   .. py:method:: from_string(scaler_type: str, **kwargs: Any) -> Scaler
      :staticmethod:


      Create a scaler instance from a string identifier.

      :param scaler_type: The type of scaling to apply. Options are:
                          - "standardize": zero mean, unit variance
                          - "minmax": scale to range [0,1]
                          - "robminmax": robust minmax scaling using percentiles
                          - "id" or "none": no scaling
      :type scaler_type: :py:class:`str`
      :param \*\*kwargs: Additional arguments to pass to the scaler constructor
      :type \*\*kwargs: :py:class:`dict`

      :returns: Instance of the appropriate scaler class
      :rtype: :py:class:`Scaler`



   .. py:method:: inverse_transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray
      :abstractmethod:


      Inverse transform scaled data back to original scale.

      :param X: Data to inverse transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to inverse transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Inverse transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:method:: to_dict() -> Dict[str, Union[bool, str, float, List[float]]]
      :abstractmethod:


      Convert scaler instance to dictionary.



   .. py:method:: transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray
      :abstractmethod:


      Transform the data using the fitted scaler.

      :param X: Data to transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:attribute:: adjust_outliers
      :value: False



.. py:class:: StandardScaler(adjust_outliers: bool = False)

   Bases: :py:obj:`Scaler`


   Standardize features by removing the mean and scaling to unit variance.

   This scaler transforms the data to have zero mean and unit variance:
   z = (x - μ) / σ

   :param adjust_outliers: Whether to clip extreme values, by default True
   :type adjust_outliers: :py:class:`bool`, *optional*

   .. attribute:: m

      Mean of the training data

      :type: :py:class:`Optional[NDArray]`

   .. attribute:: s

      Standard deviation of the training data

      :type: :py:class:`Optional[NDArray]`

   .. rubric:: Examples

   >>> import numpy as np
   >>> from pcntoolkit.dataio.scaler import StandardScaler
   >>> X = np.array([[1, 2], [3, 4], [5, 6]])
   >>> scaler = StandardScaler()
   >>> X_scaled = scaler.fit_transform(X)
   >>> print(X_scaled.mean(axis=0))  # approximately [0, 0]
   >>> print(X_scaled.std(axis=0))  # approximately [1, 1]


   .. py:method:: fit(X: numpy.typing.NDArray) -> None

      Compute the parameters needed for scaling.

      :param X: Training data to fit the scaler on, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`



   .. py:method:: from_dict(my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> StandardScaler
      :classmethod:


      Create a scaler instance from a dictionary.

      :param my_dict: Dictionary containing scaler parameters. Must include 'scaler_type' key.
      :type my_dict: :py:class:`Dict[str`, :py:class:`Union[bool`, :py:class:`str`, :py:class:`float`, :py:class:`List[float]]]`

      :returns: Instance of the appropriate scaler subclass
      :rtype: :py:class:`Scaler`

      :raises ValueError: If scaler_type is missing or invalid



   .. py:method:: inverse_transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Inverse transform scaled data back to original scale.

      :param X: Data to inverse transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to inverse transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Inverse transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:method:: to_dict() -> Dict[str, Union[bool, str, float, List[float]]]

      Convert scaler instance to dictionary.



   .. py:method:: transform(X: numpy.typing.NDArray, index: Optional[numpy.typing.NDArray] = None) -> numpy.typing.NDArray

      Transform the data using the fitted scaler.

      :param X: Data to transform, shape (n_samples, n_features)
      :type X: :py:class:`NDArray`
      :param index: Indices of features to transform, by default None (transform all)
      :type index: :py:class:`Optional[NDArray]`, *optional*

      :returns: Transformed data
      :rtype: :py:class:`NDArray`

      :raises ValueError: If the scaler has not been fitted



   .. py:attribute:: m
      :type:  Optional[numpy.typing.NDArray]
      :value: None



   .. py:attribute:: s
      :type:  Optional[numpy.typing.NDArray]
      :value: None



