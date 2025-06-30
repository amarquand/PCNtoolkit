pcntoolkit.math_functions.basis_function
========================================

.. py:module:: pcntoolkit.math_functions.basis_function


Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.basis_function.BasisFunction
   pcntoolkit.math_functions.basis_function.BsplineBasisFunction
   pcntoolkit.math_functions.basis_function.LinearBasisFunction
   pcntoolkit.math_functions.basis_function.PolynomialBasisFunction


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.basis_function.create_basis_function


Module Contents
---------------

.. py:class:: BasisFunction(basis_column: Optional[int | list[int]] = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: copy_column(X: numpy.ndarray, i: int)


   .. py:method:: fit(X: numpy.ndarray) -> None


   .. py:method:: fit_column(X: numpy.ndarray, i: int) -> None


   .. py:method:: from_args(name: str, args: dict) -> BasisFunction
      :classmethod:



   .. py:method:: from_dict(my_dict: dict) -> BasisFunction
      :classmethod:



   .. py:method:: to_dict() -> dict


   .. py:method:: transform(X: numpy.ndarray) -> numpy.ndarray


   .. py:method:: transform_column(X: numpy.ndarray, i: int) -> numpy.ndarray


   .. py:attribute:: basis_name
      :type:  str


   .. py:attribute:: compute_max
      :type:  bool


   .. py:attribute:: compute_min
      :type:  bool


   .. py:attribute:: is_fitted
      :type:  bool


   .. py:attribute:: max
      :type:  dict[int, float]


   .. py:attribute:: min
      :type:  dict[int, float]


.. py:class:: BsplineBasisFunction(basis_column: Optional[Union[int, list[int]]] = None, degree: int = 3, nknots: int = 3, left_expand: float = 0.05, right_expand: float = 0.05, knot_method: str = 'uniform', knots: dict[int, numpy.ndarray] = None, **kwargs)

   Bases: :py:obj:`BasisFunction`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: to_dict() -> dict


   .. py:attribute:: basis_name
      :value: 'bspline'



   .. py:attribute:: degree
      :value: 3



   .. py:attribute:: knot_method
      :value: 'uniform'



   .. py:attribute:: knots


   .. py:attribute:: left_expand
      :value: 0.05



   .. py:attribute:: nknots
      :value: 3



   .. py:attribute:: right_expand
      :value: 0.05



.. py:class:: LinearBasisFunction(basis_column: Optional[Union[int, list[int]]] = None, **kwargs)

   Bases: :py:obj:`BasisFunction`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: basis_name
      :value: 'linear'



.. py:class:: PolynomialBasisFunction(basis_column: Optional[Union[int, list[int]]] = None, degree: int = 3, **kwargs)

   Bases: :py:obj:`BasisFunction`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: basis_name
      :value: 'poly'



   .. py:attribute:: degree
      :value: 3



.. py:function:: create_basis_function(basis_type: str | dict | None, basis_column: Optional[Union[int, list[int]]] = None, **kwargs) -> BasisFunction

