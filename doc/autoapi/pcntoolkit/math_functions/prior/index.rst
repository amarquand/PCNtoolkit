pcntoolkit.math_functions.prior
===============================

.. py:module:: pcntoolkit.math_functions.prior


Attributes
----------

.. autoapisummary::

   pcntoolkit.math_functions.prior.DEFAULT_PRIOR_ARGS
   pcntoolkit.math_functions.prior.PM_DISTMAP


Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.prior.BasePrior
   pcntoolkit.math_functions.prior.LinearPrior
   pcntoolkit.math_functions.prior.Prior
   pcntoolkit.math_functions.prior.RandomPrior


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.prior.make_prior
   pcntoolkit.math_functions.prior.prior_from_args


Module Contents
---------------

.. py:class:: BasePrior(name: str = 'theta', dims: Optional[Union[Tuple[str, Ellipsis], str]] = None, mapping: str = 'identity', mapping_params: tuple = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: __eq__(other: BasePrior)


   .. py:method:: apply_mapping(x: Any) -> Any


   .. py:method:: compile(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> Any


   .. py:method:: from_dict(dict: BasePrior.from_dict.dict) -> BasePrior
      :classmethod:



   .. py:method:: set_name(name: str) -> None
      :abstractmethod:



   .. py:method:: to_dict()


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> BasePrior
      :abstractmethod:



   .. py:method:: update_data(model, X, be, be_maps, Y)
      :abstractmethod:



   .. py:property:: dims


   .. py:property:: has_random_effect
      :type: bool

      :abstractmethod:



   .. py:attribute:: mapping
      :value: 'identity'



   .. py:attribute:: mapping_params
      :value: (0, 1)



   .. py:attribute:: name
      :value: 'theta'



   .. py:attribute:: sample_dims
      :value: ()



.. py:class:: LinearPrior(slope: Optional[BasePrior] = None, intercept: Optional[BasePrior] = None, name: str = 'theta', dims: Optional[Union[Tuple[str, Ellipsis], str]] = None, mapping: str = 'identity', mapping_params: tuple[float, Ellipsis] = None, basis_function: pcntoolkit.math_functions.basis_function.BasisFunction = LinearBasisFunction(), **kwargs)

   Bases: :py:obj:`BasePrior`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: from_dict(dct)
      :classmethod:



   .. py:method:: set_name(name)


   .. py:method:: to_dict()


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> LinearPrior


   .. py:method:: update_data(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray)


   .. py:attribute:: basis_function


   .. py:property:: dims


   .. py:property:: has_random_effect


   .. py:attribute:: intercept


   .. py:attribute:: sample_dims
      :value: ('observations',)



   .. py:attribute:: slope


.. py:class:: Prior(name: str = 'theta', dims: Optional[Union[Tuple[str, Ellipsis], str]] = None, mapping: str = 'identity', mapping_params: tuple[float, Ellipsis] = None, dist_name: str = 'Normal', dist_params: Tuple[float | int | list[float | int], Ellipsis] = None, **kwargs)

   Bases: :py:obj:`BasePrior`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: from_dict(dct: dict)
      :classmethod:



   .. py:method:: set_name(name: str) -> None


   .. py:method:: to_dict()


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> Prior


   .. py:method:: update_data(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray)


   .. py:attribute:: dist_name
      :value: 'Normal'



   .. py:attribute:: dist_params
      :value: (0, 10.0)



   .. py:property:: has_random_effect


   .. py:attribute:: sample_dims
      :value: ()



.. py:class:: RandomPrior(mu: Optional[BasePrior] = None, sigma: Optional[BasePrior] = None, name: str = 'theta', dims: Optional[Union[Tuple[str, Ellipsis], str]] = None, mapping: str = 'identity', mapping_params: tuple[float, Ellipsis] = None, **kwargs)

   Bases: :py:obj:`BasePrior`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: from_dict(dct)
      :classmethod:



   .. py:method:: set_name(name: str)


   .. py:method:: to_dict()


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> RandomPrior


   .. py:method:: update_data(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray)


   .. py:property:: dims


   .. py:property:: has_random_effect


   .. py:attribute:: mu


   .. py:attribute:: offsets


   .. py:attribute:: sample_dims
      :value: ('observations',)



   .. py:attribute:: scaled_offsets


   .. py:attribute:: sigma


   .. py:attribute:: sigmas


.. py:function:: make_prior(name: str = 'theta', **kwargs) -> BasePrior

.. py:function:: prior_from_args(name: str, args: Dict[str, Any], dims: Optional[Union[Tuple[str, Ellipsis], str]] = None) -> BasePrior

.. py:data:: DEFAULT_PRIOR_ARGS

.. py:data:: PM_DISTMAP

