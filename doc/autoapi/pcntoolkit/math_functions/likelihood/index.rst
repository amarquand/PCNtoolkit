pcntoolkit.math_functions.likelihood
====================================

.. py:module:: pcntoolkit.math_functions.likelihood


Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.likelihood.BetaLikelihood
   pcntoolkit.math_functions.likelihood.Likelihood
   pcntoolkit.math_functions.likelihood.NormalLikelihood
   pcntoolkit.math_functions.likelihood.SHASHbLikelihood
   pcntoolkit.math_functions.likelihood.SHASHo2Likelihood
   pcntoolkit.math_functions.likelihood.SHASHoLikelihood


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.likelihood.get_default_normal_likelihood


Module Contents
---------------

.. py:class:: BetaLikelihood(alpha: pcntoolkit.math_functions.prior.BasePrior, beta: pcntoolkit.math_functions.prior.BasePrior)

   Bases: :py:obj:`Likelihood`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)


   .. py:method:: compile_params(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> dict[str, Any]


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: get_var_names() -> List[str]


   .. py:method:: has_random_effect() -> bool


   .. py:method:: to_dict() -> Dict[str, Any]


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> BetaLikelihood


   .. py:method:: yhat(*args, **kwargs)


   .. py:attribute:: alpha


   .. py:attribute:: beta


.. py:class:: Likelihood(name: str)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)
      :abstractmethod:



   .. py:method:: compile(X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> pymc.Model


   .. py:method:: compile_params(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> dict[str, Any]
      :abstractmethod:



   .. py:method:: create_model_with_data(X, be, be_maps, Y) -> pymc.Model


   .. py:method:: forward(*args, **kwargs)
      :abstractmethod:



   .. py:method:: from_args(args: Dict[str, Any]) -> Likelihood
      :staticmethod:



   .. py:method:: from_dict(dct: Dict[str, Any]) -> Likelihood
      :staticmethod:



   .. py:method:: has_random_effect() -> bool
      :abstractmethod:



   .. py:method:: to_dict() -> Dict[str, Any]
      :abstractmethod:



   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> Likelihood
      :abstractmethod:



   .. py:method:: update_data(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray)


   .. py:method:: yhat(*args, **kwargs)
      :abstractmethod:



   .. py:attribute:: name


.. py:class:: NormalLikelihood(mu: pcntoolkit.math_functions.prior.BasePrior, sigma: pcntoolkit.math_functions.prior.BasePrior)

   Bases: :py:obj:`Likelihood`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)


   .. py:method:: compile_params(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> dict[str, Any]


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: has_random_effect() -> bool


   .. py:method:: to_dict() -> Dict[str, Any]


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> Likelihood


   .. py:method:: yhat(*args, **kwargs)


   .. py:attribute:: mu


   .. py:attribute:: sigma


.. py:class:: SHASHbLikelihood(mu: pcntoolkit.math_functions.prior.BasePrior, sigma: pcntoolkit.math_functions.prior.BasePrior, epsilon: pcntoolkit.math_functions.prior.BasePrior, delta: pcntoolkit.math_functions.prior.BasePrior)

   Bases: :py:obj:`Likelihood`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)


   .. py:method:: compile_params(model: pymc.Model, X: xarray.DataArray, be: xarray.DataArray, be_maps: dict[str, dict[str, int]], Y: xarray.DataArray) -> dict[str, Any]


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: get_var_names() -> List[str]


   .. py:method:: has_random_effect() -> bool


   .. py:method:: to_dict() -> Dict[str, Any]


   .. py:method:: transfer(idata: arviz.InferenceData, **kwargs) -> SHASHbLikelihood


   .. py:method:: yhat(*args, **kwargs)


   .. py:attribute:: delta


   .. py:attribute:: epsilon


   .. py:attribute:: mu


   .. py:attribute:: sigma


.. py:class:: SHASHo2Likelihood(mu: pcntoolkit.math_functions.prior.BasePrior, sigma: pcntoolkit.math_functions.prior.BasePrior, epsilon: pcntoolkit.math_functions.prior.BasePrior, delta: pcntoolkit.math_functions.prior.BasePrior)

   Bases: :py:obj:`Likelihood`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: get_var_names() -> List[str]


   .. py:method:: has_random_effect() -> bool


   .. py:method:: to_dict() -> Dict[str, Any]


   .. py:attribute:: delta


   .. py:attribute:: epsilon


   .. py:attribute:: mu


   .. py:attribute:: sigma


.. py:class:: SHASHoLikelihood(mu: pcntoolkit.math_functions.prior.BasePrior, sigma: pcntoolkit.math_functions.prior.BasePrior, epsilon: pcntoolkit.math_functions.prior.BasePrior, delta: pcntoolkit.math_functions.prior.BasePrior)

   Bases: :py:obj:`Likelihood`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: backward(*args, **kwargs)


   .. py:method:: forward(*args, **kwargs)


   .. py:method:: get_var_names() -> List[str]


   .. py:method:: has_random_effect() -> bool


   .. py:method:: to_dict() -> Dict[str, Any]


   .. py:attribute:: delta


   .. py:attribute:: epsilon


   .. py:attribute:: mu


   .. py:attribute:: sigma


.. py:function:: get_default_normal_likelihood() -> NormalLikelihood

   Return a normal likelihood with a random intercept of the mean, and heteroskedasticity


