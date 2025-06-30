pcntoolkit.math_functions.shash
===============================

.. py:module:: pcntoolkit.math_functions.shash

.. autoapi-nested-parse::

   Sinh-Arcsinh (SHASH) Distribution Implementation Module.

   This module implements the Sinh-Arcsinh (SHASH) distribution and its variants as described in
   Jones and Pewsey (2009) [1]_. The SHASH distribution is a flexible distribution family that can
   model skewness and kurtosis through separate parameters.

   The module provides:

   1. Basic SHASH transformations (S, S_inv, C)

   2. SHASH distribution (base implementation)

   3. SHASHo distribution (location-scale variant)

   4. SHASHo2 distribution (alternative parameterization)

   5. SHASHb distribution (standardized variant)


   .. rubric:: References

   .. [1] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika, 96(4), 761-780.
          https://doi.org/10.1093/biomet/asp053

   .. rubric:: Notes

   The implementation uses PyMC and PyTensor for probabilistic programming capabilities.
   All distributions support random sampling and log-probability calculations.



Attributes
----------

.. autoapisummary::

   pcntoolkit.math_functions.shash.CONST1
   pcntoolkit.math_functions.shash.CONST2
   pcntoolkit.math_functions.shash.kv
   pcntoolkit.math_functions.shash.shash
   pcntoolkit.math_functions.shash.shashb
   pcntoolkit.math_functions.shash.shasho
   pcntoolkit.math_functions.shash.shasho2


Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.shash.Kv
   pcntoolkit.math_functions.shash.SHASH
   pcntoolkit.math_functions.shash.SHASHb
   pcntoolkit.math_functions.shash.SHASHbRV
   pcntoolkit.math_functions.shash.SHASHo
   pcntoolkit.math_functions.shash.SHASHo2
   pcntoolkit.math_functions.shash.SHASHo2RV
   pcntoolkit.math_functions.shash.SHASHoRV
   pcntoolkit.math_functions.shash.SHASHrv


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.shash.K
   pcntoolkit.math_functions.shash.P
   pcntoolkit.math_functions.shash.S
   pcntoolkit.math_functions.shash.S_inv
   pcntoolkit.math_functions.shash.m


Module Contents
---------------

.. py:class:: Kv(output_types_preference=None, name=None)

   Bases: :py:obj:`pytensor.scalar.basic.BinaryScalarOp`


   An `Op` with a C implementation.


   .. py:method:: grad(inputs: Sequence[pytensor.graph.basic.Variable[Any, Any]], output_gradients: Sequence[pytensor.graph.basic.Variable[Any, Any]]) -> List[pytensor.graph.basic.Variable]

      Construct a graph for the gradient with respect to each input variable.

      Each returned `Variable` represents the gradient with respect to that
      input computed based on the symbolic gradients with respect to each
      output. If the output is not differentiable with respect to an input,
      then this method should return an instance of type `NullType` for that
      input.

      Using the reverse-mode AD characterization given in [1]_, for a
      :math:`C = f(A, B)` representing the function implemented by the `Op`
      and its two arguments :math:`A` and :math:`B`, given by the
      `Variable`\s in `inputs`, the values returned by `Op.grad` represent
      the quantities :math:`\bar{A} \equiv \frac{\partial S_O}{A}` and
      :math:`\bar{B}`, for some scalar output term :math:`S_O` of :math:`C`
      in

      .. math::

          \operatorname{Tr}\left(\bar{C}^\top dC\right) =
              \operatorname{Tr}\left(\bar{A}^\top dA\right) +
              \operatorname{Tr}\left(\bar{B}^\top dB\right)


      :param inputs: The input variables.
      :param output_grads: The gradients of the output variables.

      :returns: The gradients with respect to each `Variable` in `inputs`.
      :rtype: :py:class:`grads`

      .. rubric:: References

      .. [1] Giles, Mike. 2008. “An Extended Collection of Matrix Derivative Results for Forward and Reverse Mode Automatic Differentiation.”



   .. py:method:: impl(p: Union[float, int], x: Union[float, int]) -> float


   .. py:method:: st_impl(p: Union[float, int], x: Union[float, int]) -> float
      :staticmethod:



   .. py:attribute:: nfunc_spec
      :value: ('scipy.special.kv', 2, 1)



.. py:class:: SHASH

   Bases: :py:obj:`pymc.distributions.Continuous`


   Base class for continuous distributions.

   Add a tensor variable corresponding to a PyMC distribution to the current model.

   Note that all remaining kwargs must be compatible with ``.dist()``

   :param cls: A PyMC distribution.
   :type cls: :py:class:`type`
   :param name: Name for the new model variable.
   :type name: :py:class:`str`
   :param rng: Random number generator to use with the RandomVariable.
   :type rng: *optional*
   :param dims: A tuple of dimension names known to the model. When shape is not provided,
                the shape of dims is used to define the shape of the variable.
   :type dims: :py:class:`tuple`, *optional*
   :param initval: Numeric or symbolic untransformed initial value of matching shape,
                   or one of the following initial value strategies: "support_point", "prior".
                   Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
                   or support_point-based initial values in the transformed space.
   :type initval: *optional*
   :param observed: Observed data to be passed when registering the random variable in the model.
                    When neither shape nor dims is provided, the shape of observed is used to
                    define the shape of the variable.
                    See ``Model.register_rv``.
   :type observed: *optional*
   :param total_size: See ``Model.register_rv``.
   :type total_size: :py:class:`float`, *optional*
   :param transform: See ``Model.register_rv``.
   :type transform: *optional*
   :param \*\*kwargs: Keyword arguments that will be forwarded to ``.dist()`` or the PyTensor RV Op.
                      Most prominently: ``shape`` for ``.dist()`` or ``dtype`` for the Op.

   :returns: **rv** -- The created random variable tensor, registered in the Model.
   :rtype: :py:class:`TensorVariable`


   .. py:method:: P(q: float) -> float
      :staticmethod:



   .. py:method:: dist(epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, **kwargs: Any) -> Any
      :classmethod:


      Create a tensor variable corresponding to the `cls` distribution.

      :param dist_params: The inputs to the `RandomVariable` `Op`.
      :type dist_params: :py:class:`array-like`
      :param shape: A tuple of sizes for each dimension of the new RV.
      :type shape: :py:class:`int`, :py:class:`tuple`, :py:class:`Variable`, *optional*
      :param \*\*kwargs: Keyword arguments that will be forwarded to the PyTensor RV Op.
                         Most prominently: ``size`` or ``dtype``.

      :returns: **rv** -- The created random variable tensor.
      :rtype: :py:class:`TensorVariable`



   .. py:method:: logp(epsilon: float, delta: float) -> float


   .. py:method:: m1(epsilon: float, delta: float) -> float
      :staticmethod:



   .. py:method:: m1m2(epsilon: float, delta: float) -> Tuple[float, float]
      :staticmethod:



   .. py:method:: m2(epsilon: float, delta: float) -> float
      :staticmethod:



   .. py:attribute:: my_K


   .. py:attribute:: rv_op


.. py:class:: SHASHb

   Bases: :py:obj:`pymc.distributions.Continuous`


   Base class for continuous distributions.

   Add a tensor variable corresponding to a PyMC distribution to the current model.

   Note that all remaining kwargs must be compatible with ``.dist()``

   :param cls: A PyMC distribution.
   :type cls: :py:class:`type`
   :param name: Name for the new model variable.
   :type name: :py:class:`str`
   :param rng: Random number generator to use with the RandomVariable.
   :type rng: *optional*
   :param dims: A tuple of dimension names known to the model. When shape is not provided,
                the shape of dims is used to define the shape of the variable.
   :type dims: :py:class:`tuple`, *optional*
   :param initval: Numeric or symbolic untransformed initial value of matching shape,
                   or one of the following initial value strategies: "support_point", "prior".
                   Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
                   or support_point-based initial values in the transformed space.
   :type initval: *optional*
   :param observed: Observed data to be passed when registering the random variable in the model.
                    When neither shape nor dims is provided, the shape of observed is used to
                    define the shape of the variable.
                    See ``Model.register_rv``.
   :type observed: *optional*
   :param total_size: See ``Model.register_rv``.
   :type total_size: :py:class:`float`, *optional*
   :param transform: See ``Model.register_rv``.
   :type transform: *optional*
   :param \*\*kwargs: Keyword arguments that will be forwarded to ``.dist()`` or the PyTensor RV Op.
                      Most prominently: ``shape`` for ``.dist()`` or ``dtype`` for the Op.

   :returns: **rv** -- The created random variable tensor, registered in the Model.
   :rtype: :py:class:`TensorVariable`


   .. py:method:: dist(mu: pytensor.tensor.TensorLike, sigma: pytensor.tensor.TensorLike, epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, **kwargs: Any) -> Any
      :classmethod:


      Create a tensor variable corresponding to the `cls` distribution.

      :param dist_params: The inputs to the `RandomVariable` `Op`.
      :type dist_params: :py:class:`array-like`
      :param shape: A tuple of sizes for each dimension of the new RV.
      :type shape: :py:class:`int`, :py:class:`tuple`, :py:class:`Variable`, *optional*
      :param \*\*kwargs: Keyword arguments that will be forwarded to the PyTensor RV Op.
                         Most prominently: ``size`` or ``dtype``.

      :returns: **rv** -- The created random variable tensor.
      :rtype: :py:class:`TensorVariable`



   .. py:method:: logp(mu: float, sigma: float, epsilon: float, delta: float) -> float


   .. py:attribute:: rv_op


.. py:class:: SHASHbRV(name=None, ndim_supp=None, ndims_params=None, dtype: str | None = None, inplace=None, signature: str | None = None)

   Bases: :py:obj:`pytensor.tensor.random.op.RandomVariable`


   An `Op` that produces a sample from a random variable.

   This is essentially `RandomFunction`, except that it removes the
   `outtype` dependency and handles shape dimension information more
   directly.


   Create a random variable `Op`.

   :param name: The `Op`'s display name.
   :type name: :py:class:`str`
   :param signature: Numpy-like vectorized signature of the random variable.
   :type signature: :py:class:`str`
   :param dtype: The default dtype of the sampled output.  If the value ``"floatX"`` is
                 given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
                 ``None`` (the default), the `dtype` keyword must be set when
                 `RandomVariable.make_node` is called.
   :type dtype: :py:class:`str (optional)`
   :param inplace: Determine whether the underlying rng state is mutated or copied.
   :type inplace: :py:class:`boolean (optional)`


   .. py:method:: rng_fn(rng: numpy.random.Generator, mu: float, sigma: float, epsilon: float, delta: float, size: Optional[Union[int, Tuple[int, Ellipsis]]] = None) -> numpy.typing.NDArray[numpy.float64]
      :classmethod:


      Sample a numeric random variate.



   .. py:attribute:: dtype
      :value: 'floatX'



   .. py:attribute:: name
      :value: 'shashb'



   .. py:attribute:: signature
      :value: '(),(),(),()->()'



.. py:class:: SHASHo

   Bases: :py:obj:`pymc.distributions.Continuous`


   Base class for continuous distributions.

   Add a tensor variable corresponding to a PyMC distribution to the current model.

   Note that all remaining kwargs must be compatible with ``.dist()``

   :param cls: A PyMC distribution.
   :type cls: :py:class:`type`
   :param name: Name for the new model variable.
   :type name: :py:class:`str`
   :param rng: Random number generator to use with the RandomVariable.
   :type rng: *optional*
   :param dims: A tuple of dimension names known to the model. When shape is not provided,
                the shape of dims is used to define the shape of the variable.
   :type dims: :py:class:`tuple`, *optional*
   :param initval: Numeric or symbolic untransformed initial value of matching shape,
                   or one of the following initial value strategies: "support_point", "prior".
                   Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
                   or support_point-based initial values in the transformed space.
   :type initval: *optional*
   :param observed: Observed data to be passed when registering the random variable in the model.
                    When neither shape nor dims is provided, the shape of observed is used to
                    define the shape of the variable.
                    See ``Model.register_rv``.
   :type observed: *optional*
   :param total_size: See ``Model.register_rv``.
   :type total_size: :py:class:`float`, *optional*
   :param transform: See ``Model.register_rv``.
   :type transform: *optional*
   :param \*\*kwargs: Keyword arguments that will be forwarded to ``.dist()`` or the PyTensor RV Op.
                      Most prominently: ``shape`` for ``.dist()`` or ``dtype`` for the Op.

   :returns: **rv** -- The created random variable tensor, registered in the Model.
   :rtype: :py:class:`TensorVariable`


   .. py:method:: dist(mu: pytensor.tensor.TensorLike, sigma: pytensor.tensor.TensorLike, epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, **kwargs: Any) -> Any
      :classmethod:


      Create a tensor variable corresponding to the `cls` distribution.

      :param dist_params: The inputs to the `RandomVariable` `Op`.
      :type dist_params: :py:class:`array-like`
      :param shape: A tuple of sizes for each dimension of the new RV.
      :type shape: :py:class:`int`, :py:class:`tuple`, :py:class:`Variable`, *optional*
      :param \*\*kwargs: Keyword arguments that will be forwarded to the PyTensor RV Op.
                         Most prominently: ``size`` or ``dtype``.

      :returns: **rv** -- The created random variable tensor.
      :rtype: :py:class:`TensorVariable`



   .. py:method:: logp(mu: float, sigma: float, epsilon: float, delta: float) -> float


   .. py:attribute:: rv_op


.. py:class:: SHASHo2

   Bases: :py:obj:`pymc.distributions.Continuous`


   Base class for continuous distributions.

   Add a tensor variable corresponding to a PyMC distribution to the current model.

   Note that all remaining kwargs must be compatible with ``.dist()``

   :param cls: A PyMC distribution.
   :type cls: :py:class:`type`
   :param name: Name for the new model variable.
   :type name: :py:class:`str`
   :param rng: Random number generator to use with the RandomVariable.
   :type rng: *optional*
   :param dims: A tuple of dimension names known to the model. When shape is not provided,
                the shape of dims is used to define the shape of the variable.
   :type dims: :py:class:`tuple`, *optional*
   :param initval: Numeric or symbolic untransformed initial value of matching shape,
                   or one of the following initial value strategies: "support_point", "prior".
                   Depending on the sampler's settings, a random jitter may be added to numeric, symbolic
                   or support_point-based initial values in the transformed space.
   :type initval: *optional*
   :param observed: Observed data to be passed when registering the random variable in the model.
                    When neither shape nor dims is provided, the shape of observed is used to
                    define the shape of the variable.
                    See ``Model.register_rv``.
   :type observed: *optional*
   :param total_size: See ``Model.register_rv``.
   :type total_size: :py:class:`float`, *optional*
   :param transform: See ``Model.register_rv``.
   :type transform: *optional*
   :param \*\*kwargs: Keyword arguments that will be forwarded to ``.dist()`` or the PyTensor RV Op.
                      Most prominently: ``shape`` for ``.dist()`` or ``dtype`` for the Op.

   :returns: **rv** -- The created random variable tensor, registered in the Model.
   :rtype: :py:class:`TensorVariable`


   .. py:method:: dist(mu: pytensor.tensor.TensorLike, sigma: pytensor.tensor.TensorLike, epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, **kwargs: Any) -> Any
      :classmethod:


      Create a tensor variable corresponding to the `cls` distribution.

      :param dist_params: The inputs to the `RandomVariable` `Op`.
      :type dist_params: :py:class:`array-like`
      :param shape: A tuple of sizes for each dimension of the new RV.
      :type shape: :py:class:`int`, :py:class:`tuple`, :py:class:`Variable`, *optional*
      :param \*\*kwargs: Keyword arguments that will be forwarded to the PyTensor RV Op.
                         Most prominently: ``size`` or ``dtype``.

      :returns: **rv** -- The created random variable tensor.
      :rtype: :py:class:`TensorVariable`



   .. py:method:: logp(mu: float, sigma: float, epsilon: float, delta: float) -> float


   .. py:attribute:: rv_op


.. py:class:: SHASHo2RV(name=None, ndim_supp=None, ndims_params=None, dtype: str | None = None, inplace=None, signature: str | None = None)

   Bases: :py:obj:`pytensor.tensor.random.op.RandomVariable`


   An `Op` that produces a sample from a random variable.

   This is essentially `RandomFunction`, except that it removes the
   `outtype` dependency and handles shape dimension information more
   directly.


   Create a random variable `Op`.

   :param name: The `Op`'s display name.
   :type name: :py:class:`str`
   :param signature: Numpy-like vectorized signature of the random variable.
   :type signature: :py:class:`str`
   :param dtype: The default dtype of the sampled output.  If the value ``"floatX"`` is
                 given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
                 ``None`` (the default), the `dtype` keyword must be set when
                 `RandomVariable.make_node` is called.
   :type dtype: :py:class:`str (optional)`
   :param inplace: Determine whether the underlying rng state is mutated or copied.
   :type inplace: :py:class:`boolean (optional)`


   .. py:method:: rng_fn(rng: numpy.random.Generator, mu: pytensor.tensor.TensorLike, sigma: pytensor.tensor.TensorLike, epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, size: Optional[Union[int, Tuple[int, Ellipsis]]] = None) -> numpy.typing.NDArray[numpy.float64]
      :classmethod:


      Sample a numeric random variate.



   .. py:attribute:: dtype
      :value: 'floatX'



   .. py:attribute:: name
      :value: 'shasho2'



   .. py:attribute:: signature
      :value: '(),(),(),()->()'



.. py:class:: SHASHoRV(name=None, ndim_supp=None, ndims_params=None, dtype: str | None = None, inplace=None, signature: str | None = None)

   Bases: :py:obj:`pytensor.tensor.random.op.RandomVariable`


   An `Op` that produces a sample from a random variable.

   This is essentially `RandomFunction`, except that it removes the
   `outtype` dependency and handles shape dimension information more
   directly.


   Create a random variable `Op`.

   :param name: The `Op`'s display name.
   :type name: :py:class:`str`
   :param signature: Numpy-like vectorized signature of the random variable.
   :type signature: :py:class:`str`
   :param dtype: The default dtype of the sampled output.  If the value ``"floatX"`` is
                 given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
                 ``None`` (the default), the `dtype` keyword must be set when
                 `RandomVariable.make_node` is called.
   :type dtype: :py:class:`str (optional)`
   :param inplace: Determine whether the underlying rng state is mutated or copied.
   :type inplace: :py:class:`boolean (optional)`


   .. py:method:: rng_fn(rng: numpy.random.Generator, mu: pytensor.tensor.TensorLike, sigma: pytensor.tensor.TensorLike, epsilon: pytensor.tensor.TensorLike, delta: pytensor.tensor.TensorLike, size: Optional[Union[int, Tuple[int, Ellipsis]]] = None) -> numpy.typing.NDArray[numpy.float64]
      :classmethod:


      Sample a numeric random variate.



   .. py:attribute:: dtype
      :value: 'floatX'



   .. py:attribute:: name
      :value: 'shasho'



   .. py:attribute:: signature
      :value: '(),(),(),()->()'



.. py:class:: SHASHrv(name=None, ndim_supp=None, ndims_params=None, dtype: str | None = None, inplace=None, signature: str | None = None)

   Bases: :py:obj:`pytensor.tensor.random.op.RandomVariable`


   An `Op` that produces a sample from a random variable.

   This is essentially `RandomFunction`, except that it removes the
   `outtype` dependency and handles shape dimension information more
   directly.


   Create a random variable `Op`.

   :param name: The `Op`'s display name.
   :type name: :py:class:`str`
   :param signature: Numpy-like vectorized signature of the random variable.
   :type signature: :py:class:`str`
   :param dtype: The default dtype of the sampled output.  If the value ``"floatX"`` is
                 given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
                 ``None`` (the default), the `dtype` keyword must be set when
                 `RandomVariable.make_node` is called.
   :type dtype: :py:class:`str (optional)`
   :param inplace: Determine whether the underlying rng state is mutated or copied.
   :type inplace: :py:class:`boolean (optional)`


   .. py:method:: rng_fn(rng: numpy.random.Generator, epsilon: float, delta: float, size: Optional[Union[int, Tuple[int, Ellipsis]]] = None) -> numpy.typing.NDArray[numpy.float64]
      :classmethod:


      Sample a numeric random variate.



   .. py:attribute:: dtype
      :value: 'floatX'



   .. py:attribute:: name
      :value: 'shash'



   .. py:attribute:: signature
      :value: '(),()->()'



.. py:function:: K(p: numpy.typing.NDArray[numpy.float64], x: float) -> numpy.typing.NDArray[numpy.float64]

   Bessel function of the second kind for unique values.



.. py:function:: P(q: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

   The P function as given in Jones et al.



.. py:function:: S(x: numpy.typing.NDArray[numpy.float64], e: numpy.typing.NDArray[numpy.float64], d: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

   Sinh arcsinh transformation.



.. py:function:: S_inv(x: numpy.typing.NDArray[numpy.float64], e: numpy.typing.NDArray[numpy.float64], d: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

   Inverse sinh arcsinh transformation.



.. py:function:: m(epsilon: numpy.typing.NDArray[numpy.float64], delta: numpy.typing.NDArray[numpy.float64], r: int) -> numpy.typing.NDArray[numpy.float64]

   The r'th uncentered moment as given in Jones et al.



.. py:data:: CONST1

.. py:data:: CONST2

.. py:data:: kv

.. py:data:: shash

.. py:data:: shashb

.. py:data:: shasho

.. py:data:: shasho2

