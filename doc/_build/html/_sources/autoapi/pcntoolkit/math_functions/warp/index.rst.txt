pcntoolkit.math_functions.warp
==============================

.. py:module:: pcntoolkit.math_functions.warp


Classes
-------

.. autoapisummary::

   pcntoolkit.math_functions.warp.WarpAffine
   pcntoolkit.math_functions.warp.WarpBase
   pcntoolkit.math_functions.warp.WarpBoxCox
   pcntoolkit.math_functions.warp.WarpCompose
   pcntoolkit.math_functions.warp.WarpLog
   pcntoolkit.math_functions.warp.WarpSinhArcsinh


Functions
---------

.. autoapisummary::

   pcntoolkit.math_functions.warp.parseWarpString


Module Contents
---------------

.. py:class:: WarpAffine

   Bases: :py:obj:`WarpBase`


   Affine warping function.

   Implements affine transformation y = a + b*x where:
       - a: offset parameter
       - b: scale parameter (constrained positive through exp transform)


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute derivative of affine warp.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Affine parameters [a, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: Constant derivative b
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply affine warping.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Affine parameters [a, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: a + b*x
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply inverse affine warping.

      :param y: Input values
      :type y: :py:class:`NDArray[np.float64]`
      :param params: Affine parameters [a, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: (y - a)/b
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:attribute:: n_params
      :value: 2



.. py:class:: WarpBase

   Bases: :py:obj:`abc.ABC`


   Base class for likelihood warping functions.

   This class implements warping functions following Rios and Torab (2019)
   Compositionally-warped Gaussian processes [1]_.

   All Warps must define the following methods:
       - get_n_params(): Return number of parameters
       - f(): Warping function (Non-Gaussian field -> Gaussian)
       - invf(): Inverse warp
       - df(): Derivatives
       - warp_predictions(): Compute predictive distribution

   .. rubric:: References

   .. [1] Rios, G., & Tobar, F. (2019). Compositionally-warped Gaussian processes.
          Neural Networks, 118, 235-246.
          https://www.sciencedirect.com/science/article/pii/S0893608019301856


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]
      :abstractmethod:


      Return the derivative of the warp, dw(x)/dx.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param param: Warping parameters
      :type param: :py:class:`NDArray[np.float64]`

      :returns: Derivative values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]
      :abstractmethod:


      Evaluate the warping function.

      Maps non-Gaussian response variables to Gaussian variables.

      :param x: Input values to warp
      :type x: :py:class:`NDArray[np.float64]`
      :param param: Warping parameters
      :type param: :py:class:`NDArray[np.float64]`

      :returns: Warped values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: get_n_params() -> int

      Return the number of parameters required by the warping function.

      :returns: Number of parameters
      :rtype: :py:class:`int`

      :raises AssertionError: If warp function is not initialized



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]
      :abstractmethod:


      Evaluate the inverse warping function.

      Maps Gaussian latent variables to non-Gaussian response variables.

      :param y: Input values to inverse warp
      :type y: :py:class:`NDArray[np.float64]`
      :param param: Warping parameters
      :type param: :py:class:`NDArray[np.float64]`

      :returns: Inverse warped values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: warp_predictions(mu: numpy.typing.NDArray[numpy.float64], s2: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64], percentiles: List[float] | None = None) -> Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]

      Compute warped predictions from a Gaussian predictive distribution.

      :param mu: Gaussian predictive mean
      :type mu: :py:class:`NDArray[np.float64]`
      :param s2: Predictive variance
      :type s2: :py:class:`NDArray[np.float64]`
      :param param: Warping parameters
      :type param: :py:class:`NDArray[np.float64]`
      :param percentiles: Desired percentiles of the warped likelihood, by default [0.025, 0.975]
      :type percentiles: :py:class:`List[float]`, *optional*

      :returns:

                - median: Median of the predictive distribution
                - pred_interval: Predictive interval(s)
      :rtype: :py:class:`Tuple[NDArray[np.float64]`, :py:class:`NDArray[np.float64]]`



   .. py:attribute:: n_params
      :type:  int
      :value: 0



.. py:class:: WarpBoxCox

   Bases: :py:obj:`WarpBase`


   Box-Cox warping function.

   Implements the Box-Cox transform with a single parameter (lambda):
   y = (sign(x) * abs(x) ** lambda - 1) / lambda

   This follows the generalization in Bicken and Doksum (1981) JASA 76
   and allows x to assume negative values.

   .. rubric:: References

   .. [1] Bickel, P. J., & Doksum, K. A. (1981). An analysis of transformations
          revisited. Journal of the American Statistical Association, 76(374), 296-311.


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute derivative of Box-Cox warp.

      :param x: Input values
                params : NDArray[np.float64]
                Box-Cox parameter [log(lambda)]
      :type x: :py:class:`NDArray[np.float64]`

      :returns: Derivative values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply Box-Cox warping.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Box-Cox parameter [log(lambda)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: Warped values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply inverse Box-Cox warping.

      :param y: Input values
      :type y: :py:class:`NDArray[np.float64]`
      :param params: Box-Cox parameter [log(lambda)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: Inverse warped values
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:attribute:: n_params
      :value: 1



.. py:class:: WarpCompose(warps: List[WarpBase])

   Bases: :py:obj:`WarpBase`


   Composition of multiple warping functions.

   Allows chaining multiple warps together. Warps are applied in sequence:
   y = warp_n(...warp_2(warp_1(x)))

   Initialize composed warp.

   :param warpnames: List of warp class names to compose, by default None
   :type warpnames: :py:class:`Optional[List[str]]`, *optional*

   :raises ValueError: If warpnames is None


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute derivative of composed warping functions.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param param: Combined parameters for all warps
      :type param: :py:class:`NDArray[np.float64]`

      :rtype: :py:class:`Determinant` of :py:class:`the Jacobian` of :py:class:`the composed warping functions`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply composed warping functions.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param param: Combined parameters for all warps
      :type param: :py:class:`NDArray[np.float64]`

      :returns: Warped values after applying all transforms
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply inverse composed warping functions.

      :param y: Input values
      :type y: :py:class:`NDArray[np.float64]`
      :param param: Combined parameters for all warps
      :type param: :py:class:`NDArray[np.float64]`

      :returns: Inverse warped values after applying all inverse transforms
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:attribute:: n_params
      :value: 0



   .. py:attribute:: warps


.. py:class:: WarpLog

   Bases: :py:obj:`WarpBase`


   Logarithmic warping function.

   Implements y = log(x) warping transformation.


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute derivative of logarithmic warp.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Not used for logarithmic warp
      :type params: :py:class:`NDArray[np.float64]`

      :returns: 1/x
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: Optional[numpy.typing.NDArray[numpy.float64]] = None) -> numpy.typing.NDArray[numpy.float64]

      Apply logarithmic warping.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Not used for logarithmic warp, by default None
      :type params: :py:class:`Optional[NDArray[np.float64]]`, *optional*

      :returns: log(x)
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: Optional[numpy.typing.NDArray[numpy.float64]] = None) -> numpy.typing.NDArray[numpy.float64]

      Apply inverse logarithmic warping.

      :param y: Input values
      :type y: :py:class:`NDArray[np.float64]`
      :param params: Not used for logarithmic warp, by default None
      :type params: :py:class:`Optional[NDArray[np.float64]]`, *optional*

      :returns: exp(y)
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:attribute:: n_params
      :value: 0



.. py:class:: WarpSinhArcsinh

   Bases: :py:obj:`WarpBase`


   Sin-hyperbolic arcsin warping function.

   Implements warping function y = sinh(b * arcsinh(x) - a) with two parameters:
       - a: controls skew
       - b: controls kurtosis (constrained positive through exp transform)

   Properties:
       - a = 0: symmetric
       - a > 0: positive skew
       - a < 0: negative skew
       - b = 1: mesokurtic
       - b > 1: leptokurtic
       - b < 1: platykurtic

   Uses alternative parameterization from Jones and Pewsey (2019) where:
   y = sinh(b * arcsinh(x) + epsilon * b) and a = -epsilon*b

   .. rubric:: References

   .. [1] Jones, M. C., & Pewsey, A. (2019). Sigmoid-type distributions:
          Generation and inference. Significance, 16(1), 12-15.
   .. [2] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions.
          Biometrika, 96(4), 761-780.


   .. py:method:: df(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Compute derivative of sinh-arcsinh warp.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Parameters [epsilon, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: (b * cosh(b * arcsinh(x) - a)) / sqrt(1 + x^2)
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: f(x: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply sinh-arcsinh warping.

      :param x: Input values
      :type x: :py:class:`NDArray[np.float64]`
      :param params: Parameters [epsilon, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: sinh(b * arcsinh(x) - a)
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:method:: invf(y: numpy.typing.NDArray[numpy.float64], param: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

      Apply inverse sinh-arcsinh warping.

      :param y: Input values
      :type y: :py:class:`NDArray[np.float64]`
      :param params: Parameters [epsilon, log(b)]
      :type params: :py:class:`NDArray[np.float64]`

      :returns: sinh((arcsinh(y) + a) / b)
      :rtype: :py:class:`NDArray[np.float64]`



   .. py:attribute:: n_params
      :value: 2



.. py:function:: parseWarpString(warp_string: str) -> WarpBase

   Parse a string into a WarpBase object.

   :param warp_string: String of a warp name or a composition of warps
   :type warp_string: :py:class:`str`


