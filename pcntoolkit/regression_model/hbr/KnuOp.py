# Third-party imports
from typing import Any, List, Sequence, Union

import scipy.special as spp  # type: ignore
from pytensor.gradient import grad_not_implemented
from pytensor.graph.basic import Apply, Variable
from pytensor.scalar.basic import BinaryScalarOp, upgrade_to_float


class KnuOp(BinaryScalarOp):
    """Modified Bessel function of the second kind, PyTensor wrapper for scipy.special.kv.

    This class implements a PyTensor operation for computing the modified Bessel function
    of the second kind (K_nu(x)). It wraps scipy.special.kv to provide automatic
    differentiation capabilities within PyTensor computational graphs.

    Parameters
    ----------
    dtype_converter : callable
        Function to convert input types (typically upgrade_to_float)
    name : str
        Name of the operation

    Notes
    -----
    The modified Bessel function K_nu(x) is a solution to the modified Bessel
    differential equation. This implementation supports automatic differentiation
    with respect to both the order (nu) and the argument (x).

    See Also
    --------
    scipy.special.kv : The underlying modified Bessel function implementation
    KnuPrimeOp : Derivative of the modified Bessel function
    """

    nfunc_spec = ("scipy.special.kv", 2, 1)

    @staticmethod
    def st_impl(p: Union[float, int], x: Union[float, int]) -> float:
        """Static implementation of the modified Bessel function.

        Parameters
        ----------
        p : float or int
            Order of the modified Bessel function
        x : float or int
            Argument where the function is evaluated

        Returns
        -------
        float
            Value of the modified Bessel function K_p(x)
        """
        return spp.kv(p, x)

    def impl(self, p: Union[float, int], x: Union[float, int]) -> float:
        """Implementation of the modified Bessel function.

        Parameters
        ----------
        p : float or int
            Order of the modified Bessel function
        x : float or int
            Argument where the function is evaluated

        Returns
        -------
        float
            Value of the modified Bessel function K_p(x)
        """
        return KnuOp.st_impl(p, x)

    def grad(
        self,
        inputs: Sequence[Variable[Any, Any]],
        output_gradients: Sequence[Variable[Any, Any]],
    ) -> List[Variable]:
        """Compute gradients of the modified Bessel function.

        Parameters
        ----------
        inputs : list of Variables
            List containing the order (p) and argument (x)
        output_grads : list of Variables
            List containing the gradient with respect to the output

        Returns
        -------
        list of Variables
            Gradients with respect to p and x

        Notes
        -----
        The gradient with respect to p is computed using finite differences
        due to the lack of a closed-form expression.
        """
        dp = 1e-16
        (p, x) = inputs
        (gz,) = output_gradients
        # Use finite differences for derivative with respect to p
        dfdp = (knuop(p + dp, x) - knuop(p - dp, x)) / (2 * dp)  # type: ignore
        return [gz * dfdp, gz * knupop(p, x)]  # type: ignore

    def c_code(
        self,
        node: Apply[Any],
        name: str,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        sub: dict[str, str],
    ) -> Any:
        raise NotImplementedError("C code generation not implemented for KnuOp")


class KnuPrimeOp(BinaryScalarOp):
    """Derivative of the modified Bessel function of the second kind.

    This class implements a PyTensor operation for computing the derivative of the
    modified Bessel function of the second kind with respect to its argument.
    It wraps scipy.special.kvp.

    Parameters
    ----------
    dtype_converter : callable
        Function to convert input types (typically upgrade_to_float)
    name : str
        Name of the operation
    """

    nfunc_spec = ("scipy.special.kvp", 2, 1)

    @staticmethod
    def st_impl(p: Union[float, int], x: Union[float, int]) -> float:
        """Static implementation of the Bessel function derivative.

        Parameters
        ----------
        p : float or int
            Order of the modified Bessel function
        x : float or int
            Argument where the derivative is evaluated

        Returns
        -------
        float
            Value of the derivative K'_p(x)
        """
        return spp.kvp(p, x)

    def impl(self, p: Union[float, int], x: Union[float, int]) -> float:
        """Implementation of the Bessel function derivative.

        Parameters
        ----------
        p : float or int
            Order of the modified Bessel function
        x : float or int
            Argument where the derivative is evaluated

        Returns
        -------
        float
            Value of the derivative K'_p(x)
        """
        return KnuPrimeOp.st_impl(p, x)

    def grad(
        self, inputs: Sequence[Variable[Any, Any]], grads: Sequence[Variable[Any, Any]]
    ) -> List[Variable]:
        """Compute gradients of the Bessel function derivative.

        Parameters
        ----------
        inputs : list of Variables
            List containing the order (p) and argument (x)
        grads : list of Variables
            List containing the gradient with respect to the output

        Returns
        -------
        list of Variables
            Gradients with respect to p and x (not implemented)
        """
        return [grad_not_implemented(self, 0, "p"), grad_not_implemented(self, 1, "x")]

    def c_code(
        self,
        node: Apply[Any],
        name: str,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        sub: dict[str, str],
    ) -> Any:
        raise NotImplementedError("C code generation not implemented for KnupOp")


# Create operation instances
knuop = KnuOp(upgrade_to_float, name="knuop")
knupop = KnuPrimeOp(upgrade_to_float, name="knupop")
