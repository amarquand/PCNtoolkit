

import bspline  #type: ignore
import numpy as np
from bspline import splinelab


def create_poly_basis(X: np.ndarray, dimpoly: int) -> np.ndarray:
    """
    Creates a polynomial basis matrix for the given input matrix.

    This function takes an input matrix `X` and a degree `dimpoly`, and returns a new matrix where each column is `X` raised to the power of a degree. The degrees range from 1 to `dimpoly`. If `X` is a 1D array, it is reshaped into a 2D array with one column.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix, a 2D array where each row is a sample and each column is a feature. If `X` is a 1D array, it is reshaped into a 2D array with one column.
    dimpoly : int
        The degree of the polynomial basis. The output matrix will have `dimpoly` times as many columns as `X`.

    Returns
    -------
    Phi : numpy.ndarray
        The polynomial basis matrix, a 2D array where each row is a sample and each column is a feature raised to a degree. The degrees range from 1 to `dimpoly`.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> create_poly_basis(X, 2)
    array([[ 1.,  2.,  1.,  4.],
           [ 3.,  4.,  9., 16.],
           [ 5.,  6., 25., 36.]])
    """
    # TODO: Check this oneliner : Phi = np.power.outer(X.reshape(-1,1), np.arange(1,dimpoly+1)).reshape(X.shape[0], -1)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
    Phi = np.zeros((X.shape[0], D * dimpoly))
    colid = np.arange(0, D)
    for d in range(1, dimpoly + 1):
        Phi[:, colid] = X**d
        colid += D

    return Phi


def create_bspline_basis(
    xmin: float | int, xmax: float | int, p: int = 3, nknots: int = 5
) -> bspline.Bspline:
    """Compute a B-spline basis.

    Parameters
    ----------
    xmin : float | int
        Lower bound of the input domain
    xmax : float | int
        Upper bound of the input domain
    p : int, optional
        Order of the BSpline, by default 3
    nknots : int, optional
        Number of Knots, by default 5

    Returns
    -------
    bspline.Bspline
        The BSpline basis
    """

    knots = np.linspace(xmin, xmax, nknots)
    k = splinelab.augknt(knots, p)  # pad the knot vector
    B = bspline.Bspline(k, p)
    return B
