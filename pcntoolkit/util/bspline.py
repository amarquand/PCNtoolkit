import numpy as np
from scipy.interpolate import BSpline


class BSplineBasis:
    def __init__(
        self, order, nknots, knot_method="uniform", left_expand=0.05, right_expand=0.05
    ):
        """
        Initialize the BSplineBasis object.
        :param order: Degree of the B-spline
        :param nknots: Number of interior knots. Mind that this is the number of interior
        knots. The final number of knots will be nknotes+2 as two knots will be added at boundries.
        :param knot_method: 'uniform' or 'percentile' for knot placement
        :param left_expand: Fraction to expand the range on the left (default 0)
        :param right_expand: Fraction to expand the range on the right (default 0)
        """
        if nknots + 2 < order + 1:
            raise ValueError("Number of knots+2 must be at least degree + 1.")
        if knot_method not in ["uniform", "percentile"]:
            raise ValueError("knot_method must be 'uniform' or 'percentile'.")
        if not (0 <= left_expand <= 1 and 0 <= right_expand <= 1):
            raise ValueError("left_expand and right_expand must be between 0 and 1.")

        self.degree = order
        self.nknots = nknots
        self.knot_method = knot_method
        self.left_expand = left_expand
        self.right_expand = right_expand
        self.knots = None
        self.feature_min = None
        self.feature_max = None

    def fit(self, X, feature_min=None, feature_max=None):
        """
        Fit B-spline basis functions to the dataset.
        :param X: [N×P] array of covariates
        :param feature_min: Minimum values for features (optional)
        :param feature_max: Maximum values for features (optional)
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")

        self.feature_min = (
            np.min(X, axis=0) if feature_min is None else np.array(feature_min)
        )
        self.feature_max = (
            np.max(X, axis=0) if feature_max is None else np.array(feature_max)
        )

        feature_num = X.shape[1]
        self.knots = []

        for i in range(feature_num):
            # Determine range of bspline basis
            minx = self.feature_min[i]
            maxx = self.feature_max[i]
            delta = maxx - minx
            t_min = minx - self.left_expand * delta
            t_max = maxx + self.right_expand * delta

            # Determine knot locations
            if self.knot_method == "uniform":
                interior_knots = np.linspace(t_min, t_max, self.nknots)
            elif self.knot_method == "percentile":
                interior_knots = np.percentile(
                    X[:, i], np.linspace(0, 100, self.nknots)
                )

            # Add boundary knots
            t = np.concatenate(
                ([t_min] * self.degree, interior_knots, [t_max] * self.degree)
            )

            self.knots.append(t)

    def transform(self, X):
        """
        Transform the dataset using the fitted B-spline basis functions.
        :param X: [N×P] array of clinical covariates
        :return: [N×(P×n_basis)] array of transformed data
        """
        if self.knots is None:
            raise ValueError(
                "B-spline basis functions have not been fitted. Call 'fit' first."
            )
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if len(self.knots) != X.shape[1]:
            raise ValueError(
                "Number of B-spline basis functions must match the number of features in X."
            )

        transformed_features = []
        for f in range(len(self.knots)):
            phi = BSpline.design_matrix(
                x=X[:, f], t=self.knots[f], k=self.degree, extrapolate=True
            ).toarray()
            transformed_features.append(phi)
        return np.concatenate(transformed_features, axis=1)

    def adapt(self, target_X):
        """
        Adapt the fitted B-spline basis functions to a target dataset.
        :param target_X: [N×P] array of target clinical covariates
        """
        if self.knots is None:
            raise ValueError(
                "B-spline basis functions have not been fitted. Call 'fit' first."
            )
        if not isinstance(target_X, np.ndarray):
            raise ValueError("Input target_X must be a NumPy array.")
        if target_X.ndim != 2:
            raise ValueError("Input target_X must be a 2D array.")
        if len(self.knots) != target_X.shape[1]:
            raise ValueError(
                "Number of B-spline basis functions must match the number of features in target_X."
            )

        # Updating feature_min and feature_max using combined datsets
        combined_min = np.minimum(self.feature_min, np.min(target_X, axis=0))
        combined_max = np.maximum(self.feature_max, np.max(target_X, axis=0))
        self.feature_min = combined_min
        self.feature_max = combined_max

        feature_num = target_X.shape[1]

        new_knots = []

        for i in range(feature_num):
            minx = self.feature_min[i]
            maxx = self.feature_max[i]
            delta = maxx - minx
            t_min = minx - self.left_expand * delta
            t_max = maxx + self.right_expand * delta

            # Adapt knots
            source_knots = self.knots[i]
            target_knots = t_min + (source_knots - source_knots[0]) * (
                t_max - t_min
            ) / (source_knots[-1] - source_knots[0])

            new_knots.append(target_knots)

        self.knots = new_knots