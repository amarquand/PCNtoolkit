from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class Scaler:
    """A class for scaling numerical data using various methods.

    Parameters
    ----------
    scaler_type : str, optional
        The type of scaling to apply. Options are:
        - "standardize": zero mean, unit variance
        - "minmax": scale to range [0,1]
        - "robminmax": robust minmax scaling using percentiles
        - "id" or "none": no scaling
        Default is "standardize"
    tail : float, optional
        The tail probability for robust scaling, by default 0.05
    adjust_outliers : bool, optional
        Whether to clip values to [0,1] range for minmax scaling, by default True

    Attributes
    ----------
    scaler_type : str
        The type of scaling being used
    tail : float
        The tail probability for robust scaling
    adjust_outliers : bool
        Whether outliers are adjusted
    """

    def __init__(
        self,
        scaler_type: str = "standardize",
        tail: float = 0.05,
        adjust_outliers: bool = True,
    ) -> None:
        self.scaler_type: str = scaler_type
        self.tail: float = tail
        self.adjust_outliers: bool = adjust_outliers

        if self.scaler_type not in ["standardize", "minmax", "robminmax", "id", "none"]:
            raise ValueError("Undefined scaler type!")

    def fit(self, X: NDArray) -> None:
        """Fit the scaler to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the scaler
        """
        if self.scaler_type == "standardize":
            self.m: NDArray = np.mean(X, axis=0)
            self.s: NDArray = np.std(X, axis=0)

        elif self.scaler_type == "minmax":
            self.min: NDArray = np.min(X, axis=0)
            self.max: NDArray = np.max(X, axis=0)

        elif self.scaler_type == "robminmax":
            self.min = np.zeros(X.shape[1])
            self.max = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                self.min[i] = np.median(
                    np.sort(X[:, i])[0 : int(np.round(X.shape[0] * self.tail))]
                )
                self.max[i] = np.median(
                    np.sort(X[:, i])[-int(np.round(X.shape[0] * self.tail)) :]
                )

        elif self.scaler_type in ["id", "none"]:
            pass

    def transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        """Transform the data using the fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            Data to transform
        index : np.ndarray, optional
            Indices for partial transformation, by default None

        Returns
        -------
        np.ndarray
            Transformed data
        """
        if self.scaler_type == "standardize":
            if index is None:
                X = (X - self.m) / self.s
            else:
                X = (X - self.m[index]) / self.s[index]

        elif self.scaler_type in ["minmax", "robminmax"]:
            if index is None:
                X = (X - self.min) / (self.max - self.min)
            else:
                X = (X - self.min[index]) / (self.max[index] - self.min[index])

            if self.adjust_outliers:
                X[X < 0] = 0
                X[X > 1] = 1

        elif self.scaler_type in ["id", "none"]:
            X = X + 0

        return X

    def inverse_transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        """Inverse transform scaled data back to original scale.

        Parameters
        ----------
        X : np.ndarray
            Scaled data to inverse transform
        index : np.ndarray, optional
            Indices for partial transformation, by default None

        Returns
        -------
        np.ndarray
            Inverse transformed data
        """
        if self.scaler_type == "standardize":
            if index is None:
                X = X * self.s + self.m
            else:
                X = X * self.s[index] + self.m[index]

        elif self.scaler_type in ["minmax", "robminmax"]:
            if index is None:
                X = X * (self.max - self.min) + self.min
            else:
                X = X * (self.max[index] - self.min[index]) + self.min[index]

        elif self.scaler_type in ["id", "none"]:
            X = X + 0

        return X

    def fit_transform(self, X: NDArray) -> NDArray:
        """Fit the scaler and transform the data in one step.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit and transform

        Returns
        -------
        np.ndarray
            Transformed data
        """
        if self.scaler_type == "standardize":
            self.m = np.mean(X, axis=0)
            self.s = np.std(X, axis=0)
            X = (X - self.m) / self.s

        elif self.scaler_type == "minmax":
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            X = (X - self.min) / (self.max - self.min)

        elif self.scaler_type == "robminmax":
            reshape = False
            if len(X.shape) == 1:
                reshape = True
                X = X.reshape(-1, 1)

            self.min = np.zeros(X.shape[1])
            self.max = np.zeros(X.shape[1])

            for i in range(X.shape[1]):
                self.min[i] = np.median(
                    np.sort(X[:, i])[0 : int(np.round(X.shape[0] * self.tail))]
                )
                self.max[i] = np.median(
                    np.sort(X[:, i])[-int(np.round(X.shape[0] * self.tail)) :]
                )

            X = (X - self.min) / (self.max - self.min)

            if self.adjust_outliers:
                X[X < 0] = 0
                X[X > 1] = 1

            if reshape:
                X = X.reshape(-1)

        elif self.scaler_type in ["id", "none"]:
            X = X + 0

        return X

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        """Convert scaler instance to dictionary.

        Returns
        -------
        Dict[str, Union[bool, str, float, List[float]]]
            Dictionary containing the scaler parameters
        """
        my_dict = {
            "adjust_outliers": self.adjust_outliers,
            "scaler_type": self.scaler_type,
            "tail": self.tail,
        }

        if self.scaler_type == "standardize":
            return my_dict | {"m": self.m.tolist(), "s": self.s.tolist()}
        elif self.scaler_type in ["minmax", "robinmax"]:
            return my_dict | {"min": self.min.tolist(), "max": self.max.tolist()}
        else:
            return my_dict

    @classmethod
    def from_dict(
        cls, my_dict: Dict[str, Union[bool, str, float, List[float]]]
    ) -> "Scaler":
        """Create a scaler instance from a dictionary.

        Parameters
        ----------
        my_dict : Dict[str, Union[bool, str, float, List[float]]]
            Dictionary containing scaler parameters

        Returns
        -------
        Scaler
            New scaler instance with the specified parameters
        """
        scaler_type = my_dict.pop("scaler_type")
        adjust_outliers = my_dict.pop("adjust_outliers")
        tail = my_dict.pop("tail")

        instance = cls(
            scaler_type=scaler_type, tail=tail, adjust_outliers=adjust_outliers
        )
        if scaler_type == "standardize":
            m = np.array(my_dict.pop("m"))
            s = np.array(my_dict.pop("s"))
            instance.m = m
            instance.s = s
        elif scaler_type in ["minmax", "robinmax"]:
            min = np.array(my_dict.pop("min"))
            max = np.array(my_dict.pop("max"))
            instance.min = min
            instance.max = max
        return instance
