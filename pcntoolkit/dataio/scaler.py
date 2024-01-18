import numpy as np


class scaler:

    def __init__(self, scaler_type='standardize', tail=0.05,
                 adjust_outliers=True):
        """ 
        A class for rescaling data using either standardization or minmax 
        normalization.

        :param scaler_type: String that decides the type of scaler including 
            1) 'standardize' for standardizing data, 2) 'minmax' for minmax normalization
            in range of [0,1], and 3) 'robminmax' for robust (to outliers) minmax 
            normalization.The default is 'standardize'.
        :param tail: Is a decimal in range [0,1] that decides the tails of
            distribution for finding robust min and max in 'robminmax' 
            normalization. The defualt is 0.05.
        :param adjust_outliers: Boolean that decides whether to adjust the 
            outliers in 'robminmax' normalization or not. If True the outliers 
            values are truncated to 0 or 1. The defauls is True.

        """

        self.scaler_type = scaler_type
        self.tail = tail
        self.adjust_outliers = adjust_outliers

        if self.scaler_type not in ['standardize', 'minmax', 'robminmax', 'id', 'none']:
            raise ValueError("Undifined scaler type!")

    def fit(self, X):

        if self.scaler_type == 'standardize':

            self.m = np.mean(X, axis=0)
            self.s = np.std(X, axis=0)

        elif self.scaler_type == 'minmax':
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)

        elif self.scaler_type == 'robminmax':
            self.min = np.zeros(X.shape[1])
            self.max = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                self.min[i] = np.median(
                    np.sort(X[:, i])[0:int(np.round(X.shape[0] * self.tail))])
                self.max[i] = np.median(
                    np.sort(X[:, i])[-int(np.round(X.shape[0] * self.tail)):])

        elif self.scaler_type in ['id', 'none']:
            pass

    def transform(self, X, index=None):

        if self.scaler_type == 'standardize':
            if index is None:
                X = (X - self.m) / self.s
            else:
                X = (X - self.m[index]) / self.s[index]

        elif self.scaler_type in ['minmax', 'robminmax']:
            if index is None:
                X = (X - self.min) / (self.max - self.min)
            else:
                X = (X - self.min[index]) / (self.max[index] - self.min[index])

            if self.adjust_outliers:

                X[X < 0] = 0
                X[X > 1] = 1

        elif self.scaler_type in ['id', 'none']:
            pass

        return X

    def inverse_transform(self, X, index=None):

        if self.scaler_type == 'standardize':
            if index is None:
                X = X * self.s + self.m
            else:
                X = X * self.s[index] + self.m[index]

        elif self.scaler_type in ['minmax', 'robminmax']:
            if index is None:
                X = X * (self.max - self.min) + self.min
            else:
                X = X * (self.max[index] - self.min[index]) + self.min[index]

        elif self.scaler_type in ['id', 'none']:
            pass
        return X

    def fit_transform(self, X):

        if self.scaler_type == 'standardize':

            self.m = np.mean(X, axis=0)
            self.s = np.std(X, axis=0)
            X = (X - self.m) / self.s

        elif self.scaler_type == 'minmax':

            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            X = (X - self.min) / (self.max - self.min)

        elif self.scaler_type == 'robminmax':

            reshape = False
            if len(X.shape) == 1:
                reshape = True
                X = X.reshape(-1, 1)

            self.min = np.zeros(X.shape[1])
            self.max = np.zeros(X.shape[1])

            for i in range(X.shape[1]):
                self.min[i] = np.median(
                    np.sort(X[:, i])[0:int(np.round(X.shape[0] * self.tail))])
                self.max[i] = np.median(
                    np.sort(X[:, i])[-int(np.round(X.shape[0] * self.tail)):])

            X = (X - self.min) / (self.max - self.min)

            if self.adjust_outliers:
                X[X < 0] = 0
                X[X > 1] = 1

            if reshape:
                X = X.reshape(-1)

        elif self.scaler_type in ['id', 'none']:
            pass

        return X
