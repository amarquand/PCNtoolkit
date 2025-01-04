from abc import abstractmethod
from typing import Protocol, runtime_checkable

from pcntoolkit.dataio.norm_data import NormData


@runtime_checkable
class AtomicModel(Protocol):
    """Most basic interface - single atomic operation.
    Can only fit and predict in one go, cannot separate training from prediction.
    Useful for simple cross-validation scenarios."""
    
    def fit_predict(self, data: NormData, test_data: NormData) -> NormData:
        """Single atomic operation to fit model and generate predictions.
        
        Parameters
        ----------
        data : NormData
            Training data containing covariates and responses
        test_data : NormData
            Test data containing covariates for prediction
            
        Returns
        -------
        NormData
            Test data with predictions added
        """
        # Shared implementation
        self._validate_data(data)
        self._validate_data(test_data, training=False)
        
        # Abstract method to be implemented
        return self._fit_predict_implementation(data, test_data)
    
    @abstractmethod 
    def _fit_predict_implementation(self, data: NormData, test_data: NormData) -> NormData:
        """Implementation of fit_predict to be provided by concrete classes"""
        pass
    
    def _validate_data(self, data: NormData, training: bool = True) -> None:
        """Shared data validation logic"""
        if training and not hasattr(data, 'y'):
            raise ValueError("Training data must contain response variables")
        if not hasattr(data, 'X'):
            raise ValueError("Data must contain covariates")

@runtime_checkable
class TrainableModel(Protocol):
    """Can separate training from prediction.
    Maintains state between fit and predict calls."""
    
    def fit(self, data: NormData) -> None:
        """Fit the model to training data"""
        self._validate_data(data)
        self._fit_implementation(data)
        self._is_fitted = True
    
    def predict(self, data: NormData) -> NormData:
        """Make predictions on new data"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before prediction")
        self._validate_data(data, training=False)
        return self._predict_implementation(data)
    
    @abstractmethod
    def _fit_implementation(self, data: NormData) -> None:
        """Implementation of model fitting"""
        pass
    
    @abstractmethod
    def _predict_implementation(self, data: NormData) -> NormData:
        """Implementation of prediction"""
        pass
    
    def _validate_data(self, data: NormData, training: bool = True) -> None:
        """Shared data validation logic"""
        if training and not hasattr(data, 'y'):
            raise ValueError("Training data must contain response variables")
        if not hasattr(data, 'X'):
            raise ValueError("Data must contain covariates")

@runtime_checkable
class StatelessModel(Protocol):
    """Can fit and predict separately, but doesn't maintain state.
    Must pass training data to predict method."""
    
    def fit(self, data: NormData) -> dict:
        """Returns learned parameters instead of storing them"""
        self._validate_data(data)
        return self._fit_implementation(data)
    
    def predict(self, data: NormData, trained_params: dict) -> NormData:
        """Requires parameters from fit"""
        self._validate_data(data, training=False)
        self._validate_params(trained_params)
        return self._predict_implementation(data, trained_params)
    
    @abstractmethod
    def _fit_implementation(self, data: NormData) -> dict:
        """Implementation of parameter learning"""
        pass
    
    @abstractmethod
    def _predict_implementation(self, data: NormData, trained_params: dict) -> NormData:
        """Implementation of prediction using parameters"""
        pass
    
    def _validate_data(self, data: NormData, training: bool = True) -> None:
        """Shared data validation logic"""
        if training and not hasattr(data, 'y'):
            raise ValueError("Training data must contain response variables")
        if not hasattr(data, 'X'):
            raise ValueError("Data must contain covariates")
            
    @abstractmethod
    def _validate_params(self, params: dict) -> None:
        """Validate the trained parameters"""
        pass

@runtime_checkable
class PersistableModel(Protocol):
    """Full featured model that can be saved/loaded"""
    
    def fit(self, data: NormData) -> None:
        """Fit the model to training data"""
        self._validate_data(data)
        self._fit_implementation(data)
        self._is_fitted = True
    
    def predict(self, data: NormData) -> NormData:
        """Make predictions on new data"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before prediction")
        self._validate_data(data, training=False)
        return self._predict_implementation(data)
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        self._save_implementation(path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        self._load_implementation(path)
        self._is_fitted = True
    
    @abstractmethod
    def _fit_implementation(self, data: NormData) -> None:
        """Implementation of model fitting"""
        pass
    
    @abstractmethod
    def _predict_implementation(self, data: NormData) -> NormData:
        """Implementation of prediction"""
        pass
    
    @abstractmethod
    def _save_implementation(self, path: str) -> None:
        """Implementation of model saving"""
        pass
    
    @abstractmethod
    def _load_implementation(self, path: str) -> None:
        """Implementation of model loading"""
        pass
    
    def _validate_data(self, data: NormData, training: bool = True) -> None:
        """Shared data validation logic"""
        if training and not hasattr(data, 'y'):
            raise ValueError("Training data must contain response variables")
        if not hasattr(data, 'X'):
            raise ValueError("Data must contain covariates") 