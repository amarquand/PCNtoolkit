from abc import ABC, abstractmethod

from pcntoolkit.dataio.norm_data import NormData
from .norm_conf import NormConf



class NormBase(ABC):  # newer abstract base class syntax, no more python2

    def __init__(self, norm_conf: NormConf):
        self._norm_conf:NormConf = norm_conf

    def fit(self, data: NormData):
        """
        Contains all the general fitting logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _fit method.
        """

        # some preparations and preprocessing
        # ...

        self._fit(data)

        # some cleanup and postprocessing
        # ...
            

    def predict(self, data: NormData) -> NormData:
        """
        Contains all the general prediction logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _predict method.
        """

        # some preparations and preprocessing
        # ...

        result = self._predict(data)

        # some cleanup and postprocessing
        # ...
            
        return result
    

    def fit_predict(self, data: NormData) -> NormData:
        """
        Contains all the general fit_predict logic that is not specific to the regression model.
        This includes cv, logging, saving, etc. Calls the subclass' _fit_predict method.
        """

        # some preparations and preprocessing
        # ...

        result = self._fit_predict(data)

        # some cleanup and postprocessing
        # ...
            
        return result
    
    
    def transfer(self, data: NormData) -> 'NormBase':
        """
        Transfers the normative model to a new dataset. Calls the subclass' _transfer method.
        """
        # some preparations and preprocessing
        # ...
        
        result = self._transfer(data)

        # some cleanup and postprocessing
        # ...
            
        return result


    def extend(self, data: NormData):
        """
        Extends the normative model with new data. Calls the subclass' _extend method.
        """
        # some preparations and preprocessing
        # ...

        result = self._extend(data)

        # some cleanup and postprocessing
        # ...
            
        return result
    
    
    def tune(self, data: NormData):
        """
        Tunes the normative model. Calls the subclass' _tune method.
        """
        # some preparations and preprocessing
        # ...

        result = self._tune(data)
        
        # some cleanup and postprocessing
        # ...
            
        return result
    
    
    def merge(self, other:'NormBase'):
        """
        Merges the normative model with another normative model. Calls the subclass' _merge method.
        """
        # some preparations and preprocessing
        # ...

        if not self.__class__ == other.__class__:
            raise ValueError('Attempted to merge two different normative models.')

        result = self._merge(other)

        # some cleanup and postprocessing
        # ...
            
        return result


    def evaluate(self, data: NormData):
        """
        Contains evaluation logic.
        """
        results = {}
        results['MSE'] = self.evaluate_mse(data)
        results['MAE'] = self.evaluate_mae(data)
        results['R2'] = self.evaluate_r2(data)
        # Add more metrics here, and add the corresponding abstract methods below.
        return results
    

    @abstractmethod
    def _fit_predict(self, data: NormData):
        """
        Acts as the adapter for fit_predict using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _fit(self, data: NormData) -> NormData:
        """
        Acts as the adapter for fitting the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass

    @abstractmethod
    def _predict(self, data: NormData) -> NormData:
        """
        Acts as the adapter for prediction using the specific regression model.
        Is not responsible for cv, logging, saving, etc.
        """
        pass


    @abstractmethod
    def _transfer(self, data: NormData) -> 'NormBase':
        """
        Transfers the normative model to a new dataset.
        """
        pass


    @abstractmethod
    def _extend(self, data: NormData):
        """
        Extends the normative model with new data.
        """
        pass

    @abstractmethod
    def _tune(self, data: NormData):
        """
        Tunes the normative model.
        """
        pass

    @abstractmethod
    def _merge(self, other:'NormBase'):
        """
        Merges the normative model with another normative model.
        """
        pass

    @abstractmethod
    def evaluate_mse(self, data: NormData) -> float:
        """
        Evaluates the model using MSE.
        """
        pass

    @abstractmethod
    def evaluate_mae(self, data: NormData) -> float:
        """
        Evaluates the model using MAE.
        """
        pass

    @abstractmethod
    def evaluate_r2(self, data: NormData) -> float:
        """
        Evaluates the model using R2.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Saves the model to the specified directory.
        """
        pass
    
    @abstractmethod
    def load(self) -> 'NormBase':
        """
        Loads the model from the specified directory.
        """
        pass
