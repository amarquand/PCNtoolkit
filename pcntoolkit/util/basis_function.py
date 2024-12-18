from abc import ABC, abstractmethod

from pcntoolkit.dataio.norm_data import NormData

def create_basis_function

class BasisFunction(ABC):
    @abstractmethod
    def fit(self, data: NormData) -> None:
        pass
    
    @abstractmethod
    def transform(self, data: NormData) -> None:
        pass

    @abstractmethod
    def transfer(self, data: NormData) -> None:
        pass

class PolynomialBasisFunction(BasisFunction):

    def __init__(self, degree: int):
        self.degree = degree

    def fit(self, data: NormData) -> None:
        self.basis_function = create_poly_basis(data.X, self.degree)
    
    def transform(self, data: NormData) -> None:
        pass
