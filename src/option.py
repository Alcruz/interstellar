from abc import ABC, abstractmethod
import numpy as np

class AmericanOption(ABC):
    def __init__(self, K: float, T: float):
        self.K = K # Strike price
        self.T = T # Maturity

    @abstractmethod
    def payoff(self, S: float):
        pass

class CallAmericanOption(AmericanOption):
    def payoff(self, S: float):
        return np.maximum(S - self.K, 0)
    
class PutAmericanOption(AmericanOption):
    def payoff(self, S: float):
        return np.maximum(self.K - S, 0)
