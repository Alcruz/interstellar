from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = 1,
    PUT = 2

class Option(ABC):
    def __init__(self, type: OptionType, K: float, T: float):
        self.type = type
        self.K = K # Strike price
        self.T = T # Maturity

    @staticmethod
    def call(K: float, T: float):
        return CallOption(K, T)

    @staticmethod
    def put(K: float, T: float):
        return PutOption(K, T)
    
    @abstractmethod
    def payoff(self, S: float):
        pass

class CallOption(Option):
    def __init__(self, K: float, T: float):
        super().__init__(OptionType.CALL, K, T)

    def payoff(self, S: float):
        return np.maximum(S - self.K, 0)
    
class PutOption(Option):
    def __init__(self, K: float, T: float):
        super().__init__(OptionType.PUT, K, T)

    def payoff(self, S: float):
        return np.maximum(self.K - S, 0)
