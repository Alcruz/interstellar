from abc import ABC, abstractmethod
import datetime

import numpy as np

from scipy import interpolate
from scipy.sparse import diags
from scipy.optimize import root

from option import OptionType, Option, PutOption, CallOption

class Solver(ABC): 
    def __init__(self, 
        option: Option,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta # dividends
    ) -> None:
        self.option = option
        self.r = r
        self.sigma = sigma
        self.dx = dx
        self.dt = dt
        self.delta = delta

class ExplicitSolver(ABC):
    def __init__(self, 
        option: Option,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta # dividends
    ) -> None:
        self.option = option
        self.r = r
        self.sigma = sigma
        self.dx = dx
        self.dt = dt
        self.delta = delta
        
        lambd = dt / np.power(dx, 2)
        alpha = dt / dx

        self.A = 0.5 * np.power(sigma*self.x_axis, 2) * lambd - 0.5 * self.x_axis * ((r-delta) - (1/dt)) * alpha
        self.B = 1 - np.power(sigma*self.x_axis, 2) * lambd - r*dt
        self.C = 0.5 * np.power(sigma*self.x_axis, 2) * lambd + 0.5 * self.x_axis * ((r-delta) - (1/dt)) * alpha

    def solve(self):
        """Solve front-fixing method explicitly.

        Parameters:
            option (AmericanOption): the option to price.
            r (float): the stock price risk-free interest rate.
            sigma (float): the stock price volatility.
            x_max (float): large value used as infity in the spatial.
            dx (float): Grid resolution in the spatial direction.
            dt (float): Grid resolution in the time direction.
        """
        V = np.zeros_like(self.x_axis)
        S_bar = self.option.K
        for _ in np.arange(0, self.option.T, self.dt):
            D = 0.5*self.x_axis/self.dx
            D[1:-1] *= (V[2:]-V[:-2]) * (1/S_bar)
            
            S_bar = self.compute_time_iteration(V, D)

        # S_region =  np.linspace(0, S_bar, num=200, endpoint=False)
        # C_region =  S_bar * x
        # P = option.payoff(S_region)
        # V_interp = interpolate.interp1d(
        #     np.concatenate([S_region, C_region]), 
        #     np.concatenate([P, V]), 
        #     kind='cubic'
        # )
        return S_bar*self.x_axis, V, S_bar
    
    @abstractmethod
    def compute_time_iteration(self, V: np.ndarray, D: np.ndarray):
        pass

class CallOptionExplicitSolver(ExplicitSolver):
    def __init__(self, 
        option: CallOption,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta=0  # dividends
    ):
        self.x_axis = np.arange(0, 1+dx, dx)
        super().__init__(option, r, sigma, dx, dt, delta)


    def compute_time_iteration(self, V: np.ndarray, D: np.ndarray):
        S_bar = self.option.K + self.A[-2]*V[-3] + self.B[-2]*V[-2] + self.C[-2]*V[-1]
        S_bar /= (1 - self.dx) - D[-2]

        V[1:-2] = self.A[1:-2]*V[:-3] + self.B[1:-2]*V[1:-2] \
            + self.C[1:-2]*V[2:-1] + D[1:-2]*S_bar
        V[-2] = (1-self.dx)*S_bar - self.option.K
        V[-1] = S_bar - self.option.K
        return S_bar

class PutOptionExplicitSolver(ExplicitSolver):
    def __init__(self, 
        option: PutOption,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        x_max: 2.,  # sufficiently large value for x
        delta=0  # dividends
    ):
        self.x_axis = np.arange(1, x_max+dx, dx)
        super().__init__(option, r, sigma, dx, dt, delta)

    def compute_time_iteration(self, V: np.ndarray, D: np.ndarray) -> float:
        S_bar = self.option.K - (self.A[1]*V[0] + self.B[1]*V[1] + self.C[1]*V[2])
        S_bar /= D[1] + 1 + self.dx

        V[2:-1] = self.A[2:-1]*V[1:-2] + self.B[2:-1] * \
            V[2:-1] + self.C[2:-1]*V[3:] + D[2:-1]*S_bar
        V[0] = self.option.payoff(S_bar)
        V[1] = self.option.payoff((1+self.dx)*S_bar)
        return S_bar

class ImplicitSolver(Solver):
    def __init__(self, 
        option: Option,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta=0, # dividends
    ) -> None:
        self.option = option
        self.r = r
        self.sigma = sigma
        self.dx = dx
        self.dt = dt
        self.delta = delta
        self.lambd = self.dt/np.power(self.dx, 2)
        self.kappa = self.dt/self.dx
        self.alpha = 1 + self.lambd*np.power(self.sigma*self.x_axis, 2) + self.r*self.dt
        self.M = self.x_axis.size

        
    def beta(self, S, S_bar):
        return -0.5*self.lambd*np.power(self.sigma, 2)*np.power(self.x_axis, 2) + 0.5*self.kappa*self.x_axis*((self.r-self.delta) - (S_bar - S)/(self.dt*S))

    def gamma(self, S, S_bar):
        return -0.5*self.lambd*np.power(self.sigma, 2)*np.power(self.x_axis, 2) - 0.5*self.kappa*self.x_axis*((self.r-self.delta) - (S_bar - S)/(self.dt*S))

    @abstractmethod
    def solve_non_linear_system(self, V, S_bar) -> float:
        pass
    
    def solve(self):
        """Solve front-fixing method explicitly.

        Parameters:
            option (AmericanOption): the option to price.
            r (float): the stock price risk-free interest rate.
            sigma (float): the stock price volatility.
            x_max (float): large value used as infity in the spatial.
            N (float): Grid resolution in the spatial direction.
            M (float): Grid resolution in the time direction.
        """
        starting_time = datetime.datetime.now()
        print("Start time:", starting_time)

        S_bar = self.option.K
        V = np.zeros_like(self.x_axis)

        for _ in np.arange(0, self.option.T, self.dt):
            S_bar = self.solve_non_linear_system(V, S_bar)

        end = datetime.datetime.now()
        print("Final time:", end)
        
        # S_region =  np.linspace(0, S_bar, num=200, endpoint=False)
        # C_region =  S_bar * x
        # P = option.payoff(S_region)
        # V_interp = interpolate.interp1d(
        #     np.concatenate([S_region, C_region]), 
        #     np.concatenate([P, V]), 
        #     kind='cubic'
        # )
        return S_bar*self.x_axis, V, S_bar

class CallOptionImplicitSolver(ImplicitSolver):
    def __init__(self, 
        option: Option,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta=0, # dividends
    ) -> None:
        self.x_axis = np.arange(0, 1+dx, dx)
        super().__init__(option, r, sigma, dx, dt, delta)

    def jacobian(self, y, S_bar):
        p, s, = y[:-1], y[-1]
        beta = self.beta(s, S_bar)
        gamma = self.gamma(s, S_bar)

        dgamma_ds = - 0.5 * (1/self.dx) * self.x_axis * S_bar / s**2
        dbeta_ds = 0.5 * (1/self.dx) * self.x_axis * S_bar / s**2

        print(self.M)
        retVal = diags([self.alpha[2:], beta[1:-1], gamma[1:-1]],
                    [-1, 0, 1], shape=(self.M-2, self.M-2)).toarray()

        retVal[-1, :] = 0
        retVal[:, -1] = 0

        retVal[0, -1] = dbeta_ds[1]*p[0] + dgamma_ds[1]*p[1]
        retVal[1:-2, -1] = dbeta_ds[2:-3]*p[1:-1] + dgamma_ds[2:-3]*p[2:]
        retVal[-2, -1] = dbeta_ds[-3]*p[-2]
        
        retVal[-2, -1] -= dgamma_ds[-3]*self.option.payoff((1-self.dx)*s) + gamma[-3]*(1 - self.dx)
        retVal[-1, -2] = self.alpha[-2]
        print(retVal)

        retVal[-1, -1] -= dgamma_ds[-2]*self.option.payoff(s) + gamma[-2] + dbeta_ds[-2]*self.option.payoff((1-self.dx)*s) - beta[1]*(1-self.dx)
        
        return retVal

    def system(self, y, b, S_bar):
        p, s, = y[:-1], y[-1]
        beta = self.beta(s, S_bar)
        gamma = self.gamma(s, S_bar)
        A = diags([self.alpha[2:-1], beta[1:-2], gamma[1:-3]],
                [-1, 0, 1], shape=(self.M-2, self.M-3)).toarray()
        f = b[1:-1]
        f[-2] -= gamma[-3]*self.option.payoff((1-self.dx)*s)
        f[-1] -= gamma[-2]*self.option.payoff(s) + beta[-2]*self.option.payoff((1-self.dx)*s)
        res = A@p - f
        return res
    
    def solve_non_linear_system(self, V: np.ndarray, S_bar: np.ndarray):
        *V[1:-2], S_bar = root(
            lambda y: self.system(y, np.copy(V[:]), S_bar),
            # jac=lambda y: self.jacobian(y, S_bar),
            x0=np.concatenate([V[1:-2], [S_bar]])
        )['x']
        V[-2] = self.option.payoff((1-self.dx)*S_bar)    
        V[-1] = self.option.payoff(S_bar)
        return S_bar

class PutOptionImplicitSolver(ImplicitSolver):
    def __init__(self, 
        option: Option,
        r: float,  # risk-free interest rate
        sigma: float,  # sigma price volatitliy
        dx: float,  # grid resolution along x-axis
        dt: float,  # grid resolution along t-axis
        delta=0, # dividends
        x_max=2
    ) -> None:
        self.x_axis = np.arange(1, x_max+dx, dx)
        super().__init__(option, r, sigma, dx, dt, delta)
    
    def jacobian(self, y, S_bar):
        p, s, = y[:-1], y[-1]
        beta = self.beta(s, S_bar)
        gamma = self.gamma(s, S_bar)

        dgamma_ds = - 0.5 * (1/self.dx) * self.x_axis * S_bar / s**2
        dbeta_ds = 0.5 * (1/self.dx) * self.x_axis * S_bar / s**2

        retVal = diags([beta[3:-1], self.alpha[2:-1], gamma[1:-1]],
                    [-2, -1, 0], shape=(self.M-2, self.M-2)).toarray()

        retVal[-1, :] = 0
        retVal[:, -1] = 0

        retVal[0, -1] = dgamma_ds[1]*p[0] + dbeta_ds[1] * \
            (self.option.K - S_bar) - beta[1] - self.alpha[1]*(1+self.dx)
        retVal[1, -1] = dgamma_ds[2]*p[1] + dbeta_ds[2] * \
            (self.option.K - (1+self.dx)*S_bar) - beta[2]*(1+self.dx)
        retVal[2:-1, -1] = dbeta_ds[3:-2]*p[:-2] + dgamma_ds[3:-2]*p[2:]

        retVal[-1, -3] = beta[-2]
        retVal[-1, -2] = self.alpha[-2]
        retVal[-1, -1] = dbeta_ds[-2]*p[-2]

        return retVal

    def system(self, y, b, S_bar):
        p, s, = y[:-1], y[-1]
        _beta = self.beta(s, S_bar)
        _gamma = self.gamma(s, S_bar)
        A = diags([_beta[3:-1], self.alpha[2:-1], _gamma[1:-2]],
                [-2, -1, 0], shape=(self.M-2, self.M-3)).toarray()
        f = b[1:-1]
        f[0] -= _beta[1]*(self.option.K-s) + self.alpha[1]*((self.option.K-(1+self.dx)*s))
        f[1] -= _beta[2]*(self.option.K-(1+self.dx)*s)
        res = A@p - f
        return res
    
    def solve_non_linear_system(self, V: np.ndarray, S_bar: np.ndarray):
        *V[2:-1], S_bar = root(
            lambda y: self.system(y, np.copy(V[:]), S_bar),
            jac=lambda y: self.jacobian(y, S_bar),
            x0=np.concatenate([V[2:-1], [S_bar]])
        )['x']
        V[0] = self.option.payoff(S_bar)
        V[1] = self.option.payoff((1+self.dx)*S_bar)    
        return S_bar

def solve_explicitly(
    option: Option,  # the option
    r: float,  # risk-free interest rate
    sigma: float,  # sigma price volatitliy
    dx: float,  # grid resolution along x-axis
    dt: float,  # grid resolution along t-axis
    delta=0.,  # dividends
    x_max = 2.  # sufficiently large value for x
):
    match option.type:
        case OptionType.CALL:
            return CallOptionExplicitSolver(option, r, sigma, dx, dt, delta).solve()
        case OptionType.PUT:
            return PutOptionExplicitSolver(option, r, sigma, dx, dt, delta=delta, x_max=x_max).solve()

def solve_implicitly(
    option: Option,  # the option
    r: float,  # risk-free interest rate
    sigma: float,  # sigma price volatitliy
    dx: float,  # grid resolution along x-axis
    dt: float,  # grid resolution along t-axis
    delta=0.,  # dividends
    x_max = 2.
):

    """Solve front-fixing method explicitly.

    Parameters:
        option (AmericanOption): the option to price.
        r (float): the stock price risk-free interest rate.
        sigma (float): the stock price volatility.
        x_max (float): large value used as infity in the spatial.
        N (float): Grid resolution in the spatial direction.
        M (float): Grid resolution in the time direction.
    """
    match option.type:
        case OptionType.CALL:
            return CallOptionImplicitSolver(option, r, sigma, dx, dt, delta=delta).solve()
        case OptionType.PUT:
            return PutOptionImplicitSolver(option, r, sigma, dx, dt, delta,x_max).solve()