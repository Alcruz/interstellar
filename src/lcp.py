from abc import abstractmethod
import numpy as np
from scipy import interpolate
from option import Option, OptionType

class LCPSolver:
    def __init__(
        self,
        r: float, 
        sigma: float, 
        dx: float, 
        dt: float,
        x_min = -3.,
        x_max = 3.,
        theta = 0., 
        delta=0.
    ):
        self.r = r
        self.sigma = sigma
        self.dx = dx
        self.dt = dt
        self.x_min = x_min
        self.x_max = x_max
        self.theta = theta
        self.delta = delta
        self.x_axis = np.arange(self.x_min, self.x_max + self.dx, step=self.dx)
    
    @abstractmethod
    def get_g(self, x_axis, tao_axis, q_delta, q):
        pass

    def solve(
        self,
        option: Option,
        wR = 1,
        eps = 1e-24
    ):
        # Parameters artificail space
        q = (2*self.r) / np.power(self.sigma, 2)
        q_delta = 2*(self.r-self.delta) / np.power(self.sigma, 2)

        t_axis = np.arange(0, option.T+self.dt, step=self.dt)
        tao_axis = 0.5*np.power(self.sigma,2)*t_axis
        dtao = np.diff(tao_axis)[0]

        # calculate lambda and alpha
        lambd = dtao/np.power(self.dx, 2)
        alpha = lambd*self.theta

        g = self.get_g(self.x_axis, tao_axis, q_delta, q)

        # compute initial condition
        w = np.empty(self.x_axis.size)
        w = g[:, 0]
        b = np.empty_like(self.x_axis)
        v_new = np.empty_like(self.x_axis)

        for i in range(tao_axis.size - 1):
            b[2:-2] = w[2:-2] + lambd*(1-self.theta)*(w[3:-1] - 2*w[2:-2] + w[1:-3])
            b[1] = w[1] + lambd*(1-self.theta)*(w[2] -2*w[1] + g[0, i]) + alpha*g[0, i+1]
            b[-2] = w[-2] + lambd*(1-self.theta)*(g[-1, i]-2*w[-2] + w[-3]) + alpha*g[-1, i+1]

            v = np.maximum(w, g[:, i+1])
            while True:
                v_new[0] = v_new[-1] = 0
                for j in range(1, v.size - 1):
                    rho = b[j] + alpha*(v_new[j-1] + v[j+1])
                    rho /= 1+2*alpha
                    v_new[j] = np.maximum(g[j, i+1], v[j] + wR*(rho - v[j]))
                if np.linalg.norm(v_new - v, ord=2) <= eps:
                    break
                v = v_new.copy()
            w = v.copy()

        S_axis = option.K*np.exp(self.x_axis)
        S_axis[0] = 0
        V = self.get_surface(option, w, tao_axis, q, q_delta)
        # S_bar = self.get_early_exercise(option, S_axis, V)
        S_bar= None
        return S_axis, V, S_bar
    

class CallLCPSolver(LCPSolver):
    def __init__(self, r: float, sigma: float, dx: float, dt: float, x_min=-3, x_max=3, theta=0, delta=0):
        super().__init__(r, sigma, dx, dt, x_min, x_max, theta, delta)
    
    def get_g(self, x_axis, tao_axis, q_delta, q):
        return CallLCPSolver.GUtil(x_axis, tao_axis, q_delta, q)
    
    def get_early_exercise(self, option, S, V):
        eps = 1e-5
        S_bar = S[np.abs(option.K - S + V) <= eps][-1]
        return S_bar
    
    def get_surface(self, option, w, tao_axis, q, q_delta, ):
        V = option.K*w*np.exp(-(self.x_axis/2)*(q_delta - 1)) * np.exp(-tao_axis[-1]*((1/4)*np.power(q_delta - 1, 2) + q))
        V[0] = 0
        return V[:-1]
    
    class GUtil: 
        def __init__(self, x, tao, q_delta, q):
            self.x = x
            self.tao = tao
            self.q_delta = q_delta
            self.q = q

        def __getitem__(self, indx):
            xi, ti = indx
            return self.payoff(self.x[xi], self.tao[ti])

        def payoff(self, x, tao): return np.exp((tao/4)*(np.power(self.q_delta-1, 2) + 4*self.q)) * \
            np.maximum(np.exp((x/2)*(self.q_delta + 1)) - np.exp((x/2)*(self.q_delta - 1)), 0)

class PutLCPSolver(LCPSolver):
    def __init__(self, r: float, sigma: float, dx: float, dt: float, x_min=-3, x_max=3, theta=0, delta=0):
        super().__init__(r, sigma, dx, dt, x_min, x_max, theta, delta)
    
    def get_g(self, x_axis, tao_axis, q_delta, q):
        return PutLCPSolver.GUtil(x_axis, tao_axis, q_delta, q)
    
    def get_early_exercise(self, option, S, V):
        eps = 1e-5
        S_bar = S[np.abs(V + S - option.K) <= eps][-1]
        return S_bar

    def get_surface(self, option, w, tao_axis, q, q_delta):
        V = option.K*w*np.exp(-(self.x_axis/2)*(q_delta - 1)) * np.exp(-tao_axis[-1]*((1/4)*np.power(q_delta - 1, 2) + q))
        V[0] = option.K
        return V

    class GUtil: 
        def __init__(self, x, tao, q_delta, q):
            self.x = x
            self.tao = tao
            self.q_delta = q_delta
            self.q = q

        def __getitem__(self, indx):
            xi, ti = indx
            return self.payoff(self.x[xi], self.tao[ti])

        def payoff(self, x, tao): return np.exp((tao/4)*(np.power(self.q_delta-1, 2) + 4*self.q)) * \
            np.maximum(np.exp((x/2)*(self.q_delta - 1)) - np.exp((x/2)*(self.q_delta + 1)), 0)

def solve(option: Option,
        r: float, 
        sigma: float, 
        dx: float, 
        dt: float,
        x_min = -3.,
        x_max = 3.,
        theta = 0., 
        delta=0.,
        wR = 1,
        eps = 1e-24):
    match option.type:
        case OptionType.CALL:
            return CallLCPSolver(r, sigma, dx, dt, x_min, x_max, theta, delta)
        case OptionType.PUT:
            return PutLCPSolver(r, sigma, dx, dt, x_min, x_max, theta, delta)




