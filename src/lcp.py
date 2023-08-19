import numpy as np
from scipy import interpolate
from option import AmericanOption

def solve(option: AmericanOption,
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
    x_axis = np.arange(x_min, x_max + dx, step=dx)
    t_axis = np.arange(0, option.T+dt, step=dt)
    tao_axis = 0.5*np.power(sigma,2)*t_axis
    dtao = np.diff(tao_axis)[0]

    # Parameters artificail space
    q = (2*r) / np.power(sigma, 2)
    q_delta = 2*(r-delta) / np.power(sigma, 2)

    def payoff(x, tao): return np.exp((tao/4)*(np.power(q_delta-1, 2) + 4*q)) * \
        np.maximum(np.exp((x/2)*(q_delta - 1)) - np.exp((x/2)*(q_delta + 1)), 0)

    class GUtil: 
        def __init__(self, x, tao):
            self.x = x
            self.tao = tao
        def __getitem__(self, indx):
            xi, ti = indx
            return payoff(self.x[xi], self.tao[ti])

    g = GUtil(x_axis, tao_axis)

    # calculate lambda and alpha
    lambd = dtao/np.power(dx, 2)
    alpha = lambd*theta

    # compute initial condition
    w = np.empty(x_axis.size)
    w = g[:, 0]
    b = np.empty_like(x_axis)
    v_new = np.empty_like(x_axis)
    for i in range(tao_axis.size - 1):
        b[2:-2] = w[2:-2] + lambd*(1-theta)*(w[3:-1] - 2*w[2:-2] + w[1:-3])
        b[1] = w[1] + lambd*(1-theta)*(w[2] -2*w[1] + g[0, i]) + alpha*g[0, i+1]
        b[-2] = w[-2] + lambd*(1-theta)*(g[-1, i]-2*w[-2] + w[-3]) + alpha*g[-1, i+1]

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
    

    S_axis = option.K*np.exp(x_axis)
    S_axis[0] = 0

    V = option.K*w*np.exp(-(x_axis/2)*(q_delta - 1)) * np.exp(-tao_axis[-1]*((1/4)*np.power(q_delta - 1, 2) + q))
    V[0] = option.K
    V_interp = interpolate.interp1d(S_axis, V)
    eps = 1e-5
    S_bar = S_axis[np.abs(V + S_axis - option.K) <= eps][-1]

    return V_interp, S_bar


