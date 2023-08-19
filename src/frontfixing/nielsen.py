import datetime

import numpy as np

from scipy import interpolate
from scipy.sparse import diags
from scipy.optimize import root

from option import AmericanOption


def solve_explicitly(
    option: AmericanOption,  # the option
    r: float,  # risk-free interest rate
    sigma: float,  # sigma price volatitliy
    x_max: float,  # sufficiently large value for x
    dx: float,  # grid resolution along x-axis
    dt: float,  # grid resolution along t-axis
    delta=0  # dividends
):
    """Solve front-fixing method explicitly.

    Parameters:
        option (AmericanOption): the option to price.
        r (float): the stock price risk-free interest rate.
        sigma (float): the stock price volatility.
        x_max (float): large value used as infity in the spatial.
        dx (float): Grid resolution in the spatial direction.
        dt (float): Grid resolution in the time direction.
    """
    gamma = dt / np.power(dx, 2)
    alpha = 0.5 * dt / dx

    x = np.arange(1, x_max+dx, dx)

    A = 0.5 * np.power(sigma*x, 2) * gamma - x * ((r-delta) - (1/dt)) * alpha
    B = 1 - np.power(sigma*x, 2) * gamma - r*dt
    C = 0.5 * np.power(sigma*x, 2) * gamma + x * ((r-delta) - (1/dt)) * alpha

    V = np.zeros_like(x)
    S_bar = option.K
    for _ in np.arange(0, option.T, dt):
        D = 0.5*x*(1/dx)
        D[1:-1] *= (V[2:]-V[:-2]) * (1/S_bar)

        S_bar = option.K - (A[1]*V[0] + B[1]*V[1] + C[1]*V[2])
        S_bar /= D[1] + 1 + dx

        V[2:-1] = A[2:-1]*V[1:-2] + B[2:-1] * \
            V[2:-1] + C[2:-1]*V[3:] + D[2:-1]*S_bar
        V[0] = option.K - S_bar
        V[1] = option.K - (1+dx)*S_bar
    
    S_region =  np.linspace(0, S_bar, num=200, endpoint=False)
    C_region =  S_bar * x
    P = option.payoff(S_region)
    V_interp = interpolate.interp1d(
        np.concatenate([S_region, C_region]), 
        np.concatenate([P, V]), 
        kind='cubic'
    )
    return V_interp, S_bar


def solve_implicitly(
    option: AmericanOption,  # the option
    r: float,  # risk-free interest rate
    sigma: float,  # sigma price volatitliy
    x_max: float,  # sufficiently large value for x
    dx: float,  # grid resolution along x-axis
    dt: float,  # grid resolution along t-axis
    delta=0  # dividends
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
    lambd = dt/np.power(dx, 2)
    kappa = dt/dx
    
    x = np.arange(1, x_max+dx, dx)
    M = x.size

    alpha = 1 + lambd*np.power(sigma*x, 2) + r*dt

    def beta(s):
        return -0.5*lambd*np.power(sigma, 2)*np.power(x, 2) + 0.5*kappa*x*((r-delta) - (S_bar - s)/(dt*s))

    def gamma(s):
        return -0.5*lambd*np.power(sigma, 2)*np.power(x, 2) - 0.5*kappa*x*((r-delta) - (S_bar - s)/(dt*s))

    def F(y, b, _):
        p, s, = y[:-1], y[-1]
        _beta = beta(s)
        _gamma = gamma(s)
        A = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-2]],
                  [-2, -1, 0], shape=(M-2, M-3)).toarray()
        f = b[1:-1]
        f[0] -= _beta[1]*(option.K-s) + alpha[1]*((option.K-(1+dx)*s))
        f[1] -= _beta[2]*(option.K-(1+dx)*s)
        res = A@p - f
        return res

    def J(y, _, S_bar):
        p, s, = y[:-1], y[-1]
        _beta = beta(s)
        _gamma = gamma(s)

        dgamma_ds = - 0.5 * (1/dx) * x * S_bar / s**2
        dbeta_ds = 0.5 * (1/dx) * x * S_bar / s**2

        retVal = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-1]],
                       [-2, -1, 0], shape=(M-2, M-2)).toarray()

        retVal[-1, :] = 0
        retVal[:, -1] = 0

        retVal[0, -1] = dgamma_ds[1]*p[0] + dbeta_ds[1] * \
            (option.K - S_bar) - _beta[1] - alpha[1]*(1+dx)
        retVal[1, -1] = dgamma_ds[2]*p[1] + dbeta_ds[2] * \
            (option.K - (1+dx)*S_bar) - _beta[2]*(1+dx)
        retVal[2:-1, -1] = dbeta_ds[3:-2]*p[:-2] + dgamma_ds[3:-2]*p[2:]

        retVal[-1, -3] = _beta[-2]
        retVal[-1, -2] = alpha[-2]
        retVal[-1, -1] = dbeta_ds[-2]*p[-2]

        return retVal

    starting_time = datetime.datetime.now()
    print("Start time:", starting_time)
    S_bar = option.K
    V = np.zeros_like(x)

    for _ in np.arange(0, option.T, dt):
        *V[2:-1], S_bar = root(
            lambda y: F(y, np.copy(V[:]), S_bar),
            jac=lambda y: J(y, np.copy(V[:]), S_bar),
            x0=np.concatenate([V[2:-1], [S_bar]]),
        )['x']
        V[0] = option.K - S_bar
        V[1] = option.K - (1+dx)*S_bar

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
    return V, S_bar
