import datetime

import numpy as np

from scipy.sparse import diags
from scipy.optimize import root

from option import AmericanOption
from utils import solve


def solve_explicitly(option: AmericanOption,
                     r: float,
                     sigma: float,
                     x_max: float,
                     dx: float,
                     dt: float):
    """Solve front-fixing method explicitly.

    Parameters:
        option (AmericanOption): the option to price.
        r (float): the stock price risk-free interest rate.
        sigma (float): the stock price volatility.
        x_max (float): large value used as infity in the spatial.
        dx (float): Grid resolution in the spatial direction.
        dt (float): Grid resolution in the time direction.
    """
    dx_inv = np.round(dx)
    dt_inv = np.round(dt)
    print(dx, dt)
    print(dx_inv, dt_inv)

    gamma = dt / np.power(dx, 2)
    alpha = 0.5 * dt / dx

    x = np.arange(1, x_max+dx, dx)

    A = 0.5 * np.power(sigma*x, 2) * gamma - x * (r - dt_inv) * alpha
    B = 1 - np.power(sigma*x, 2) * gamma - r*dt
    C = 0.5 * np.power(sigma*x, 2) * gamma + x * (r - dt_inv) * alpha

    p = np.zeros_like(x)
    S_bar = option.K
    N = int(option.T/dt)
    for n in range(N, -1, -1):
        D = 0.5*x*dx_inv
        D[1:-1] *= (p[2:]-p[:-2]) * (1/S_bar)

        S_bar = option.K - (A[1]*p[0] + B[1]*p[1] + C[1]*p[2])
        S_bar /= D[1] + 1 + dx

        p[2:-1] = A[2:-1]*p[1:-2] + B[2:-1] * \
            p[2:-1] + C[2:-1]*p[3:] + D[2:-1]*S_bar
        p[0] = option.K - S_bar
        p[1] = option.K - (1+dx)*S_bar

    S = S_bar * np.arange(0, x_max+dx, step=dx)
    P = np.concatenate([option.payoff(S[S < S_bar]), p])

    return S, P, S_bar


def solve_implicitly(option: AmericanOption,
                     r: float,
                     sigma: float,
                     x_max: float,
                     dx: float,
                     dt: float):
    """Solve front-fixing method explicitly.

    Parameters:
        option (AmericanOption): the option to price.
        r (float): the stock price risk-free interest rate.
        sigma (float): the stock price volatility.
        x_max (float): large value used as infity in the spatial.
        N (float): Grid resolution in the spatial direction.
        M (float): Grid resolution in the time direction.
    """

    N = option.T / dt - 1
    M = (x_max-1) / dx - 1
    dt_inv = option.T/(N+1), (N+1)/option.T
    dx_inv = (x_max-1)/(M+1), (M+1)/(x_max-1)

    lambd = (np.power(M+1, 2)/(N+1)) * (option.T/np.power(x_max - 1, 2))
    kappa = 0.5*((M+1)/(N+1)) * (option.T/(x_max - 1))

    x = np.arange(1, 2+dx, dx)

    alpha = 1 + lambd*np.power(sigma*x, 2) + r*dt

    def beta(s):
        return -0.5*lambd*np.power(sigma, 2)*np.power(x, 2) + kappa*x*(r - (S_bar - s)/(dt*s))

    def gamma(s):
        return -0.5*lambd*np.power(sigma, 2)*np.power(x, 2) - kappa*x*(r - (S_bar - s)/(dt*s))

    def F(y, b, _):
        p, s, = y[:-1], y[-1]
        _beta = beta(s)
        _gamma = gamma(s)

        A = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-2]],
                  [-2, -1, 0], shape=(M, M-1)).toarray()
        f = b[1:-1]
        f[0] -= _beta[1]*(option.K-s) + alpha[1]*((option.K-(1+dx)*s))
        f[1] -= _beta[2]*(option.K-(1+dx)*s)
        res = A@p - f
        return res

    def J(y, _, S_bar):
        p, s, = y[:-1], y[-1]
        _beta = beta(s)
        _gamma = gamma(s)

        dgamma_ds = - 0.5 * dx_inv * x * S_bar / s**2
        dbeta_ds = 0.5 * dx_inv * x * S_bar / s**2

        retVal = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-1]],
                       [-2, -1, 0], shape=(M, M)).toarray()

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
    p = np.zeros((M+2))
    for n in range(N, -1, -1):
        *p[2:-1], S_bar = solve(
            F=lambda y: F(y, np.copy(p[:]), S_bar),
            J=lambda y: J(y, np.copy(p[:]), S_bar),
            x0=np.concatenate([p[2:-1], [S_bar]]),
            epsilon=1e-10,
            max_iterations=100
        )
        if (n % 10) == 0:
            cur_time = datetime.datetime.now()
            print("Elapsed seconds since last iteration:",
                  (cur_time - starting_time).seconds)
            print(S_bar)
        p[0] = option.K - S_bar
        p[1] = option.K - (1+dx)*S_bar

    end = datetime.datetime.now()
    print("Final time:", end)
    print("S_bar:", S_bar)
    S = S_bar * np.arange(0, x_inf+dx, step=dx)
    P = np.concatenate([option.K - S[S < S_bar], p])

    # plt.plot(S, np.maximum(K-S, 0))
    # plt.plot(S, P)
    # plt.vlines(S_bar, 0, p[0], linestyles='dashed')
    # plt.xlim(0, S_bar*x_inf)
    # plt.ylim(bottom=0, top=K)
    # plt.show()
