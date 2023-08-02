import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.optimize import root

import datetime

import numpy as np

def solve(F, J, x0, epsilon=1e-6, max_iterations=100):
    """
    Newton's method for solving a system of nonlinear equations.
    
    Arguments:
        f: A function that takes a vector x and returns a vector of equations evaluated at x.
        J: A function that takes a vector x and returns the Jacobian matrix of f evaluated at x.
        x0: The initial guess for the solution vector.
        epsilon: The desired level of accuracy.
        max_iterations: The maximum number of iterations.
    
    Returns:
        The approximate solution vector.
    """
    x = x0
    for i in range(max_iterations):
        f_val = F(x)
        if np.linalg.norm(f_val) < epsilon:
            return x
        J_val = J(x)
        if np.linalg.det(J_val) == 0:
            break
        delta_x = np.linalg.solve(J_val, -f_val)
        x = x + delta_x
    return x


r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1.0  # Strike price.
T = 1.0  # Maturity date.
x_inf = 2.0  # Front-fixing infinity.

M = 999  # Number of internal nodes in spatial direction.
N = 999  # Number of internal nodes in time direction.

dt, dt_inv = T/(N+1), (N+1)/T
dx, dx_inv = (x_inf-1)/(M+1), (M+1)/(x_inf - 1)

lambd = (np.power(M+1, 2)/(N+1)) * (T/np.power(x_inf - 1, 2))
kappa = 0.5*((M+1)/(N+1)) * (T/(x_inf - 1))

x = np.arange(1, 2+dx, dx)

alpha = 1 + lambd*np.power(sigma*x, 2) + r*dt

def beta(s): 
    return -0.5*lambd*np.power(sigma,2)*np.power(x,2) + kappa*x*(r - (S_bar - s)/(dt*s))

def gamma(s): 
    return -0.5*lambd*np.power(sigma,2)*np.power(x,2) - kappa*x*(r - (S_bar - s)/(dt*s))

def F(y, b, _):
    p, s, = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    A = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-2]], [-2, -1, 0], shape=(M, M-1)).toarray()
    f = b[1:-1]
    f[0] -= _beta[1]*(K-s) + alpha[1]*((K-(1+dx)*s))
    f[1] -= _beta[2]*(K-(1+dx)*s)
    res = A@p - f
    return res

def J(y, _, S_bar):
    p, s, = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    dgamma_ds = - 0.5 * dx_inv * x * S_bar / s**2
    dbeta_ds = 0.5 * dx_inv * x * S_bar / s**2 

    retVal = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-1]], [-2, -1, 0], shape=(M,M)).toarray()

    retVal[-1, :] = 0
    retVal[:, -1] = 0

    retVal[0, -1] = dgamma_ds[1]*p[0] + dbeta_ds[1]*(K - S_bar) - _beta[1] - alpha[1]*(1+dx)
    retVal[1, -1] = dgamma_ds[2]*p[1] + dbeta_ds[2]*(K - (1+dx)*S_bar) - _beta[2]*(1+dx)
    retVal[2:-1, -1] = dbeta_ds[3:-2]*p[:-2] + dgamma_ds[3:-2]*p[2:]

    retVal[-1, -3] = _beta[-2]
    retVal[-1, -2] = alpha[-2]
    retVal[-1, -1] = dbeta_ds[-2]*p[-2]

    return retVal


starting_time = datetime.datetime.now()
print("Start time:", starting_time)
S_bar = K
p = np.zeros((M+2))
for n in range(N, -1, -1):
    *p[2:-1], S_bar = solve(
        F=lambda y: F(y, np.copy(p[:]), S_bar), 
        J=lambda y: J(y, np.copy(p[:]), S_bar), 
        x0=np.concatenate([p[2:-1], [S_bar]]),
        epsilon=1e-10,
        max_iterations=100
    )
    if (n%10) == 0:
        cur_time = datetime.datetime.now()
        print("Elapsed seconds since last iteration:", (cur_time - starting_time).seconds)
        print(S_bar)
    p[0] = K - S_bar
    p[1] = K - (1+dx)*S_bar


end = datetime.datetime.now()
print("Final time:", end)
print("S_bar:", S_bar)
S = S_bar * np.arange(0, x_inf+dx, step=dx) 
P = np.concatenate([K - S[S < S_bar], p])

plt.plot(S, np.maximum(K-S, 0))
plt.plot(S, P)
plt.vlines(S_bar, 0, p[0], linestyles='dashed')
plt.xlim(0, S_bar*x_inf)
plt.ylim(bottom=0, top=K)
plt.show()
