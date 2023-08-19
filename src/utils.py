import numpy as np
from scipy import stats

from option import Option

def calculate_amer_option(option: Option, S0, r, sigma, M, delta=0):
    dt = option.T / M 
    discount = np.exp(-r*dt)

    A = np.exp(-(r-delta)*dt)
    A += np.exp((r - delta + np.power(sigma,2))*dt)
    A *= 0.5

    u = A + np.sqrt(np.power(A, 2) - 1)
    d = A - np.sqrt(np.power(A, 2) - 1)
    p = np.exp((r-delta)*dt) - d
    p /= u-d

    S = np.empty((M+2, M+2))
    V = np.empty_like(S)
    S[0,0] = S0
    for m in range(1, M+1):
        for n in range(m+1, 0, -1):
            S[m,n] = u*S[m-1,n-1]
        S[m,0] = d*S[m-1,0]

    V[M] = option.payoff(S[M])

    for m in range(M, -1, -1):
        for n in range(0, m+1):
            hold = (1-p)*V[m+1][n] + p*V[m+1][n+1]
            hold *= discount
            V[m][n] = np.maximum(hold, option.payoff(S[m][n]))
    
    return V[0][0]
            

def black_scholes(S, T, K, r, sigma, option_type='put', d = 0):
    d1 = (np.log(S / K) + (r - d + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-d*T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:  # option_type == "put"
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-d*T) * stats.norm.cdf(-d1)


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

