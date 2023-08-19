import numpy as np
from scipy import stats

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

