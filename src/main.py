from frontfixing import nielsen, company
import lcp
from option import PutAmericanOption

import numpy as np

import matplotlib.pyplot as plt

K = 1.0  # Strike price.
T = 1.0  # Maturity date.
r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1.0  # Strike price.
T = 1.0  # Maturity date.
x_inf = 2.0  # Front-fixing infinity.

option = PutAmericanOption(K, T)
S = np.linspace(0, 2, num=200)

# dx = 1e-3 
# dt = 5e-6
# S, V, S_bar = nielsen.solve_explicitly(option, r, sigma, x_inf, dx, dt)
# print(S_bar)

# plt.plot(S, option.payoff(S))
# plt.plot(S, V)
# plt.vlines(S_bar, 0, V[V != option.payoff(S)][0], linestyles='dashed')
# plt.xlim(0, S_bar*x_inf)
# plt.ylim(bottom=0, top=option.K)
# plt.show()

# dx = 1e-2 
# dt = 1e-2
# V, S_bar = nielsen.solve_implicitly(option, r, sigma, 3, dx, dt)
# print(S_bar)

# S = np.linspace(0, 1.5, num=200)
# plt.plot(S, option.payoff(S))
# plt.plot(S, V(S))
# plt.vlines(S_bar, 0, V(S_bar), linestyles='dashed')
# plt.xlim(0, 1.5)
# plt.ylim(bottom=0, top=option.K)
# plt.show()

dx, dt = 1e-2, 0.5*1e-4
V, S_bar = lcp.solve(option, r, sigma, dx, dt, theta=0)
print(S_bar)

plt.plot(S, option.payoff(S))
plt.plot(S, V(S))
plt.vlines(S_bar, 0, option.payoff(S_bar), linestyles='dashed')
plt.xlim(0, 2)
plt.ylim(bottom=0, top=option.K)
plt.show()


# dx = 1e-3
# dt = np.power(dx,2) / (np.power(sigma,2)+r*np.power(dx, 2))
# option = PutAmericanOption(K, T)
# V, S_bar = company.solve_explicitly(option, r, sigma, 3, dx, dt)
# print(S_bar)
# S = np.linspace(0, 2, num=200)
# plt.plot(S, option.payoff(S))
# plt.plot(S, V(S))
# plt.vlines(S_bar, 0, V(S_bar), linestyles='dashed')
# plt.xlim(left=0, right=2)
# plt.ylim(bottom=0, top=option.K)
# plt.show()