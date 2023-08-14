from frontfixingmethod import nielsen
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

# option = PutAmericanOption(K, T)
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


option = PutAmericanOption(K, T)
dx = 1e-3 
dt = 1e-3
S, V, S_bar = nielsen.solve_implicitly(option, r, sigma, x_inf, dx, dt)
print(S_bar)

plt.plot(S, option.payoff(S))
plt.plot(S, V)
plt.vlines(S_bar, 0, V[V != option.payoff(S)][0], linestyles='dashed')
plt.xlim(0, S_bar*x_inf)
plt.ylim(bottom=0, top=option.K)
plt.show()
