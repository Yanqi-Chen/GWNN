import os
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import math
#要求信号输入是列向量
def fc(L):
    return lambda x: -(x.T @ L @ x)
def get_lamda(L, n):
    cons = ({'type' : 'eq', 'fun' : lambda x: np.array([x.T @ x - 1])})
    x0 = np.ones((n, 1))
    res = minimize(fc(L), x0, method = 'SLSQP', constraints=cons)
    return -res
    """x = cp.Variable(n)
    goal = cp.Minimize(cp.quad_form(x, L))
    cons = [cp.sum_squares(x) == 1]
    prob = cp.Problem(cp.Minimize(goal), cons)
    print("asdasd")
    prob.solve()
    #a = x.value
    print(x.value)
    #La = L * a 
    #lamda_max = np.sum(La) / np.sum(a)
    """
def integrate_f(k, a, s):
    return lambda x : math.cos(k * x) * math.exp(-s * a (math.cos(x) + 1))
def get_c(i, a, s):
    v = integrate.quad(integrate_f(i, a, s), 0, math.pi) * 2 / math.pi
    return v
def get_fi(L, k, n, s):
    a = get_lamda(L, n) / 2
    c = []
    for i in range(k + 1):
        c.append(get_c(i, a))
    t = []
    t0 = np.identity(n)
    all_a = np.ones(n) * a
    t.append(t0)
    t.append((L - all_a) / a)
    for i in range(2, k + 1):
        now = (2 / a) * (L - t0) * t[i - 1] - t[i - 2]
        t.append(now)
    goal = (1 / 2) * c[0]
    for i in range(k + 1):
        goal += c[i] * t[i]
    return goal