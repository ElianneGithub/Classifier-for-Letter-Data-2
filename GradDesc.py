#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:43:36 2021

@author: joelda
"""

import numpy as np


def grad_n(f, epsilon = 10**(-6)):
    def grad(x, param):
        y = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_temp = np.array(x)
            x_temp[i] += epsilon
            x_temp2 = np.array(x)
            x_temp2[i] -= epsilon
            y[i] = (f(x_temp, param)-f(x_temp2, param))/(2*epsilon)
        return y
    return grad

def grad_desc_n(f, param, dim, nb_iter, step = 0.01, x_0 = None):
    if x_0 is None:
        x_0 = np.random.randn(dim)
    grad_f = grad_n(f)
    x = x_0
    for i in range(nb_iter):
        dx = grad_f(x, param)*step
        x -= dx
        print(i, f(x,param))
    return np.array(x)

def grad_desc_stoch(f, param, dim, nb_iter, mini_batch = 100, step = 0.01, x_0 = None):
    if x_0 is None:
        x_0 = np.random.randn(dim)
    grad_f = grad_n(f)
    x = x_0
    for i in range(nb_iter):
        idx = np.random.randint(param.shape[0], size=mini_batch)
        param_stoch = param[idx]
        k = 1000
        step_stoch = step*k/(i+k)
        dx = grad_f(x, param_stoch)*step
        x -= dx
        if i%10 == 0: print(i, f(x,param))
    return np.array(x)












