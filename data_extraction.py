#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:28:13 2021

@author: raphaelbailly
"""
import matplotlib.pyplot as plt
import numpy as np

i = 0

X = []
Y = []
Z = []
with open("letter.data.txt", 'r') as infile:
    for line in infile:
        line = line.split()
        V = line[6:]
        V = [int(v) for v in V]
        V = np.array(V)
        X.append(V)
        Y.append(ord(line[1]) - 97)
        V = V.reshape(16,8)
        Z.append(V)
        
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
        
        
        