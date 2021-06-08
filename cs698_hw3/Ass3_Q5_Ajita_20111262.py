#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:24:21 2021

@author: ajitashree
"""


import numpy as np
import matplotlib.pyplot as plt

# True function
def px(x):
    return np.exp(np.sin(x))

# Proposal function
def qx(x, mu, sig):
    num = np.exp(-(((x-mu)**2))/(2*sig**2))
    den = np.sqrt(2 * 3.14 * (sig**2))
    return num/den


iterations = 10000

'''
sigma = 1.1 # 440.631
sigma = 2 # 46
sigma = 1 # 942.452
'''
sigma = 1.5 #91.39


M_min = sigma * np.sqrt(2*3.14) * np.exp(1 + 0.5 * ((3.14/sigma)**2)) 


# Rejection sampling algorithm
samples = []

while len(samples) < 10000:
    
    # sample x* from q(x)
    x_ = np.random.normal(0, (sigma**2))
    
    # sample a uniform rv r [0, Mq(x*)]
    u = np.random.uniform(0, M_min * qx(x_, 0, (sigma**2)))
    
    # if u <= p(z*), accept else reject
    if (u  <= px(x_) and x_ <= 3.14 and x_ >= -3.14):
        samples += [x_]
        
# All accepted z*'s are random samples from p(x)
X = np.linspace(-2*3.14, 2*3.14, 100)
plt.figure(figsize = (10,10))

#plt.hist(samples, 100, density=True)
plt.hist(samples, 500)
plt.grid()
plt.show()






