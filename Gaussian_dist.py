# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import csv
import math
from scipy import special
import scipy.stats
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib

plt.rc('font', size=18, weight='regular') 
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'cm'
matplotlib.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "arial"
np.random.seed(2021)
font = {'family': 'arial',
        'color':  'black',
        'weight': 'regular',
        'size': 16,
        }
#%%

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 6*sigma, mu +6*sigma, 100)

plt.rc('font', size=22, weight='bold') 

lin_w = 5
lim = 17
fig1 = plt.figure(figsize=(14,5))
ax1 = fig1.add_subplot(1,1,1)

#plt.xticks([])
#plt.yticks([])
plt.yticks(fontname = "Arial",fontsize=32) 
plt.xticks(fontname = "Arial",fontsize=32) 
plt.tick_params(axis='both', which='major', pad=15)

plt.plot(x, stats.norm.pdf(x, mu, sigma),c='black', markersize=20,linewidth=4.0)
plt.box(False) 
ax1.set_axisbelow(True)
#ax1.grid(1,'major',  axis='both',linewidth=0.5, color='black')
#ax1.grid(1,'minor',  axis='both', linewidth=0.5, ls='--')

#plt.ylabel(None)
#plt.xlabel(None)


#plt.show()

#ax1.legend(loc='best',prop={'size': 25})

plt.savefig('Distribution_shape.pdf')

