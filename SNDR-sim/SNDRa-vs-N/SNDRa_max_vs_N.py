 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:58:37 2021

@author: saionroy
"""

#Plot SNR_max
import numpy as np

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
ReRAM = np.max(np.load('SNDRa_dB_ReRAM_vs_N.npy'),axis=1)
MRAM = np.max(np.load('SNDRa_dB_MRAM_vs_N.npy'),axis=1)
FeFET = np.max(np.load('SNDRa_dB_FeFET_vs_N.npy'),axis=1)

N_all = np.logspace(1.5,4,30)

plt.close('all')

plt.rc('font', size=22, weight='bold') 

lin_w = 5

fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)
plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 

ax1.plot(N_all*0.5, ReRAM,c='blue', markersize=10, marker='s', linewidth=lin_w,label='ReRAM')
ax1.plot(N_all*0.5, MRAM,c='black', markersize=10, marker='s', linewidth=lin_w,label='MRAM')
ax1.plot(N_all*0.5, FeFET,c='red', markersize=10, marker='s', linewidth=lin_w,label='FeFET')

plt.ylabel(r'$\mathrm{SNDR}_{\mathrm{a,max}}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$N$',fontsize=32,fontdict=font)

ax1.grid(1,'major', linewidth=0.5, ls='--')
ax1.grid(1,'minor', linewidth=0.5, ls='--')
plt.xscale("log")
#ax1.legend(loc='best',prop={'size': 25})
ax1.set_axisbelow(True)
plt.box(1)
plt.savefig('SNDRa_max_vs_N.pdf', bbox_inches='tight',dpi=200)