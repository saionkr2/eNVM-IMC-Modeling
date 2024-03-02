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

ReRAM = np.max(np.load('SNDRa_dB_ReRAM_vs_ratio.npy'),axis=1)
MRAM = np.max(np.load('SNDRa_dB_MRAM_vs_ratio.npy'),axis=1)
FeFET = np.max(np.load('SNDRa_dB_FeFET_vs_ratio.npy'),axis=1)

ratio = np.logspace(0,3.25,15)


plt.close('all')

plt.rc('font', size=22, weight='bold') 

lin_w = 5

fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)
plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 
ax1.plot(ratio[1:], ReRAM[1:],c='blue', markersize=10, marker='s', linewidth=lin_w,label='ReRAM')
ax1.plot(ratio[1:], MRAM[1:],c='black', markersize=10, marker='s', linewidth=lin_w,label='MRAM')
ax1.plot(ratio[1:], FeFET[1:],c='red', markersize=10, marker='s', linewidth=lin_w,label='FeFET')
plt.xscale("log")
ax1.grid(1,'major', linewidth=0.5, ls='--')
ax1.grid(1,'minor', linewidth=0.5, ls='--')
plt.ylabel(r'$\mathrm{SNDR}_{\mathrm{a,max}}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$g_{\mathrm{on}}/g_{\mathrm{off}}$',fontsize=32,fontdict=font)
plt.savefig('SNDR_max_vs_ratio.pdf', bbox_inches='tight',dpi=200)
ax1.set_axisbelow(True)
plt.box(1)