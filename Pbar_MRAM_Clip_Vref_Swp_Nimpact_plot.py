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
V_bl_real = np.array([10,12,14,17,20,24,28,33,38,44,50])

MRAM_1 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_CV.npy'))[0,:]
MRAM_2 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_CV_PC.npy'))[0,:]
MRAM_3 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_CV_PC_FL.npy'))[0,:]
MRAM_4 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_CV_PC_FL_MM.npy'))[0,:]
MRAM_5 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_AQ_AC.npy'))[0,:]
MRAM_6 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_all.npy'))[0,:]

V_bl_all = np.logspace(-2,-0.09,25)

plt.close('all')

plt.rc('font', size=22, weight='bold') 

lin_w = 5
lim = 11
fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)

plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 
plt.xscale("log")
ax1.set_yscale('log', base=2)
ax1.set_ylim([2.5,20])
ax1.set_xlim([9, 55])
#ax1.plot(V_bl_real, MRAM_5[:lim],c="black", markersize=10, marker='o', linewidth=lin_w,label='Ideal')
ax1.plot(V_bl_real, MRAM_1[:lim],c="blue", markersize=10, marker='o', linewidth=lin_w,label='CV')
ax1.plot(V_bl_real, MRAM_2[:lim],c="red", markersize=10, marker='o', linewidth=lin_w,label='CV-PC')
ax1.plot(V_bl_real, MRAM_3[:lim],c="green", markersize=10, marker='o', linewidth=lin_w,label='CV-FL')
ax1.plot(V_bl_real, MRAM_4[:lim],c="purple", markersize=10, marker='o', linewidth=lin_w,label='CV-MM')
ax1.plot(V_bl_real, MRAM_6[:lim],c="orange", markersize=10, marker='o', linewidth=lin_w,label='All')

ax1.grid(1,'major',  axis='both',linewidth=0.5, ls='--')
ax1.grid(1,'minor',  axis='both', linewidth=0.5, ls='--')

plt.ylabel(r'$\mathrm{SNDR}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$V_{\mathrm{ref}}$ (mV)',fontsize=32,fontdict=font)

plt.legend(loc='upper right', ncol=6,bbox_to_anchor=(1.0, 1.18),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax1.set_axisbelow(True)
plt.box(1)
plt.savefig('SNR_vs_Vref_MRAM_all.pdf', bbox_inches='tight',dpi=200)

