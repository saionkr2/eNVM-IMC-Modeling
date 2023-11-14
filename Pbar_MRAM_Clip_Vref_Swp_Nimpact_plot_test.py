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

MRAM_1 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar.npy'))[0,:]
MRAM_2 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar_par.npy'))[0,:]
MRAM_3 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar_par_fl.npy'))[0,:]
MRAM_4 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar_par_fl_m.npy'))[0,:]
MRAM_5 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar_par_fl_m_t.npy'))[0,:]
MRAM_6 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_bcvar_par_fl_m_t_q.npy'))[0,:]
MRAM_7 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_all.npy'))[0,:]
MRAM_8 = 10*np.log10(np.load('SNR_MRAM_vs_Vbl_only_quant_clip.npy'))[0,:]

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
ax1.set_ylim([3.5,50])
ax1.set_xlim([9, 55])
ax1.plot(V_bl_real, MRAM_1[:lim],c="blue", markersize=10, marker='o', linewidth=lin_w,label='CV')
ax1.plot(V_bl_real, MRAM_2[:lim],c="red", markersize=10, marker='o', linewidth=lin_w,label='CV-PC')
ax1.plot(V_bl_real, MRAM_3[:lim],c="green", markersize=10, marker='o', linewidth=lin_w,label='CV-FL')
ax1.plot(V_bl_real, MRAM_4[:lim],c="purple", markersize=10, marker='o', linewidth=lin_w,label='CV-MM')
#ax1.plot(V_bl_all[:lim]*1e3, MRAM_5[:lim],c="purple", markersize=10, marker='o', linewidth=lin_w,label=r'$\hat{\sigma}^2_{\mathrm{np}}$')
#ax1.plot(V_bl_all[:lim]*1e3, MRAM_6[:lim],c="grey", markersize=10, marker='o', linewidth=lin_w,label=r'$\hat{\sigma}^2_{\mathrm{nl}}$')
ax1.plot(V_bl_real, MRAM_8[:lim],c="black", markersize=10, marker='o', linewidth=lin_w,label='AC-AQ')
ax1.plot(V_bl_real, MRAM_7[:lim],c="orange", markersize=10, marker='o', linewidth=lin_w,label='All')

ax1.grid(1,'major',  axis='both',linewidth=0.5, ls='--')
ax1.grid(1,'minor',  axis='both', linewidth=0.5, ls='--')

plt.ylabel(r'$\mathrm{SNR}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$V_{\mathrm{ref}}$ (mV)',fontsize=32,fontdict=font)

plt.legend(loc='upper right', ncol=6,bbox_to_anchor=(1.1, 1.18),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax1.set_axisbelow(True)
plt.box(1)
plt.savefig('SNR_vs_Vref_MRAM_all_w_ADC.pdf', bbox_inches='tight',dpi=200)

