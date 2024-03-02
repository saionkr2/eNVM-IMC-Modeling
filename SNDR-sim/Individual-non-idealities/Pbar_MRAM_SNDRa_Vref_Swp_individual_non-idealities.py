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
V_bl_real = np.array([ 9.28317767, 11.2468265 , 13.62584138, 16.50808371, 20.        ,
       24.23055317, 29.35598535, 35.5655882 , 43.0886938 , 52.20314431])

MRAM_1 = np.load('SNDRa_MRAM_vs_Vbl_CV.npy')[0,:]
MRAM_2 = np.load('SNDRa_MRAM_vs_Vbl_PC.npy')[0,:]
MRAM_3 = np.load('SNDRa_MRAM_vs_Vbl_MM.npy')[0,:]
MRAM_4 = np.load('SNDRa_MRAM_vs_Vbl_AT.npy')[0,:]
MRAM_5 = np.load('SNDRa_MRAM_vs_Vbl_All.npy')[0,:]
MRAM_6 = np.load('SNDRd_MRAM_vs_Vbl_All.npy')[0,:]

#%%
V_bl_all = np.logspace(-2,-1.25,10)

plt.close('all')

plt.rc('font', size=22, weight='bold') 

lin_w = 3
lim = 9
fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)

ms = 12
plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 
plt.xscale("log")
#ax1.set_yscale('log', base=2)
ax1.set_ylim([0,27])
#ax1.set_xlim([9, 55])

ax1.plot(V_bl_real[:lim], MRAM_1[:lim],c="blue", markersize=ms, marker='o', linewidth=lin_w,label='CV')
ax1.plot(V_bl_real[:lim], MRAM_2[:lim],c="red", markersize=ms, marker='o', linewidth=lin_w,label='PC')
ax1.plot(V_bl_real[:lim], MRAM_3[:lim],c="purple", markersize=ms, marker='o', linewidth=lin_w,label='MM')
ax1.plot(V_bl_real[:lim], MRAM_4[:lim],c="orange", markersize=ms, marker='o', linewidth=lin_w,label='AT')
ax1.plot(V_bl_real[:lim], MRAM_5[:lim],c="black", markersize=ms, marker='o', linewidth=lin_w,label='All')
ax1.plot(V_bl_real[:lim], MRAM_6[:lim],c="black", markersize=ms, marker='d', linewidth=lin_w)

ax1.grid(1,'major',  axis='both',linewidth=0.5, ls='--')
ax1.grid(1,'minor',  axis='both', linewidth=0.5, ls='--')

plt.ylabel(r'$\mathrm{SNDR}_\mathrm{a}$/$\mathrm{SNDR}_\mathrm{d}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$V_{\mathrm{ref}}$ (mV)',fontsize=32,fontdict=font)

plt.legend(loc='upper right', ncol=6,bbox_to_anchor=(0.89, 1.18),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax1.set_axisbelow(True)
plt.box(1)
plt.savefig('SNDR_vs_Vref_MRAM_all.pdf', bbox_inches='tight',dpi=200)

