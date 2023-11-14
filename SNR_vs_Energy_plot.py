#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 07:19:03 2022

@author: saionroy
"""
#Plotting Energy vs SNR
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
ReRAM = np.load('SNR_dB_ReRAM_vs_N.npy')
MRAM = np.load('SNR_dB_MRAM_vs_N.npy')
FeFET = np.load('SNR_dB_FeFET_vs_N.npy')

V_bl_all = np.logspace(-2,-0.09,25)
V_bl_FeFET = np.logspace(-2,0.7,25)

N_all = np.logspace(1.5,4,30)

ReRAM_max = np.max(ReRAM,axis=1)
MRAM_max = np.max(MRAM,axis=1)
FeFET_max = np.max(FeFET,axis=1)

max_idx = np.zeros(len(ReRAM_max))
for i in range(len(N_all)):
    for j in range(1,len(V_bl_all)):
        if(ReRAM[i,j]-ReRAM[i,j-1]<0.001):
            max_idx[i] = int(j)
            break
    ReRAM_max[i] = ReRAM[i,int(max_idx[i])]
Vbl_ReRAM_max = V_bl_all[max_idx.astype(int)]
    
max_idx = np.zeros(len(MRAM_max))
for i in range(len(N_all)):
    for j in range(1,len(V_bl_all)):
        if(MRAM[i,j]-MRAM[i,j-1]<0.001):
            max_idx[i] = int(j)
            break
    MRAM_max[i] = MRAM[i,int(max_idx[i])]
Vbl_MRAM_max = V_bl_all[max_idx.astype(int)]
   
max_idx = np.zeros(len(FeFET_max))
for i in range(len(N_all)):
    for j in range(1,len(V_bl_FeFET)):
        if(FeFET[i,j]-FeFET[i,j-1]<0.001):
            max_idx[i] = int(j)
            break
    FeFET_max[i] = FeFET[i,int(max_idx[i])]
Vbl_FeFET_max = V_bl_FeFET[max_idx.astype(int)]


E1op_ReRAM = np.zeros(len(ReRAM_max))
E1op_MRAM = np.zeros(len(MRAM_max))
E1op_FeFET = np.zeros(len(FeFET_max))

VDD = 0.8
VDD_FeFET = 5

Ravg_MRAM = (5e3 + 9e3)*0.5
Ravg_ReRAM = (25e3 + 300e3)*0.5
Ravg_FeFET = (1e6 + 1e9)*0.5

Tcore = 20e-9
k1 = 100e-15
k2 = 1e-18
#ENOB = 5.78
ENOB = 6
for i in range(len(ReRAM_max)):
    E1op_ReRAM[i] = VDD*Vbl_ReRAM_max[i]*Tcore/(2*Ravg_ReRAM) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
    E1op_MRAM[i] = VDD*Vbl_MRAM_max[i]*Tcore/(2*Ravg_MRAM) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
    E1op_FeFET[i] = VDD_FeFET*Vbl_FeFET_max[i]*Tcore/(2*Ravg_FeFET) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])

#%%
lin_w = 5
fig, ax = plt.subplots(figsize=(11,8))

plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 

plt.ylabel(r'$\mathrm{SNR}_{\mathrm{max}}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$E_{\mathrm{op1}}$ (fJ)',fontsize=32,fontdict=font)
plt.xscale("log")

plt.scatter(E1op_ReRAM, ReRAM_max,c="blue", s=100, marker='s', linewidth=lin_w)
plt.scatter(E1op_MRAM, MRAM_max,c="black", s=100, marker='s', linewidth=lin_w)
plt.scatter(E1op_FeFET, FeFET_max,c="red", s=100, marker='s', linewidth=lin_w)

ax.grid(1,'major', linewidth=0.5, color='black')
ax.grid(1,'minor', linewidth=0.5, ls='--')

ax.set_axisbelow(True)
plt.box(1)
#plt.savefig('SNR_max_vs_Eop1.png', bbox_inches='tight', dpi =200)   