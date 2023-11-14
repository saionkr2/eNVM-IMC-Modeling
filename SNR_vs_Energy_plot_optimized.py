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

E1op_ReRAM = np.zeros(np.shape(ReRAM))
E1op_MRAM = np.zeros(np.shape(MRAM))
E1op_FeFET = np.zeros(np.shape(FeFET))

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
for i in range(len(N_all)):
    for j in range(len(V_bl_all)):
        E1op_ReRAM[i,j] = VDD*V_bl_all[j]*Tcore/(2*Ravg_ReRAM) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
        E1op_MRAM[i,j] = VDD*V_bl_all[j]*Tcore/(2*Ravg_MRAM) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
        E1op_FeFET[i,j] = VDD_FeFET*V_bl_FeFET[j]*Tcore/(2*Ravg_FeFET) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])

lin_w = 5
fig, ax = plt.subplots(figsize=(11,8))

plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 

plt.ylabel(r'$\mathrm{SNR}_{\mathrm{max}}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$E_{\mathrm{op1}}$ (fJ)',fontsize=32,fontdict=font)
plt.xscale("log")

#plt.scatter(E1op_ReRAM, ReRAM,c="blue", s=100, marker='s', linewidth=lin_w)
plt.scatter(E1op_MRAM, MRAM,c="black", s=100, marker='s', linewidth=lin_w)
#plt.scatter(E1op_FeFET, FeFET,c="red", s=100, marker='s', linewidth=lin_w)

ax.grid(1,'major', linewidth=0.5, color='black')
ax.grid(1,'minor', linewidth=0.5, ls='--')

ax.set_axisbelow(True)
plt.box(1)
#%%
ReRAM_max = np.zeros(100)
MRAM_max = np.zeros(100)
FeFET_max = np.zeros(100)

E1op_MRAM_range = np.linspace(np.max(E1op_MRAM)/80,np.max(E1op_MRAM),100)
E1op_ReRAM_range = np.linspace(np.max(E1op_ReRAM)/80,np.max(E1op_ReRAM),100)
E1op_FeFET_range = np.linspace(np.max(E1op_FeFET)/80,np.max(E1op_FeFET),100)

for i in range(100):
    
    ReRAM_max[i] = np.max(ReRAM[np.where(E1op_ReRAM<E1op_ReRAM_range[i])])
    MRAM_max[i] = np.max(MRAM[np.where(E1op_MRAM<E1op_MRAM_range[i])])
    FeFET_max[i] = np.max(ReRAM[np.where(E1op_FeFET<E1op_FeFET_range[i])])


MRAM_chip = np.array([196,24])*1e-15
ReRAM_chip = np.array([41.49,4.04,1.88])*1e-15

MRAM_SNR_chip = np.zeros(len(MRAM_chip))
ReRAM_SNR_chip = np.zeros(len(ReRAM_chip))

for i in range(len(ReRAM_chip)):   
    ReRAM_SNR_chip[i] = np.max(ReRAM[np.where(E1op_ReRAM<ReRAM_chip[i])])
    
for i in range(len(MRAM_chip)):   
    MRAM_SNR_chip[i] = np.max(MRAM[np.where(E1op_MRAM<MRAM_chip[i])])

#%%
ReRAM_Emin = np.zeros(100)
MRAM_Emin = np.zeros(100)
FeFET_Emin = np.zeros(100)

SNR_MRAM_range = np.linspace(np.max(MRAM)/100,np.max(MRAM)*0.9,100)
SNR_ReRAM_range = np.linspace(np.max(ReRAM)/100,np.max(ReRAM)*0.9,100)
SNR_FeFET_range = np.linspace(np.max(FeFET)/100,np.max(FeFET)*0.9,100)

# SNR_MRAM_range = np.linspace(np.max(MRAM)/80,np.max(MRAM),100)
# SNR_ReRAM_range = np.linspace(np.max(ReRAM)/80,np.max(ReRAM),100)
# SNR_FeFET_range = np.linspace(np.max(FeFET)/80,np.max(FeFET),100)

for i in range(100):
    
    ReRAM_Emin[i] = np.min(E1op_ReRAM[np.where(ReRAM>SNR_ReRAM_range[i])])
    MRAM_Emin[i] = np.min(E1op_MRAM[np.where(MRAM>SNR_MRAM_range[i])])
    FeFET_Emin[i] = np.min(E1op_ReRAM[np.where(FeFET>SNR_FeFET_range[i])])


    
#%%
lin_w = 2
fig, ax = plt.subplots(figsize=(11,8))

plt.rc('font', size=22, weight='bold') 

plt.yticks(fontname = "Arial",fontsize=24,fontweight='bold') 
plt.xticks(fontname = "Arial",fontsize=24,fontweight='bold') 

plt.ylabel(r'$\mathrm{SNR}_{\mathrm{max}}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$E_{\mathrm{op1}}$ (fJ)',fontsize=32,fontdict=font)
plt.xscale("log")

plt.plot(E1op_ReRAM_range, ReRAM_max,c="blue", marker='s', linewidth=lin_w, markersize=10)
plt.plot(E1op_MRAM_range, MRAM_max,c="black", marker='s', linewidth=lin_w, markersize=10)
plt.plot(E1op_FeFET_range, FeFET_max,c="red", marker='s', linewidth=lin_w, markersize=10)
ax.set_axisbelow(True)
#plt.scatter(MRAM_chip, MRAM_SNR_chip,c="black", s=1000, marker='*', linewidth=lin_w,zorder=2)
#plt.scatter(ReRAM_chip, ReRAM_SNR_chip,c="blue", s=1000, marker='*', linewidth=lin_w,zorder=2)

# plt.scatter(E1op_ReRAM_range, ReRAM_max,c="blue", s=100, marker='s', linewidth=lin_w)
# plt.scatter(E1op_MRAM_range, MRAM_max,c="black", s=100, marker='s', linewidth=lin_w)
# plt.scatter(E1op_FeFET_range, FeFET_max,c="red", s=100, marker='s', linewidth=lin_w)

ax.grid(1,'major', linewidth=0.5, color='black')
ax.grid(1,'minor', linewidth=0.5, ls='--')


plt.box(1)
#plt.savefig('SNR_max_vs_Eop1.png', bbox_inches='tight', dpi =200)   

#%%
lin_w = 2
fig, ax = plt.subplots(figsize=(11,8))

plt.rc('font', size=22, weight='bold') 

plt.yticks(fontname = "Arial",fontsize=24,fontweight='bold') 
plt.xticks(fontname = "Arial",fontsize=24,fontweight='bold') 

plt.xlabel(r'$\mathrm{SNR}$ (dB)',fontsize=32,fontdict=font)
plt.ylabel(r'$E_{\mathrm{op1,min}}$ (fJ)',fontsize=32,fontdict=font)
plt.yscale("log")

plt.plot(SNR_ReRAM_range, ReRAM_Emin, c="blue", marker='s', linewidth=lin_w, markersize=10)
plt.plot(SNR_MRAM_range, MRAM_Emin, c="black", marker='s', linewidth=lin_w, markersize=10)
plt.plot(SNR_FeFET_range, FeFET_Emin,c="red", marker='s', linewidth=lin_w, markersize=10)
ax.set_axisbelow(True)
#plt.scatter(MRAM_chip, MRAM_SNR_chip,c="black", s=1000, marker='*', linewidth=lin_w,zorder=2)
#plt.scatter(ReRAM_chip, ReRAM_SNR_chip,c="blue", s=1000, marker='*', linewidth=lin_w,zorder=2)

# plt.scatter(E1op_ReRAM_range, ReRAM_max,c="blue", s=100, marker='s', linewidth=lin_w)
# plt.scatter(E1op_MRAM_range, MRAM_max,c="black", s=100, marker='s', linewidth=lin_w)
# plt.scatter(E1op_FeFET_range, FeFET_max,c="red", s=100, marker='s', linewidth=lin_w)

ax.grid(1,'major', linewidth=0.5, color='black')
ax.grid(1,'minor', linewidth=0.5, ls='--')


plt.box(1)
#plt.savefig('SNR_max_vs_Eop1.png', bbox_inches='tight', dpi =200)  