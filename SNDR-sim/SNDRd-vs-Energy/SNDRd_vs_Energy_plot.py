#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 07:19:03 2022

@author: saionroy
"""
#Plotting Energy vs SNDRd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

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
ReRAM = np.load('SNDRd_dB_ReRAM_vs_N.npy')
MRAM = np.load('SNDRd_dB_MRAM_vs_N.npy')
FeFET = np.load('SNDRd_dB_FeFET_vs_N.npy')

ReRAM_AG = np.load('AGeq_ReRAM_vs_N.npy')
MRAM_AG = np.load('AGeq_MRAM_vs_N.npy')
FeFET_AG = np.load('AGeq_FeFET_vs_N.npy')

V_bl_all = np.logspace(-3,-0.09,40)
V_bl_FeFET = np.logspace(-3,0.7,40)

N_all = np.logspace(1.5,4,30)*0.5

E1op_ReRAM = np.zeros(np.shape(ReRAM))
E1op_MRAM = np.zeros(np.shape(MRAM))
E1op_FeFET = np.zeros(np.shape(FeFET))

VDD = 0.8
VDD_FeFET = 5

Tcore = 20e-9
k1 = 100e-15
k2 = 1e-18
#ENOB = 5.78
ENOB = 6
for i in range(len(N_all)):
    for j in range(len(V_bl_all)):
        E1op_ReRAM[i,j] = VDD*V_bl_all[j]*Tcore*ReRAM_AG[i,j]/(2*N_all[i]) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
        E1op_MRAM[i,j] = VDD*V_bl_all[j]*Tcore*MRAM_AG[i,j]/(2*N_all[i]) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])
        E1op_FeFET[i,j] = VDD_FeFET*V_bl_FeFET[j]*Tcore*FeFET_AG[i,j]/(2*N_all[i]) + (k1*ENOB + k2*4**ENOB)/(2*N_all[i])

#%%
lin_w = 2
N_list = [4,10,18]

for i in range(3):
    print(N_all[N_list[i]])
    fig, ax = plt.subplots(figsize=(11,8))
    
    plt.rc('font', size=22, weight='bold') 
    
    plt.yticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    plt.xticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    
    plt.ylabel(r'$\mathrm{SNDR}$ (dB)',fontsize=32,fontdict=font)
    plt.xlabel(r'$E_{\mathrm{op1}}$ (fJ)',fontsize=32,fontdict=font)
    plt.xscale("log")
    #plt.xscale("log")
    if(i==0):
        ax.set_xlim([7,120])
    elif(i==1):
        ax.set_xlim([2,110])
    else:
        ax.set_xlim([0.3,50])
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        axis.set_major_formatter(formatter)
        
    min_E = np.min([E1op_ReRAM[N_list[i],-2],E1op_MRAM[N_list[i],-2],E1op_FeFET[N_list[i],-2]])
    lim_ReRAM = np.where(E1op_ReRAM[N_list[i],:]<min_E)[0][-1]
    lim_MRAM = np.where(E1op_MRAM[N_list[i],:]<min_E)[0][-1]
    lim_FeFET = np.where(E1op_FeFET[N_list[i],:]<min_E)[0][-1]
    
    plt.plot(E1op_ReRAM[N_list[i],:lim_ReRAM]/1e-15,ReRAM[N_list[i],:lim_ReRAM], c="blue", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(E1op_MRAM[N_list[i],:lim_MRAM]/1e-15,MRAM[N_list[i],:lim_MRAM], c="black", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(E1op_FeFET[N_list[i],:lim_FeFET]/1e-15,FeFET[N_list[i],:lim_FeFET],c="red", marker='s', linewidth=lin_w, markersize=10)
    ax.set_axisbelow(True)
    
    ax.grid(1,'major', linewidth=0.5, ls='--')
    ax.grid(1,'minor', linewidth=0.5, ls='--')
    
    plt.box(1)
    plt.savefig('E_1bop_vs_SNDRd_N_%d.pdf'%(int(N_all[N_list[i]])), bbox_inches='tight', dpi =200)  

