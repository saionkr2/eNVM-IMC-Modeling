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
ReRAM = np.load('SNR_dB_ReRAM_vs_N_E1b.npy')
MRAM = np.load('SNR_dB_MRAM_vs_N_E1b.npy')
FeFET = np.load('SNR_dB_FeFET_vs_N_E1b.npy')

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

lin_w = 5
fig, ax = plt.subplots(figsize=(11,8))

plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 

plt.ylabel(r'$\mathrm{SNDR}_{\mathrm{max}}$ (dB)',fontsize=32,fontdict=font)
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
V_all = 40
ReRAM_Emin = np.zeros([30,V_all])
MRAM_Emin = np.zeros([30,V_all])
FeFET_Emin = np.zeros([30,V_all])

SNR_MRAM_range = np.zeros([30,V_all])
SNR_ReRAM_range = np.zeros([30,V_all])
SNR_FeFET_range = np.zeros([30,V_all])


for i in range(30):
    SNR_MRAM_range[i,:] = np.linspace(np.max(MRAM[i,:])/50,np.max(MRAM[i,:])*0.99,V_all)
    SNR_ReRAM_range[i,:] = np.linspace(np.max(ReRAM[i,:])/50,np.max(ReRAM[i,:])*0.99,V_all)
    SNR_FeFET_range[i,:] = np.linspace(np.max(FeFET[i,:])/50,np.max(FeFET[i,:])*0.99,V_all)


for i in range(30):
    for k in range(V_all):        
        ReRAM_Emin[i,k] = np.min(E1op_ReRAM[i,np.where(ReRAM[i,:]>SNR_ReRAM_range[i,k])])
        MRAM_Emin[i,k] = np.min(E1op_MRAM[i,np.where(MRAM[i,:]>SNR_MRAM_range[i,k])])
        FeFET_Emin[i,k] = np.min(E1op_FeFET[i,np.where(FeFET[i,:]>SNR_FeFET_range[i,k])])


#%%
lin_w = 2
N_list = [7,14,21]

for i in range(3):
    print(N_all[N_list[i]])
    fig, ax = plt.subplots(figsize=(11,8))
    
    plt.rc('font', size=22, weight='bold') 
    
    plt.yticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    plt.xticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    
    plt.xlabel(r'$\mathrm{SNDR}$ (dB)',fontsize=32,fontdict=font)
    plt.ylabel(r'$E_{\mathrm{op1,min}}$ (fJ)',fontsize=32,fontdict=font)
    plt.yscale("log")
    plt.xscale("log")
    if(i==0):
        ax.set_xlim([0.08,50])
    elif(i==1):
        ax.set_xlim([0.005,50])
    else:
        ax.set_xlim([0.002,50])
    plt.plot(SNR_ReRAM_range[N_list[i],:], ReRAM_Emin[N_list[i],:], c="blue", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(SNR_MRAM_range[N_list[i],:], MRAM_Emin[N_list[i],:], c="black", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(SNR_FeFET_range[N_list[i],:], FeFET_Emin[N_list[i],:],c="red", marker='s', linewidth=lin_w, markersize=10)
    ax.set_axisbelow(True)
    #plt.scatter(MRAM_chip, MRAM_SNR_chip,c="black", s=1000, marker='*', linewidth=lin_w,zorder=2)
    #plt.scatter(ReRAM_chip, ReRAM_SNR_chip,c="blue", s=1000, marker='*', linewidth=lin_w,zorder=2)
    
    # plt.scatter(E1op_ReRAM_range, ReRAM_max,c="blue", s=100, marker='s', linewidth=lin_w)
    # plt.scatter(E1op_MRAM_range, MRAM_max,c="black", s=100, marker='s', linewidth=lin_w)
    # plt.scatter(E1op_FeFET_range, FeFET_max,c="red", s=100, marker='s', linewidth=lin_w)
    
    ax.grid(1,'major', linewidth=0.5, ls='--')
    ax.grid(1,'minor', linewidth=0.5, ls='--')
    
    plt.box(1)
    plt.savefig('E_1bop_min_vs_SNR_N_%d.pdf'%(int(N_all[N_list[i]])), bbox_inches='tight', dpi =200)  
    
#%%
lin_w=2
for i in range(3):
    print(N_all[N_list[i]])
    fig, ax = plt.subplots(figsize=(11,8))
    
    plt.rc('font', size=22, weight='bold') 
    
    plt.yticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    plt.xticks(fontname = "Arial",fontsize=24,fontweight='bold') 
    
    plt.xlabel(r'$\mathrm{SNDR}$ (dB)',fontsize=32,fontdict=font)
    plt.ylabel(r'$E_{\mathrm{op1,min}}$ (fJ)',fontsize=32,fontdict=font)
    plt.yscale("log")
    plt.xscale("log")
    if(i==0):
        ax.set_xlim([0.01,50])
    elif(i==1):
        ax.set_xlim([0.01,50])
    else:
        ax.set_xlim([0.01,50])
    plt.plot(ReRAM[N_list[i],np.where(ReRAM[N_list[i],:]>0.05)][0], E1op_ReRAM[N_list[i],np.where(ReRAM[N_list[i],:]>0.05)][0], c="blue", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(MRAM[N_list[i],np.where(MRAM[N_list[i],:]>0.05)][0], E1op_MRAM[N_list[i],np.where(MRAM[N_list[i],:]>0.05)][0], c="black", marker='s', linewidth=lin_w, markersize=10)
    plt.plot(FeFET[N_list[i],np.where(FeFET[N_list[i],:]>0.05)][0], E1op_FeFET[N_list[i],np.where(FeFET[N_list[i],:]>0.05)][0],c="red", marker='s', linewidth=lin_w, markersize=10)
    ax.set_axisbelow(True)
    #plt.scatter(MRAM_chip, MRAM_SNR_chip,c="black", s=1000, marker='*', linewidth=lin_w,zorder=2)
    #plt.scatter(ReRAM_chip, ReRAM_SNR_chip,c="blue", s=1000, marker='*', linewidth=lin_w,zorder=2)
    
    # plt.scatter(E1op_ReRAM_range, ReRAM_max,c="blue", s=100, marker='s', linewidth=lin_w)
    # plt.scatter(E1op_MRAM_range, MRAM_max,c="black", s=100, marker='s', linewidth=lin_w)
    # plt.scatter(E1op_FeFET_range, FeFET_max,c="red", s=100, marker='s', linewidth=lin_w)
    
    ax.grid(1,'major', linewidth=0.5, ls='--')
    ax.grid(1,'minor', linewidth=0.5, ls='--')
    
    plt.box(1)

