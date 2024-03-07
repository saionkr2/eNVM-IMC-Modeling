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
import copy

import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "arial"
np.random.seed(2021)
font = {'family': 'arial',
        'color':  'black',
        'weight': 'regular',
        'size': 22,
        }

def Qfunc(x):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def Quantize(x,clpmax,clpmin,BADC):
    xclp = np.minimum(np.maximum(x,clpmin),clpmax) - clpmin
    lim = (clpmax-clpmin)/(2**BADC-2)
    val = np.round(xclp/lim)*2-2**BADC
    return val

def par(a,b):
    return a*b/(a+b)
#%%

g_on = 1/(4.5e3)    #1/Ohm
g_off = 1/(7.5e3)   #1/Ohm

VDD = 0.8     # Volts

#MRAM
A = 1000
gm = 100e-6

m = 20
rbl = 183/512 
rsl = 84/512
Sig_Ith = 1.28e-6/m
Sig_mb_goff = 0.04*g_off
Sig_mb_gon = 0.04*g_on

Sig_m = 0.05

V_bl_all = np.logspace(-2,-0.09,25)

N_all = np.logspace(1.5,4,30)
a_vec_no = 61
w_vec_no = 61

Var_sig = np.zeros((len(N_all),len(V_bl_all)))

In = np.zeros((len(N_all),len(V_bl_all),a_vec_no*w_vec_no))

Var_noi_tot = np.zeros((len(N_all),len(V_bl_all)))
Var_noi_OCCtot = np.zeros((len(N_all),len(V_bl_all)))

for Vind in range(len(V_bl_all)):
    
    V_bl_s = V_bl_all[Vind]

    for Nind in range(len(N_all)):
        
        ## array dependent variable setup
        N = int(N_all[Nind]/2)
        ## Generating G vector for given N
        ran_w_bin = np.random.randint(0,2,size=(N,w_vec_no))
        ran_w_ap = 2*(ran_w_bin>0.5)-1
        g_vec = np.zeros([2*N,w_vec_no])
        g_vec_ideal = np.zeros([2*N,w_vec_no])
        
        g_vec_ideal[np.where(ran_w_ap<0)[0]*2,np.where(ran_w_ap<0)[1]] = g_off 
        g_vec_ideal[np.where(ran_w_ap<0)[0]*2+1,np.where(ran_w_ap<0)[1]] = g_on 

        g_vec_ideal[np.where(ran_w_ap>0)[0]*2,np.where(ran_w_ap>0)[1]] = g_on 
        g_vec_ideal[np.where(ran_w_ap>0)[0]*2+1,np.where(ran_w_ap>0)[1]] = g_off 
        
        g_vec[np.where(ran_w_ap<0)[0]*2,np.where(ran_w_ap<0)[1]] = g_off +Sig_mb_goff*np.random.normal(size=(np.where(ran_w_ap<0)[0].shape[0]))
        g_vec[np.where(ran_w_ap<0)[0]*2+1,np.where(ran_w_ap<0)[1]] = g_on + Sig_mb_gon*np.random.normal(size=(np.where(ran_w_ap<0)[0].shape[0]))

        g_vec[np.where(ran_w_ap>0)[0]*2,np.where(ran_w_ap>0)[1]] = g_on +Sig_mb_gon*np.random.normal(size=(np.where(ran_w_ap>0)[0].shape[0]))
        g_vec[np.where(ran_w_ap>0)[0]*2+1,np.where(ran_w_ap>0)[1]] = g_off + Sig_mb_goff*np.random.normal(size=(np.where(ran_w_ap>0)[0].shape[0]))
        
        ## Generating all input V vectors 
        a_mat = np.random.randint(0,2,size=(N,a_vec_no))*2-1
        a_mat_float = a_mat.astype('float64')
        a_mat_bin = np.zeros([2*N,a_vec_no]) 
        
        a_mat_bin[np.where(a_mat<0)[0]*2,np.where(a_mat<0)[1]] = 0
        a_mat_bin[np.where(a_mat<0)[0]*2+1,np.where(a_mat<0)[1]] = 1
        a_mat_bin[np.where(a_mat>0)[0]*2,np.where(a_mat>0)[1]] = 1
        a_mat_bin[np.where(a_mat>0)[0]*2+1,np.where(a_mat>0)[1]] = 0
                    
        # Element Wise Multiplication
        y_ideal = np.matmul(a_mat_float.T,ran_w_ap).reshape(-1)
        I_sig = np.matmul(a_mat_bin.T,g_vec_ideal).reshape(-1)*V_bl_s/m
        
        y_mult_wdg = np.zeros([a_vec_no,2*N,w_vec_no])
        for i in range(w_vec_no):
            y_mult_wdg[:,:,i] = np.multiply(a_mat_bin.T,1/(g_vec[:,i]))
        
        R_mul_wdg = np.where(y_mult_wdg == 0 , 1e23, y_mult_wdg)
        R_mul_ones = np.where(g_vec_ideal == g_off , 1e23, 1/g_vec)
        R_mul_m_ones = np.where(g_vec_ideal == g_on , 1e23, 1/g_vec)
        
        R_temp_wdg = R_mul_wdg[:,2*N-1,:]
        R_temp_ones = R_mul_ones[2*N-1,:]
        R_temp_m_ones = R_mul_m_ones[2*N-1,:]
        
        for r in range(2*N-1,0,-1):
            R_eq_wdg = par(rbl + R_temp_wdg + rsl, R_mul_wdg[:,r-1,:])
            R_temp_wdg = R_eq_wdg
            
            R_eq_ones = par(rbl + R_temp_ones + rsl, R_mul_ones[r-1,:])
            R_temp_ones = R_eq_ones
            
            R_eq_m_ones = par(rbl + R_temp_m_ones + rsl, R_mul_m_ones[r-1,:])
            R_temp_m_ones = R_eq_m_ones
        
        
        R_temp_wdg = R_temp_wdg.reshape(-1)
        Isig_wpar_wl = (V_bl_s*((A*gm*R_temp_wdg)/(1+A*gm*R_temp_wdg))*1/R_temp_wdg)
        Isig_wpar_wl_wm = (Isig_wpar_wl)*1/(m+np.sqrt(m)*Sig_m*np.random.normal(size=(np.shape(Isig_wpar_wl))))
        
        Ith = np.random.normal(0, Sig_Ith, size=(np.shape(Isig_wpar_wl)))
        
        Isig_wpar_wl_wm_wth = Isig_wpar_wl_wm + Ith 
        
        
        Isig_wpar_wl_wm_wth_scaled = Isig_wpar_wl_wm_wth - np.mean(Isig_wpar_wl_wm_wth)
        Isig_wpar_wl_wm_wth_scaled = Isig_wpar_wl_wm_wth_scaled*(np.std(I_sig)/np.std(Isig_wpar_wl_wm_wth))
        Isig_wpar_wl_wm_wth_scaled = Isig_wpar_wl_wm_wth_scaled + np.mean(I_sig)
        
        In[Nind,Vind,:] = Isig_wpar_wl_wm_wth_scaled - I_sig
        
        Var_sig[Nind,Vind] = np.var(I_sig)
        
        Var_noi_tot[Nind,Vind] = np.var(In[Nind,Vind,:])
        
        print(np.var(I_sig),np.var(In[Nind,Vind,:]))    
            

#%%

SNDRa = Var_sig/Var_noi_tot

SNDRa_dB = 10*np.log10(SNDRa)

SNDRa_max = np.max(SNDRa_dB,axis=1)
Volt_argmax = np.argmax(SNDRa_dB,axis=1)

np.save('SNDRa_dB_MRAM_vs_N.npy',SNDRa_dB)
#%% 
plt.rc('font', size=22, weight='regular') 

lin_w = 5

fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(N_all*0.5, SNDRa_max,c="r", markersize=10, marker='o', linewidth=lin_w)
plt.ylabel(r'$\mathrm{SNDR}_a$',fontsize=32)
plt.xlabel(r'$N$',fontsize=32,fontdict=font)

ax1.grid(1,'major', linewidth=0.5, color='black')
ax1.grid(1,'minor', linewidth=0.5, ls='--')
plt.xscale("log")
ax1.set_axisbelow(True)
plt.box(1)
#plt.savefig('MRAM_SNDRa_max_vs_N.pdf', bbox_inches='tight',dpi=200)
