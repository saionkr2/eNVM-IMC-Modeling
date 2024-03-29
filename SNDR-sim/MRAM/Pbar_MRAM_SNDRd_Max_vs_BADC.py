# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import csv
import math
import random
from scipy import special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib

plt.rc('font', size=18, weight='bold') 
matplotlib.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "arial"
np.random.seed(2021)
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
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
B_ADC = np.arange(2,11)
zeta = np.linspace(0.2,6,10)
zlen = len(zeta)

V_bl_all = np.logspace(-2,-0.09,25)

N = 64
a_vec_no = 515
w_vec_no = 515

Var_sig = np.zeros((len(B_ADC),len(V_bl_all)))

I_mu = np.zeros((len(B_ADC),len(V_bl_all)))
I_std = np.zeros((len(B_ADC),len(V_bl_all)))

yn = np.zeros((len(B_ADC),len(V_bl_all),zlen,a_vec_no*w_vec_no))
y_bev = np.zeros((len(B_ADC),len(V_bl_all),zlen,a_vec_no*w_vec_no))

Var_noi_tot = np.zeros((len(B_ADC),len(V_bl_all),zlen))
Var_noi_OCCtot = np.zeros((len(B_ADC),len(V_bl_all)))

for Bind in range(len(B_ADC)):
    
    for Vind in range(len(V_bl_all)):
        
        V_bl_s = V_bl_all[Vind]
        ## Generating G vector for given N
        ran_w_bin = np.random.uniform(size=(N,w_vec_no))
        ran_w_ap = 2*(ran_w_bin>0.5)-1
        g_vec = np.zeros([2*N,w_vec_no])
        del_g_vec = np.zeros([2*N,w_vec_no])
        
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
        y_mult_wdg = np.zeros([a_vec_no,2*N,w_vec_no])
        for i in range(w_vec_no):
            y_mult_wdg[:,:,i] = np.multiply(a_mat_bin.T,1/(g_vec[:,i]))
        
        R_mul_wdg = np.where(y_mult_wdg == 0 , 1e9, y_mult_wdg)
        R_temp_wdg = R_mul_wdg[:,2*N-1,:]
        for r in range(2*N-1,0,-1):
            R_eq_wdg = par(rbl + R_temp_wdg + rsl, R_mul_wdg[:,r-1,:])
            R_temp_wdg = R_eq_wdg
        
        R_temp_wdg = R_temp_wdg.reshape(-1)
        Isig_wpar_wl = (V_bl_s*((A*gm*R_temp_wdg)/(1+A*gm*R_temp_wdg))*1/R_temp_wdg)
        Isig_wpar_wl_wm = (Isig_wpar_wl)*1/(m+np.sqrt(m)*Sig_m*np.random.normal(size=(np.shape(Isig_wpar_wl)))) 
        
        Ith = np.random.normal(0, Sig_Ith, size=(np.shape(Isig_wpar_wl)))
        I_mu[Bind,Vind] = np.mean(Isig_wpar_wl_wm)
        I_std[Bind,Vind] = np.std(Isig_wpar_wl_wm)
        Var_sig[Bind,Vind] = np.var(y_ideal)
        
        #Sweep to find the clipping range
        for Zind in range(zlen):
            
            I_lsb = 2*zeta[Zind]*I_std[Bind,Vind]/(2**B_ADC[Bind]-2)
                
            IA_Clp_min = I_mu[Bind,Vind] - I_lsb*(2**B_ADC[Bind]-2)/2
            IA_Clp_max = I_mu[Bind,Vind] + I_lsb*(2**B_ADC[Bind]-2)/2
            
            ## Final noise codes
            y_bev[Bind,Vind,Zind,:] = (Quantize(Isig_wpar_wl_wm + Ith,IA_Clp_max,IA_Clp_min,B_ADC[Bind])/2**B_ADC[Bind])*(zeta[Zind]*np.std(y_ideal))
            yn[Bind,Vind,Zind,:] = y_bev[Bind,Vind,Zind,:] - y_ideal
            Var_noi_tot[Bind,Vind,Zind] = np.var(yn[Bind,Vind,Zind,:])
            
        print('SNDRd optimal clipping at',zeta[np.argmin(Var_noi_tot[Bind,Vind,:])])
        Var_noi_OCCtot[Bind,Vind] = np.min(Var_noi_tot[Bind,Vind,:])
        

#%%

SNDRd = Var_sig/Var_noi_OCCtot

SNDRd_dB = 10*np.log10(SNDRd)

SNDRd_max = np.max(SNDRd_dB,axis=1)
max_idx = np.argmax(SNDRd_dB,axis=1)
for i in range(len(B_ADC)):
    SNDRd_max[i] = SNDRd_dB[i,max_idx[i]]

V_max = V_bl_all[max_idx]
np.save('SNDRd_dB_MRAM_vs_BADC.npy',SNDRd_dB)
#%% 
lin_w = 5

fig1 = plt.figure(figsize=(11,8))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(B_ADC, SNDRd_max,c="r", markersize=10, marker='o', linewidth=lin_w)
ax1.grid(1,'major', linewidth=0.5, color='black')
ax1.grid(1,'minor', linewidth=0.5, ls='--')
ax1.set_axisbelow(True)
plt.box(1)
#plt.savefig('MRAM_SNDRd_max_vs_BADC.pdf', bbox_inches='tight',dpi=200)

