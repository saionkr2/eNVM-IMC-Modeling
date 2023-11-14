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

def Qfunc(x):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def Qunatize(x,clpmax,clpmin,BADC):
    xclp = np.minimum(np.maximum(x,clpmin),clpmax) - clpmin
    lim = (clpmax-clpmin)/(2**BADC-2)
    val = np.round(xclp/lim)
    return clpmin + val*lim

def par(a,b):
    return a*b/(a+b)
#%%

G_P = 1/(1e6)    #1/Ohm
G_AP = 1/(1e9)   #1/Ohm


VDD = 0.8     # Volts
B_ADC = 6


#MRAM
A = 1500
gm = 200e-6

m = 20
rbl = 183/512 
rsl = 84/512
Sig_Ith = 1.28e-6/m
Sig_mb_gap = 0.04*G_AP
Sig_mb_gp = 0.04*G_P
Sig_m = 0.05
zeta = np.linspace(1,8,20)
# zeta = 3.5*np.ones(20)
zlen = 20

# Sig_Ith = 0e-6
# Sig_m = 0
# Sig_mb_gap = 0
# Sig_mb_gp = 0
# rbl = 0 
# rsl = 0
# A = 15000
# gm = 200e-3
# B_ADC = 60
# zeta = np.linspace(10,10,1)
# zlen = 1

V_bl_all = np.logspace(-2,0.7,25)

#N_all = np.array([128,1024,2048])
#N_all = np.array([64,128,256])
N_all = np.array([256])
#N_all = np.array([50])
a_vec_no = 1000

Var_sig = np.zeros((len(N_all),len(V_bl_all)))

I_mu = np.zeros((len(N_all),len(V_bl_all)))
I_std = np.zeros((len(N_all),len(V_bl_all)))

Clp_min = np.zeros((len(N_all),len(V_bl_all)))
Clp_max = np.zeros((len(N_all),len(V_bl_all)))

Inoi = np.zeros((len(N_all),len(V_bl_all),zlen,a_vec_no))
Itot_quantized = np.zeros((len(N_all),len(V_bl_all),zlen,a_vec_no))

Var_noi_tot = np.zeros((len(N_all),len(V_bl_all),zlen))
Var_noi_OCCtot = np.zeros((len(N_all),len(V_bl_all)))

for Vind in range(len(V_bl_all)):
    
    V_bl_s = V_bl_all[Vind]

    for Nind in range(len(N_all)):
        
        ## array dependent variable setup
        N = N_all[Nind]
        ## Generating G vector for given N
        ran_uni_4w = np.expand_dims(np.random.uniform(size=(int(N/2),)),1)
        ran_w_vec = 2*(ran_uni_4w>0.5)-1
        g_vec = np.zeros([N,1])
        del_g_vec = np.zeros([N,a_vec_no])
        
        for i in range(int(N/2)):
            if(ran_w_vec[i]==-1):
                g_vec[2*i,0] = G_AP
                g_vec[2*i+1,0] = G_P
                del_g_vec[2*i,:] = Sig_mb_gap*np.random.normal(size=(a_vec_no))
                del_g_vec[2*i+1,:] = Sig_mb_gp*np.random.normal(size=(a_vec_no))
            else:
                g_vec[2*i,0] = G_P
                g_vec[2*i+1,0] = G_AP
                del_g_vec[2*i,:] = Sig_mb_gp*np.random.normal(size=(a_vec_no))
                del_g_vec[2*i+1,:] = Sig_mb_gap*np.random.normal(size=(a_vec_no))
                
        ## Generating all input V vectors and their mismatches
        a_mat = np.random.randint(0,2,size=(int(N/2),a_vec_no))*2-1
        a_mat_float = a_mat.astype('float64')
        a_mat_01 = np.zeros([N,a_vec_no]) 
        
        for i in range(int(N/2)):
            for j in range(a_vec_no):
                if(a_mat[i,j]==-1):
                    a_mat_01[2*i,j] = 0
                    a_mat_01[2*i+1,j] = 1
                else:
                    a_mat_01[2*i,j] = 1
                    a_mat_01[2*i+1,j] = 0
                    
        # Element Wise Multiplication
        y_mult_wdg = np.multiply(1/(g_vec+del_g_vec),a_mat_01).T
        
        R_mul_wdg = np.where(y_mult_wdg == 0 , 1e9, y_mult_wdg)
        R_temp_wdg = R_mul_wdg[:,N-1]
        for r in range(N-1,0,-1):
            R_eq_wdg = par(rbl + R_temp_wdg + rsl, R_mul_wdg[:,r-1])
            R_temp_wdg = R_eq_wdg
        
        Nreal = N/2
        Isig =  V_bl_s*np.sum(a_mat_01*g_vec,0)/m
        Isig_wpar_wl = (V_bl_s*((A*gm*R_temp_wdg)/(1+A*gm*R_temp_wdg))*1/R_temp_wdg)
        Isig_wpar_wl_wm = (Isig_wpar_wl)*1/(m+np.sqrt(m)*Sig_m*np.random.normal(size=(a_vec_no))) 
        
        Ith = np.random.normal(0, Sig_Ith, size=(a_vec_no,))
        I_mu[Nind,Vind] = np.mean(Isig_wpar_wl_wm)
        I_std[Nind,Vind] = np.std(Isig_wpar_wl_wm)
        Var_sig[Nind,Vind] = np.var(Isig)
        
        for Zind in range(zlen):
            
            R_lsb = VDD/(2*zeta[Zind]*I_std[Nind,Vind]/(2**B_ADC-2))
                
            I_lsb = VDD/R_lsb
            IA_Clp_min = I_mu[Nind,Vind] - I_lsb*(2**B_ADC-2)/2
            IA_Clp_max = I_mu[Nind,Vind] + I_lsb*(2**B_ADC-2)/2
            
            ## Final signal and noise current
            
            Itot_quantized[Nind,Vind,Zind,:] = Qunatize(Isig_wpar_wl_wm + Ith,IA_Clp_max,IA_Clp_min,B_ADC) 
            Inoi[Nind,Vind,Zind,:] = Itot_quantized[Nind,Vind,Zind,:] - Isig
            Var_noi_tot[Nind,Vind,Zind] = np.var(Inoi[Nind,Vind,Zind,:])
        print(zeta[np.argmin(Var_noi_tot[Nind,Vind,:])])
        Var_noi_OCCtot[Nind,Vind] = np.min(Var_noi_tot[Nind,Vind,:])
#%%

SNR = Var_sig/Var_noi_OCCtot

SNR_dB = 10*np.log10(SNR)

#np.save('SNR_MRAM_vs_Vbl_adc.npy',SNR)
#%% 

plt.close('all')

plt.rc('font', size=22, weight='bold') 

lin_w = 5

fig1 = plt.figure(figsize=(9,8))
ax1 = fig1.add_subplot(1,1,1)

plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 

ax1.plot(V_bl_all*1000, SNR_dB[0,:],c="r", markersize=10, marker='o', linestyle='-', linewidth=lin_w,label='Behavioral')
ax1.grid(1,'major', linewidth=0.5, color='black')
ax1.grid(1,'minor', linewidth=0.5, ls='--')
plt.ylabel(r'$\mathrm{SNR}$ (dB)',fontsize=32,fontdict=font)
plt.xlabel(r'$V_{\mathrm{ref}}$ (mV)',fontsize=32,fontdict=font)

#plt.savefig('MRAM_SNR_vs_Vbl_plot.pdf', bbox_inches='tight', dpi =200)   
ax1.set_axisbelow(True)
plt.box(1)

