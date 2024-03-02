#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:53:33 2022

@author: saionroy
"""
#Selected input pattern SNR computation with mid-code callibration
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from numpy import genfromtxt
import copy
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    #return array[idx]

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def Qunatize(x,clpmax,clpmin,BADC):
    xclp = np.minimum(np.maximum(x,clpmin),clpmax) - clpmin
    lim = (clpmax-clpmin)/(2**BADC-2)
    val = np.round(xclp/lim)
    return clpmin + val*lim, 2*val-64

def DACval(x,clpmax,clpmin,BADC):
    val = (x+64)/2
    lim = (clpmax-clpmin)/(2**BADC-2)
    return clpmin + val*lim

def par(a,b):
    return a*b/(a+b)

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
input_patterns = np.load('Jan25_ADC_SNDR_input_pattern_onlyN.npy')
input_patterns = input_patterns.reshape([64*30,64])
input_patterns = input_patterns.T
input_patterns[input_patterns==0] = -1

#%%

b_w = 4
N = 64
Narr = N*2
np.random.seed(11279)

#Pbar Parameters
g_on = 1/(4.5e3)    #1/Ohm
g_off = 1/(7.5e3)   #1/Ohm
V_ref = 20e-3
m = 30
outC = 10

#Design Parameters
A = 1000
gm = 100e-6
Sig_mb_on = 0.04*g_on
Sig_mb_off = 0.04*g_off
rbl = 183/512
rsl = 84/512
Sig_m = 0.05*np.sqrt(m)
Sig_Ith = 1.26e-6/1.75

#A = 10000
#gm = 200e-3
#Sig_mb_on = 0
#Sig_mb_off = 0
#rbl = 0
#rsl = 0
#Sig_m = 0
#Sig_Ith = 0
zeta = 3
B_ADC = 6

#Full Range Quantization
IA_Clp_min_fr = (V_ref*N*g_off*(2**b_w-1))/m
IA_Clp_max_fr = (V_ref*N*g_on*(2**b_w-1))/m
lim_fr = (IA_Clp_max_fr-IA_Clp_min_fr)/(2**B_ADC)

#Spatial NCse Sources
g_on_noise = Sig_mb_on*np.random.randn(outC, Narr, b_w)
g_off_noise = Sig_mb_off*np.random.randn(outC, Narr, b_w)
current_mismatch = Sig_m*np.random.randn(outC, b_w)

#Calibration Parameters
NOI = N
NC = 30
times = 10

output_ideal_c = 2*np.arange(N) - N
ideal_code = output_ideal_c
weight_c = np.ones([outC, N, b_w])
input_c = np.zeros([NC, N, NOI, times])
g_c_bin = np.ones([outC, Narr, b_w])
input_c_bin = np.zeros([NC, Narr, NOI, times])
for i in range(NOI):
    for j in range(NC):
        for k in range(times):
            input_c[j,:i,i,k] = np.ones([i])
            input_c[j,:,i,k] = input_c[j,np.random.permutation(N),i,k]
input_c[input_c==0] = -1
        
g_c_bin[np.where(weight_c<0)[0],np.where(weight_c<0)[1]*2,np.where(weight_c<0)[2]] = g_off 
g_c_bin[np.where(weight_c<0)[0],np.where(weight_c<0)[1]*2+1,np.where(weight_c<0)[2]] = g_on 
g_c_bin[np.where(weight_c>0)[0],np.where(weight_c>0)[1]*2,np.where(weight_c>0)[2]] = g_on 
g_c_bin[np.where(weight_c>0)[0],np.where(weight_c>0)[1]*2+1,np.where(weight_c>0)[2]] = g_off 
#pdb.set_trace()
np.where(g_c_bin==g_on,g_c_bin+g_on_noise, g_c_bin+g_off_noise)
r_c_bin = 1/g_c_bin

input_c_bin[np.where(input_c<0)[0],np.where(input_c<0)[1]*2,np.where(input_c<0)[2],np.where(input_c<0)[3]] = 0
input_c_bin[np.where(input_c<0)[0],np.where(input_c<0)[1]*2+1,np.where(input_c<0)[2],np.where(input_c<0)[3]] = 1
input_c_bin[np.where(input_c>0)[0],np.where(input_c>0)[1]*2,np.where(input_c>0)[2],np.where(input_c>0)[3]] = 1
input_c_bin[np.where(input_c>0)[0],np.where(input_c>0)[1]*2+1,np.where(input_c>0)[2],np.where(input_c>0)[3]] = 0
input_c_bin = input_c_bin.transpose([0,2,3,1])
#matrix multiplication
y_mult_c = np.zeros([NC, NOI, times, Narr, outC, b_w])

for oind in range(outC):
    for wind in range(b_w):
        y_mult_c[:,:,:,:,oind,wind] = np.multiply(input_c_bin,r_c_bin[oind,:,wind])

y_mult_c[np.where(y_mult_c==0)] = 1e9

R_temp_wdg_c = y_mult_c[:,:,:,Narr-1,:,:]
for r in range(Narr-1,0,-1):
    R_eq_wdg_c = par(rbl + R_temp_wdg_c + rsl, y_mult_c[:,:,:,r-1,:,:])
    R_temp_wdg_c = R_eq_wdg_c

I_th_c = Sig_Ith*np.random.randn(NC, NOI, times, outC)

Isig_wpar_wl_c = (V_ref*((A*gm*R_temp_wdg_c)/(1+A*gm*R_temp_wdg_c))*1/R_temp_wdg_c)
Isig_wpar_wl_wm_c = Isig_wpar_wl_c*1/(m+current_mismatch)    
Isig_wpar_total_c = np.zeros([NC, NOI, times, outC])

#Clipped Range Quantization
#IA_Clp_min_clp = (V_ref*1/np.max(R_temp_wdg_c)*(2**b_w-1))/m
#IA_Clp_max_clp = (V_ref*1/np.min(R_temp_wdg_c)*(2**b_w-1))/m

#Dot-product computation with N_map
for j in range(b_w):
    Isig_wpar_total_c = Isig_wpar_total_c + Isig_wpar_wl_wm_c[:,:,:,:,j]*2**j 
    
I_mu = np.mean(Isig_wpar_total_c)
I_std = np.std(Isig_wpar_total_c)
IA_Clp_min = I_mu - zeta*I_std
IA_Clp_max = I_mu + zeta*I_std
lim_clp = (IA_Clp_max-IA_Clp_min)/(2**B_ADC-2)

#xclp_c = np.clip(Isig_wpar_total_c + I_th_c, min=IA_Clp_min_fr, max=IA_Clp_max_fr)-IA_Clp_min_fr*np.ones(Isig_wpar_total_c.shape)
#norm_dp = (np.round(xclp_c/lim_fr)*2-(2**B_ADC))/(2**B_ADC)

xclp_c = np.clip(Isig_wpar_total_c + I_th_c, IA_Clp_min, IA_Clp_max)-IA_Clp_min*np.ones(Isig_wpar_total_c.shape)
#norm_dp = (np.round(xclp_c/lim_clp)*2-(2**B_ADC))/(2**B_ADC)
#out_sim_c = np.multiply(norm_dp,N)

out_sim_c = np.round(xclp_c/lim_clp)*2-(2**B_ADC)

#MMSE offset and gain parameters
P_Y = np.zeros(N)
pxpw = 0.5
#Binomial Probability Calculation
for k in range(N):
    P_Y[k] = ncr(N, k)*(pxpw**k)*((1-pxpw)**(N-k))

P_Y = P_Y/np.sum(P_Y)
Var_Y = np.sum(output_ideal_c**2*P_Y) - np.sum(output_ideal_c*P_Y)**2

code_offset = np.zeros(outC)
code_gain = np.zeros(outC)
Cross_E = np.zeros(outC)
Mean_Measured = np.zeros(outC)
Moment2nd_Measured = np.zeros(outC)
Var_Measured = np.zeros(outC)

Mean_ideal = np.sum(P_Y*output_ideal_c)
for i in range(outC):
    for j in range(NC):
        for k in range(times):
            Cross_E[i] = Cross_E[i] + np.sum((1/NC)*P_Y*(1/times)*(np.multiply(out_sim_c[j,:,k,i],output_ideal_c)))
            Mean_Measured[i] = Mean_Measured[i] + np.sum((1/NC)*P_Y*(1/times)*out_sim_c[j,:,k,i])
            Moment2nd_Measured[i] = Moment2nd_Measured[i] + np.sum((1/NC)*P_Y*(1/times)*out_sim_c[j,:,k,i]**2)

    Var_Measured[i] = Moment2nd_Measured[i] - Mean_Measured[i]**2

    if(Var_Measured[i]!=0):
        code_offset[i] = (Moment2nd_Measured[i]*Mean_ideal-Cross_E[i]*Mean_Measured[i])/Var_Measured[i]
        code_gain[i] = (Cross_E[i]-Mean_ideal*Mean_Measured[i])/Var_Measured[i]

transformed_code = np.zeros([NC, NOI, times, outC])
for i in range(outC):
    transformed_code[:,:,:,i] = out_sim_c[:,:,:,i]*code_gain[i] + code_offset[i]

transformed_code[transformed_code>N-1]=N-1
transformed_code[transformed_code<-N]=-N

Error_temporal_bev = np.zeros([NC, NOI, outC])
for k in range(NOI):
    for i in range(outC):
        for j in range(NC):
            if(rbl==0 and rsl==0):
                Error_temporal_bev[j,k,i] = np.sum(np.square(np.mean(transformed_code[j,k,:,i]) - output_ideal_c[k]))
            else:
                Error_temporal_bev[j,k,i] = np.sum(np.square(transformed_code[j,k,:,i] - output_ideal_c[k]))/(times)
    
MSE_code = np.zeros([NOI,outC])
for i in range(NOI):
    for j in range(outC):
        MSE_code[i,j] = np.sum(Error_temporal_bev[:,i,j]*1/NC)

SNR_ADC_bev = np.zeros(outC)
for i in range(outC):
    SNR_ADC_bev[i] = 10*np.log10(Var_Y/(np.sum(MSE_code[:,i]*P_Y)))
print(SNR_ADC_bev)

transformed_code_bev = transformed_code
#%%
        
ADC_npz_codeset = np.load('Jan25_N64_8p3MHz_ADC64-72_SNDR_data.npz')
N=64
GRID_ALPHA = 0.5
FIG_SIZE = (11,8)
adc_read_codeset = ADC_npz_codeset[ADC_npz_codeset.files[0]]*2-N

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

colors = 'bgrckymalp'
marker = 'odos'
marker_size = 100 

NOAE = 8
NOI = 64
times = 20

ADC_ideal = np.arange(64)*2-N
Mean_ideal = np.mean(ADC_ideal)

#%%
#Finding Offset and Gain Correction factors

ADC_selected = copy.deepcopy(adc_read_codeset)
ADC_temp = copy.deepcopy(adc_read_codeset)

#%%

Error_temporal = np.zeros([NOAE,N,NC])

code_offset = np.zeros([NOAE])
code_gain = np.zeros([NOAE])
Cross_E = np.zeros(NOAE)
Mean_Measured = np.zeros(NOAE)
Moment2nd_Measured = np.zeros(NOAE)
Var_Measured = np.zeros(NOAE)

Mean_ideal = np.sum(P_Y*ideal_code)
code_measured = adc_read_codeset
times_calib = 10

for i in range(NOAE):
    for j in range(NC):
        for k in range(times_calib):
            Cross_E[i] = Cross_E[i] + np.sum((1/NC)*P_Y*(1/times_calib)*(np.multiply(code_measured[i,:,j,k],ideal_code)))
            Mean_Measured[i] = Mean_Measured[i] + np.sum((1/NC)*P_Y*(1/times_calib)*code_measured[i,:,j,k])
            Moment2nd_Measured[i] = Moment2nd_Measured[i] + np.sum((1/NC)*P_Y*(1/times_calib)*code_measured[i,:,j,k]**2)
        
    Var_Measured[i] = Moment2nd_Measured[i] - Mean_Measured[i]**2
    
    if(Var_Measured[i]!=0):
        code_offset[i] = (Moment2nd_Measured[i]*Mean_ideal-Cross_E[i]*Mean_Measured[i])/Var_Measured[i]
        code_gain[i] = (Cross_E[i]-Mean_ideal*Mean_Measured[i])/Var_Measured[i]


transformed_code = np.zeros([NOAE, N, NC, times])

for k in range(N):
    for i in range(NOAE):
        for j in range(NC):
            transformed_code[i,k,j,:] = (adc_read_codeset[i,k,j,:])*code_gain[i] + code_offset[i]

transformed_code[transformed_code>N-1]=N-1
transformed_code[transformed_code<-N]=-N

for k in range(N):
    for i in range(NOAE):
        for j in range(NC):
            Error_temporal[i,k,j] = np.sum(np.square(transformed_code[i,k,j,times_calib:] - ideal_code[k]))/(times-times_calib)           
            
MSE_code = np.zeros([N,NOAE])
for i in range(N):
    for j in range(NOAE):
        MSE_code[i,j] = np.sum(Error_temporal[j,i,:]*1/NC)

#print(MSE_code)

SNR_ADC = np.zeros(NOAE)
for i in range(NOAE):
    SNR_ADC[i] = 10*np.log10(Var_Y/(np.sum(MSE_code[:,i]*P_Y)))

np.set_printoptions(suppress=True)
print(SNR_ADC)

SNR_IMC_inv = 0
for i in range(NOAE):
    SNR_IMC_inv = SNR_IMC_inv + (1/(10**(SNR_ADC[i]/10)))*1/NOAE
    
SNR_IMC = 10*np.log10(1/SNR_IMC_inv)
print(SNR_IMC)



#%%
FIG_SIZE = (9,8)
loc_idx = 3

#Error bar plot
times_calib = 10

ADC_mean = []
ADC_std = []

ADC_raw = np.reshape(transformed_code[loc_idx,:,:,times_calib:],64*NC*(times-times_calib))
for i in range(NOI):
    ADC_mean.append(np.mean(np.reshape(transformed_code[loc_idx,i,:,times_calib:],NC*(times-times_calib))))
    ADC_std.append(np.std(np.reshape(transformed_code[loc_idx,i,:,times_calib:],NC*(times-times_calib))))
    #ADC_mean.append(np.mean(np.reshape(adc_read_codeset[loc_idx,i,:,times_calib:],NC*(times-times_calib))))
    #ADC_std.append(np.std(np.reshape(adc_read_codeset[loc_idx,i,:,times_calib:],NC*(times-times_calib))))
   

ADC_mean_bev = []
ADC_std_bev = []
ADC_bev = np.reshape(transformed_code_bev[:,:,:,loc_idx],64*NC*(10))
Var_noi_comp = np.var(ADC_raw-ADC_bev)
SNR_comp_dB = 10*np.log10(np.var(ideal_code)/Var_noi_comp)

print(SNR_comp_dB)

for i in range(NOI):
    ADC_mean_bev.append(np.mean(np.reshape(transformed_code_bev[:,i,:,loc_idx],NC*(10))))
    ADC_std_bev.append(np.std(np.reshape(transformed_code_bev[:,i,:,loc_idx],NC*(10))))
    #ADC_mean_bev.append(np.mean(np.reshape(Dig_val[i,:,times_calib:],NC*(times-times_calib))))
    #ADC_std_bev.append(np.std(np.reshape(Dig_val[i,:,times_calib:],NC*(times-times_calib))))


fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
plt.yticks(fontname = "Arial",fontsize=24) 
plt.xticks(fontname = "Arial",fontsize=24) 
ax.set_xlabel(r'Ideal Dot-Product', fontsize = 32)
ax.set_ylabel(r"Measured/ Behavioral" "\n" "Dot-Product", fontsize = 32)
ax.plot(ideal_code, ideal_code, color='black',linewidth=4,label=r'ideal ($y_o$)')
ax.errorbar(ideal_code, ADC_mean, ADC_std,color='blue',ecolor='red',linewidth=4,elinewidth=2,label=r'measured ($\hat{y}$)')
ax.errorbar(ideal_code, ADC_mean_bev, ADC_std_bev,color='green',ecolor='darkorange',linewidth=4,elinewidth=2,label=r'behavioral ($\tilde{y}$)')
plt.legend(loc='upper left', ncol=1,prop={'size': 24, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',handlelength=1.0,handletextpad=1)
fig.savefig('ADC_errorbar_measured_vs_bev.pdf', bbox_inches = "tight",dpi=500)


