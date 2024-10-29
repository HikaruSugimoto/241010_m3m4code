import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from scipy import interpolate
from scipy.optimize import differential_evolution
import math
from scipy.integrate import odeint
import statsmodels.api as sm
from scipy.integrate import solve_ivp
from scipy import stats as st
from scipy.optimize import curve_fit
import sympy as sym
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.subplot.bottom'] = 0
plt.rcParams['figure.subplot.left'] = 0
plt.rcParams['figure.subplot.top'] = 1
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['figure.dpi'] = 300

df_K=pd.read_csv('241010Estimated_parameter.csv')
df = pd.read_csv("241010m3m4_sti.csv")

Type="Ln"
color='blue'
times = ([0, 15,30,60,120,180])
t_span = [0,180]
t_eval = np.linspace(*t_span,180)

#M4
G = interpolate.interp1d(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0],kind='linear')
V = interpolate.interp1d(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0],kind='linear')
init=[0,0,0]

def glu(t,XY, k1,k2,k3,k4,k5,k6):
    I,A,R= XY
    dI=k1*(G(t)-df[df['ID']==Type+'M4']['Gc0'].values[0])-k2*I+A-R
    dA=k3*np.log(V(t)/100)-k4*A
    dR=k5*np.max([np.log(V(t)/100),0])-k6*R
    return [dI,dA,dR]
sol = solve_ivp(glu,t_span,init,method='Radau',t_eval=t_eval,
                args=(df_K[df_K['LnOB']==Type]["k1"].values[0],
                      df_K[df_K['LnOB']==Type]["k2"].values[0],
                      df_K[df_K['LnOB']==Type]["k3"].values[0],
                      df_K[df_K['LnOB']==Type]["k4"].values[0],
                      df_K[df_K['LnOB']==Type]["k5"].values[0],
                      df_K[df_K['LnOB']==Type]["k6"].values[0]))
I4,A4,R4= sol.y    

fig, ax = plt.subplots(1,5,figsize=(11,2))
ax[0].scatter(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].plot(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].set_xlim([-18, 198]) 
ax[0].set_xticks([0,60,120,180])
ax[0].set_ylim([-60, 660]) 
ax[0].set_yticks([0,200,400,600])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Gc')

ax[1].plot(t_eval, df[df['ID']==Type+'M4']['I0'].values[0]+I4, linestyle="solid", color = color)
ax[1].scatter(times, df[df['ID']==Type+'M4'][['I0','I15','I30','I60','I120','I180']].values[0], c=color)
ax[1].set_xlim([-18, 198]) 
ax[1].set_xticks([0,60,120,180])
ax[1].set_ylim([-0.04, 0.44]) 
ax[1].set_yticks([0,0.2,0.4])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('I')

ax[2].scatter(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].plot(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].set_xlim([-18, 198]) 
ax[2].set_xticks([0,60,120,180])
ax[2].set_ylim([-12, 132]) 
ax[2].set_yticks([0,40,80,120])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('cVNA')

ax[3].plot(t_eval, A4, c=color)
ax[3].set_xlim([-18, 198]) 
ax[3].set_xticks([0,60,120,180])
ax[3].set_ylim([-44, 4]) 
ax[3].set_yticks([-40,-20,0])
ax[3].set_xlabel('Time')
ax[3].set_ylabel('A')

ax[4].plot(t_eval, R4, c=color)
ax[4].set_xlim([-18, 198]) 
ax[4].set_xticks([0,60,120,180])
ax[4].set_ylim([-0.12,0.12]) 
ax[4].set_yticks([-0.1,0,0.1])
ax[4].set_xlabel('Time')
ax[4].set_ylabel('R')
fig.tight_layout()  
plt.show()

#M3
G = interpolate.interp1d(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0],kind='linear')
V = interpolate.interp1d(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0],kind='linear')
init=[0,0,0]

def glu(t,XY, k1,k2,k3,k4,k5,k6):
    I,A,R= XY
    dI=k1*(G(t)-df[df['ID']==Type+'M3']['Gc0'].values[0])-k2*I+A-R
    dA=k3*np.log(V(t)/100)-k4*A
    dR=k5*np.max([np.log(V(t)/100),0])-k6*R
    return [dI,dA,dR]
sol = solve_ivp(glu,t_span,init,method='Radau',t_eval=t_eval,
                args=(df_K[df_K['LnOB']==Type]["k1"].values[0],
                      df_K[df_K['LnOB']==Type]["k2"].values[0],
                      df_K[df_K['LnOB']==Type]["k3"].values[0],
                      df_K[df_K['LnOB']==Type]["k4"].values[0],
                      df_K[df_K['LnOB']==Type]["k5"].values[0],
                      df_K[df_K['LnOB']==Type]["k6"].values[0]))
I4,A4,R4= sol.y    

fig, ax = plt.subplots(1,5,figsize=(11,2))
ax[0].scatter(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].plot(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].set_xlim([-18, 198]) 
ax[0].set_xticks([0,60,120,180])
ax[0].set_ylim([-60, 660]) 
ax[0].set_yticks([0,200,400,600])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Gc')

ax[1].plot(t_eval, df[df['ID']==Type+'M3']['I0'].values[0]+I4, linestyle="solid", color = color)
ax[1].scatter(times, df[df['ID']==Type+'M3'][['I0','I15','I30','I60','I120','I180']].values[0], c=color)
ax[1].set_xlim([-18, 198]) 
ax[1].set_xticks([0,60,120,180])
ax[1].set_ylim([-0.06, 0.66]) 
ax[1].set_yticks([0,0.3,0.6])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('I')

ax[2].scatter(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].plot(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].set_xlim([-18, 198]) 
ax[2].set_xticks([0,60,120,180])
ax[2].set_ylim([-60, 660]) 
ax[2].set_yticks([0,200,400,600])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('cVNA')

ax[3].plot(t_eval, A4, c=color)
ax[3].set_xlim([-18, 198]) 
ax[3].set_xticks([0,60,120,180])
ax[3].set_ylim([-4, 44]) 
ax[3].set_yticks([0,20,40])
ax[3].set_xlabel('Time')
ax[3].set_ylabel('A')

ax[4].plot(t_eval, R4, c=color)
ax[4].set_xlim([-18, 198]) 
ax[4].set_xticks([0,60,120,180])
ax[4].set_ylim([-6,66]) 
ax[4].set_yticks([0,30,60])
ax[4].set_xlabel('Time')
ax[4].set_ylabel('R')
fig.tight_layout()  
plt.show()

Type="OB"
color='red'
times = ([0, 15,30,60,120,180])
t_span = [0,180]
t_eval = np.linspace(*t_span,180)

#M4
G = interpolate.interp1d(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0],kind='linear')
V = interpolate.interp1d(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0],kind='linear')
init=[0,0,0]

def glu(t,XY, k1,k2,k3,k4,k5,k6):
    I,A,R= XY
    dI=k1*(G(t)-df[df['ID']==Type+'M4']['Gc0'].values[0])-k2*I+A-R
    dA=k3*np.log(V(t)/100)-k4*A
    dR=k5*np.max([np.log(V(t)/100),0])-k6*R
    return [dI,dA,dR]
sol = solve_ivp(glu,t_span,init,method='Radau',t_eval=t_eval,
                args=(df_K[df_K['LnOB']==Type]["k1"].values[0],
                      df_K[df_K['LnOB']==Type]["k2"].values[0],
                      df_K[df_K['LnOB']==Type]["k3"].values[0],
                      df_K[df_K['LnOB']==Type]["k4"].values[0],
                      df_K[df_K['LnOB']==Type]["k5"].values[0],
                      df_K[df_K['LnOB']==Type]["k6"].values[0]))
I4,A4,R4= sol.y    

fig, ax = plt.subplots(1,5,figsize=(11,2))
ax[0].scatter(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].plot(times, df[df['ID']==Type+'M4'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].set_xlim([-18, 198]) 
ax[0].set_xticks([0,60,120,180])
ax[0].set_ylim([-60, 660]) 
ax[0].set_yticks([0,200,400,600])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Gc')

ax[1].plot(t_eval, df[df['ID']==Type+'M4']['I0'].values[0]+I4, linestyle="solid", color = color)
ax[1].scatter(times, df[df['ID']==Type+'M4'][['I0','I15','I30','I60','I120','I180']].values[0], c=color)
ax[1].set_xlim([-18, 198]) 
ax[1].set_xticks([0,60,120,180])
ax[1].set_ylim([-0.6, 6.6]) 
ax[1].set_yticks([0,3,6])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('I')

ax[2].scatter(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].plot(times, df[df['ID']==Type+'M4'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].set_xlim([-18, 198]) 
ax[2].set_xticks([0,60,120,180])
ax[2].set_ylim([-12, 132]) 
ax[2].set_yticks([0,40,80,120])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('cVNA')

ax[3].plot(t_eval, A4, c=color)
ax[3].set_xlim([-18, 198]) 
ax[3].set_xticks([0,60,120,180])
ax[3].set_ylim([-2200, 200]) 
ax[3].set_yticks([-2000,-1000,0])
ax[3].set_xlabel('Time')
ax[3].set_ylabel('A')

ax[4].plot(t_eval, R4, c=color)
ax[4].set_xlim([-18, 198]) 
ax[4].set_xticks([0,60,120,180])
ax[4].set_ylim([-0.12,0.12]) 
ax[4].set_yticks([-0.1,0,0.1])
ax[4].set_xlabel('Time')
ax[4].set_ylabel('R')
fig.tight_layout()  
plt.show()

#M3
G = interpolate.interp1d(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0],kind='linear')
V = interpolate.interp1d(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0],kind='linear')
init=[0,0,0]

def glu(t,XY, k1,k2,k3,k4,k5,k6):
    I,A,R= XY
    dI=k1*(G(t)-df[df['ID']==Type+'M3']['Gc0'].values[0])-k2*I+A-R
    dA=k3*np.log(V(t)/100)-k4*A
    dR=k5*np.max([np.log(V(t)/100),0])-k6*R
    return [dI,dA,dR]
sol = solve_ivp(glu,t_span,init,method='Radau',t_eval=t_eval,
                args=(df_K[df_K['LnOB']==Type]["k1"].values[0],
                      df_K[df_K['LnOB']==Type]["k2"].values[0],
                      df_K[df_K['LnOB']==Type]["k3"].values[0],
                      df_K[df_K['LnOB']==Type]["k4"].values[0],
                      df_K[df_K['LnOB']==Type]["k5"].values[0],
                      df_K[df_K['LnOB']==Type]["k6"].values[0]))
I4,A4,R4= sol.y    

fig, ax = plt.subplots(1,5,figsize=(11,2))
ax[0].scatter(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].plot(times, df[df['ID']==Type+'M3'][['Gc0','Gc15','Gc30','Gc60','Gc120','Gc180']].values[0], c=color)
ax[0].set_xlim([-18, 198]) 
ax[0].set_xticks([0,60,120,180])
ax[0].set_ylim([-60, 660]) 
ax[0].set_yticks([0,200,400,600])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Gc')

ax[1].plot(t_eval, df[df['ID']==Type+'M3']['I0'].values[0]+I4, linestyle="solid", color = color)
ax[1].scatter(times, df[df['ID']==Type+'M3'][['I0','I15','I30','I60','I120','I180']].values[0], c=color)
ax[1].set_xlim([-18, 198]) 
ax[1].set_xticks([0,60,120,180])
ax[1].set_ylim([-0.6, 6.6]) 
ax[1].set_yticks([0,3,6])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('I')

ax[2].scatter(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].plot(times, df[df['ID']==Type+'M3'][['V0','V15','V30','V60','V120','V180']].values[0], c=color)
ax[2].set_xlim([-18, 198]) 
ax[2].set_xticks([0,60,120,180])
ax[2].set_ylim([-40, 440]) 
ax[2].set_yticks([0,200,400])
ax[2].set_xlabel('Time')
ax[2].set_ylabel('cVNA')

ax[3].plot(t_eval, A4, c=color)
ax[3].set_xlim([-18, 198]) 
ax[3].set_xticks([0,60,120,180])
ax[3].set_ylim([-400, 4400]) 
ax[3].set_yticks([0,2000,4000])
ax[3].set_xlabel('Time')
ax[3].set_ylabel('A')

ax[4].plot(t_eval, R4, c=color)
ax[4].set_xlim([-18, 198]) 
ax[4].set_xticks([0,60,120,180])
ax[4].set_ylim([-400,4400]) 
ax[4].set_yticks([0,2000,4000])
ax[4].set_xlabel('Time')
ax[4].set_ylabel('R')
fig.tight_layout()  
plt.show()