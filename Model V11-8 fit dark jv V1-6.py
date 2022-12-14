# python3
"""
Wed 23 Now 2022

@author: cbogh
@author: 100mW
"""

from pathlib import Path 						# Mac/Windows compatibility
import pandas as pd                             # spreadsheets
import numpy as np                              # arrays
import matplotlib.pyplot as plt                 # plots 
from matplotlib.widgets import Slider, Button   # sliders, buttons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # inset plot
from scipy import interpolate                   # curve interpolation
from scipy.optimize import curve_fit, fsolve    # curve fitting

# Path to file (Mac/Windows/Unix compatible, replace / with : in filename)
files = Path(
    "/Users/arangolab/Library/Mobile Documents/com~apple~CloudDocs/iCloud Data/Masha Helen/Single layer JV data/20220815-2-retest10d-9_8:25:22.txt"
)

# Set global constants here:
A = 0.0121				# device area
k = 8.617333262e-5		# Boltzmann constant
T = 300					# temperature
sweep_up_or_down = 0    # select 0 for upward and 2 for downward

# offset between voltage and energy
d = 0.3

# power law exponent for diode blocking term
bd = 3


# Read file data into DataFrame, remove zeroes, sort
jvcurve = pd.read_csv(files, sep='\t')			              # tab delimited
jvcurve = jvcurve[jvcurve['Voltage [V]']!=0]
jvcurve.sort_values(by = [jvcurve.columns[sweep_up_or_down]], 
                    ascending=True, 
                    inplace=True)
v = jvcurve[jvcurve.columns[sweep_up_or_down]].to_numpy()	 # select columns
j = jvcurve[jvcurve.columns[sweep_up_or_down+1]].to_numpy()	 # second columns

# Creat figure
plt.style.use('dark_background')							#black background
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
axins = inset_axes(axs[1], width="50%", height="50%",
                   bbox_to_anchor=(.1, .45, .6, .5), 
                   bbox_transform=axs[1].transAxes, loc=2)       # inset plot

"""
Horizontal sliders:
"""
spacing = 0.037
axVbi = fig.add_axes([0.6, 25*spacing, 0.32, 0.03])
axVbi.set_title('Preset parameters:')
Vbi_slider = Slider(ax=axVbi,label='Vbi', 
    valmin = 0, 
    valinit = 0.5,     # built-in potential from electrode offset
    valmax = 0.8
)
axVo = fig.add_axes([0.6, 24*spacing, 0.32, 0.03])
Vo_slider = Slider(ax=axVo,label='Vo', 
    valmin = 0.5, 
    valinit = 1.5,     # compensation voltage (at which photocurrent = 0?)
    valmax = 2
)
axwidth = fig.add_axes([0.6, 23*spacing, 0.32, 0.03])
width_slider = Slider(ax=axwidth,label='width', 
    valmin = 0.2, 
    valinit = 0.5,     # curvature of Vrs curve
    valmax = 0.8
)
axRs = fig.add_axes([0.6, 21*spacing, 0.32, 0.03])			  # create slider
axRs.set_title('Series resistance terms:')
Rs_slider = Slider(
    ax=axRs, label='Rs',
    valmin = 50,
    valinit = 86, 	                      # Rs Ohmic series resistance [Ohms]
    valmax = 200
)
axJoTs = fig.add_axes([0.6, 19*spacing, 0.32, 0.03])
axJoTs.set_title('Jseries terms (below $V_0+V_{bi}$):')
JoTs_slider = Slider(
    ax=axJoTs, label='JoTs', 
    valmin = 0.8,
    valinit = 1.5,	         # JoTs Coefficient for series transport [mA/cm2]
    valmax = 3
)
axts = fig.add_axes([0.6, 18*spacing, 0.32, 0.03])
ts_slider = Slider(
    ax=axts, label='ts',
    valmin = 5,
    valinit = 9.6,	             # ts Power law exponent for series transport
    valmax = 12
)
axJoIp = fig.add_axes([0.6, 17*spacing, 0.32, 0.03])
JoIp_slider = Slider(
    ax=axJoIp, label='JoIp',
    valmin = 0.8,
    valinit = 1.5, 	       # JoIp Coefficient for Injection (p-side) [mA/cm2]
    valmax = 3
)
axip = fig.add_axes([0.6, 16*spacing, 0.32, 0.03])
ip_slider = Slider(
    ax=axip, label='ip',
    valmin = 15,
    valinit = 20,		       # ip Power law exponent for injection (p-side)
    valmax = 40
)
axJoBsh = fig.add_axes([0.6, 14*spacing, 0.32, 0.03])
axJoBsh.set_title('Jshunt terms (below $V_{0}$):')
JoBsh_slider = Slider(
    ax=axJoBsh, label='JoBsh',
    valmin = 1,
    valinit = 3.3,	# JoBsh Coefficient for blockling layer (shunt) [mA/cm2]
    valmax = 15
)
axJoTsh = fig.add_axes([0.6, 13*spacing, 0.32, 0.03])
JoTsh_slider = Slider(
    ax=axJoTsh, label='JoTsh',
    valmin = 0.5,
    valinit = 1,		             # JoTsh Coefficient for shunt transport
    valmax = 3
)
axtsh = fig.add_axes([0.6, 12*spacing, 0.32, 0.03])
tsh_slider = Slider(
    ax=axtsh, label='tsh',
    valmin = 4,
    valinit = 7.2,				# tsh Power law exponent for shunt transport
    valmax = 12
)
axJoBd = fig.add_axes([0.6, 10*spacing, 0.32, 0.03])
axJoBd.set_title('Jdiffusion terms (below $V_{bi}$):')
JoBd_slider = Slider(ax=axJoBd,label='JoBd'
    ,valmin = 1e-2 
    ,valinit = 0.083  # JoBd Coefficient for blockling layer (diode) [mA/cm2]
    ,valmax = 1
)
axJo = fig.add_axes([0.6, 9*spacing, 0.32, 0.03])
Jo_slider = Slider(ax=axJo,label='Jo'
    ,valmin = 1e-7
    ,valinit = 1e-4				# Jo Diode saturation current [mA/cm2]
    ,valmax = 1e-3
)
axn = fig.add_axes([0.6, 8*spacing, 0.32, 0.03])
n_slider = Slider(
    ax=axn,label='n', 
    valmin = 1,
    valinit = 2.5,				                          # n Diode ideality
    valmax = 10
)
axb = fig.add_axes([0.6, 6*spacing, 0.32, 0.03])
axb.set_title('Blocking layer exponent:')
b_slider = Slider(ax=axb,label='b', 
    valmin = 0.5, 
    valinit = 2,     # b Power law exponent for blocking layer
    valmax = 4
)
axRsh = fig.add_axes([0.6, 4*spacing, 0.32, 0.03])
axRsh.set_title('Shunt resistance terms:')
Rsh_slider = Slider(
    ax=axRsh, label='Rsh',
    valmin = 1e5,
    valinit = 1e8,				# Rsh Ohmic shunt resistance [Ohms]
    valmax = 1e9
)
axJoffset = fig.add_axes([0.6, 3*spacing, 0.32, 0.03])
Joffset_slider = Slider(ax=axJoffset,label='Joffset', 
    valmin = -1e-4, 
    valinit = 0,     # Joffset Instrumentation dc offset or illumination [mA/cm2]
    valmax = 1e-4
)



# Edit circuit model equations below
model_version = 'Model V11-7' 

# Circuit functions
def Rshunt(x, Rsh, Joffset):
    return x/Rsh/A*1000+Joffset				#Ohmic shunt and dc offset
def Diode(x, Jo, n):
    return Jo*(np.exp(x/n/k/T)-1) 			#ideal diode
def block_Diode(x, JoBd):
    return np.power(JoBd*x/d, bd)			#effect of blocking layer on ideal diode
def Tshunt(x, JoTsh, tsh, Vbi = Vbi_slider.val):
    return np.power(JoTsh*x/(Vbi+d), tsh)	#space charge transport (shunt)
def block_Tshunt(x, JoBsh, b, Vbi = Vbi_slider.val):
    return np.power(JoBsh*x/(Vbi+d), b)	#effect of blocking layer on transport (shunt)
def Inj_p(x, JoIp, ip, Vo = Vo_slider.val):
    return np.power(JoIp*x/(Vo+d), ip)		#Injection from p-type electrode
def Tseries(x, JoTs, ts, Vo = Vo_slider.val):
    return np.power(JoTs*x/(Vo+d), ts)		#space charge transport (series)
def Rs_volt_drop_func(x, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return width*np.log(np.exp((x-Vo-Vbi)/width)+1) #voltage drop across Rs
def Rseries(x, Rs, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return Rs_volt_drop_func(x, width, Vbi, Vo)/Rs/A*1000	#Ohmic series

# Circuit legs
def Jdark_exp(x, Jo, n, JoBd, b):
    return 1/(1/Diode(x, Jo, n) + 1/block_Diode(x, JoBd))				#exponential leg
def Jdark_shunt(x, JoTsh, tsh, JoBsh, b, Vbi = Vbi_slider.val):
    return 1/(1/Tshunt(x, JoTsh, tsh, Vbi) + 1/block_Tshunt(x, JoBsh, b, Vbi))		#transport (shunt) leg
def Jdark_series(x, JoTs, ts, JoIp, ip, Vo = Vo_slider.val):
    return 1/(1/Tseries(x, JoTs, ts, Vo)+1/Inj_p(x, JoIp, ip, Vo))				#transport (series) leg

# Complete circuit solution
def Jdark(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, b, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return 1/(1/(Jdark_exp(x, Jo, n, JoBd, b) 
                + Jdark_shunt(x, JoTsh, tsh, JoBsh, b, Vbi) 
                + Jdark_series(x, JoTs, ts, JoIp, ip, Vo) 
                + Rshunt(x, Rsh, Joffset)
                ) 
            + 1/Rseries(x, Rs, width, Vbi, Vo)
            )


# Voltage drop across Rs
def Rs_volt_drop(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, b, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return Jdark(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, b, width, Vbi, Vo) * A * Rs / 1000

"""
Derivative of log-log Jdark functions
"""

def Slope(x,Rsh,Joffset,Jo,n,JoBd,JoTsh,tsh,JoBsh,JoTs,ts,JoIp,ip, Rs, b, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return np.gradient(np.log10(Jdark(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, b, width, Vbi, Vo)), np.log10(x))
# slope of exponential leg
def Slope_exp(x, Jo, n, JoBd, b):
    return np.gradient(np.log10(Jdark_exp(x, Jo, n, JoBd, b)), np.log10(x))
# slope of transport (shunt) leg
def Slope_shunt(x, JoTsh, tsh, JoBsh, b, Vbi = Vbi_slider.val):
    return np.gradient(np.log10(Jdark_shunt(x, JoTsh, tsh, JoBsh, b, Vbi)), np.log10(x))	
# slope of transport (series) leg
def Slope_series(x, JoTs, ts, JoIp, ip, Vo = Vo_slider.val):
    return np.gradient(np.log10(Jdark_series(x, JoTs, ts, JoIp, ip, Vo)), np.log10(x))
# slope of Rs element
def Slope_Rseries(x, Rs, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
    return np.gradient(np.log10(Rseries(x, Rs, width, Vbi, Vo)), np.log10(x)) 

# Find slope of logj-logv plot 
jvslope = np.gradient(np.log10(j), np.log10(v), edge_order=2)		#derivative of log-log jv data
jvcurve['Slope of log-log j-v curve'] = jvslope		   #append to DataFrame

# Create scatter plot
axs[1].set_xlabel("Voltage [V]")
axs[0].set_ylabel("Current density [mA/cm$^2$]")
axs[1].set_ylabel("Slope of log-log j-v curve")
axins.set_xlabel("V [V]")
axins.set_ylabel("V$_{Rs}$ [V]")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
fig.subplots_adjust(top=0.98, right=0.5, bottom=0.06, left=0.07, hspace=0) 		#adjust plot for room for sliders
axs[0].set_ylim(			# set axis limits
    0.2*j.min(),			# grab min from current data and adjust spacing
    3*j.max()				# grab max from current data and adjust spacing
)
axs[1].set_ylim(			# set axis limits
    1,			            # grab min from current data and adjust spacing
    3*np.nanmax(jvslope)			# grab max from current data and adjust spacing
)
axs[0].scatter(v, j, s=1.3, color='#4d96f9', label = files.name)	#plot current density
axs[0].text(0.8, 0.05,model_version,transform=axs[0].transAxes)		#text box for model version
axs[1].scatter(v,jvslope,s=1.3,color='#4d96f9')						#plot slope

# Add Vbi and Vo to plot
fint_j = interpolate.interp1d(v, j, kind='linear')
Vo_point, = axs[0].plot(Vo_slider.val, fint_j(Vo_slider.val), marker="+")
Vbi_point, = axs[0].plot(Vbi_slider.val, fint_j(Vbi_slider.val), marker="+")
Vo_Vbi_point, = axs[0].plot(Vo_slider.val + Vbi_slider.val, fint_j(Vo_slider.val + Vbi_slider.val), marker="+")
Vo_text = axs[0].text(Vo_slider.val, fint_j(Vo_slider.val), '$V_0$', ha='right')
Vbi_text = axs[0].text(Vbi_slider.val, fint_j(Vbi_slider.val), '$V_{bi}$', ha='right')
Vo_Vbi_text = axs[0].text(Vbi_slider.val + Vo_slider.val, fint_j(Vbi_slider.val+Vo_slider.val), '$V_0+V_{bi}$', ha='left')
fint_jvslope = interpolate.interp1d(v, jvslope, kind='linear')
Vbi_slope_point, = axs[1].plot(Vo_slider.val, fint_jvslope(Vo_slider.val), marker='+')
Vo_slope_point, = axs[1].plot(Vbi_slider.val, fint_jvslope(Vbi_slider.val), marker='+')
Vo_Vbi_slope_point, = axs[1].plot(Vo_slider.val + Vbi_slider.val, fint_jvslope(Vo_slider.val + Vbi_slider.val), marker='+')
Vo_slope_text = axs[1].text(Vo_slider.val, fint_jvslope(Vo_slider.val), '$V_0$', ha='right')
Vbi_slope_text = axs[1].text(Vbi_slider.val, fint_jvslope(Vbi_slider.val), '$V_{bi}$', ha='right')
Vo_Vbi_slope_text = axs[1].text(Vo_slider.val + Vbi_slider.val, fint_jvslope(Vo_slider.val + Vbi_slider.val), '$V_0+V_{bi}$', ha='left')
axins.plot(Vo_slider.val, Rs_volt_drop_func(Vo_slider.val), marker='+')
axins.plot(Vbi_slider.val, Rs_volt_drop_func(Vbi_slider.val), marker='+')
axins.plot(Vo_slider.val + Vbi_slider.val, Rs_volt_drop_func(Vo_slider.val + Vbi_slider.val), marker='+')

# Set initial parameters from sliders
init_Rs = Rs_slider.val
init_JoTs = JoTs_slider.val
init_ts = ts_slider.val
init_JoIp = JoIp_slider.val
init_ip = ip_slider.val
init_JoBsh = JoBsh_slider.val
init_JoTsh = JoTsh_slider.val
init_tsh = tsh_slider.val
init_JoBd = JoBd_slider.val
init_Jo = Jo_slider.val
init_n = n_slider.val
init_Rsh = Rsh_slider.val
init_Joffset = Joffset_slider.val
init_b = b_slider.val

# Add individual circuit elements to plot
Rshunt_line, = axs[0].plot(v, 
                           Rshunt(v,
                                  init_Rsh,
                                  init_Joffset), 
                           color='gray', linewidth=1, linestyle='dashed')
Diode_line, = axs[0].plot(v, 
                          Diode(v,
                                init_Jo,
                                init_n), 
                          color='gray', linewidth=1, linestyle='dotted')
block_Diode_line, = axs[0].plot(v, 
                                block_Diode(v, 
                                            init_JoBd), 
                                color='gray', linewidth=1, linestyle='dotted')
Tshunt_line, = axs[0].plot(v, 
                           Tshunt(v,
                                  init_JoTsh,
                                  init_tsh), 
                           color='gray', linewidth=1, linestyle='dotted')
block_Tshunt_line, = axs[0].plot(v, 
                                 block_Tshunt(v,
                                              init_JoBsh, 
                                              init_b), 
                                 color='gray', linewidth=1, linestyle='dotted')
Inj_p_line, = axs[0].plot(v, 
                          Inj_p(v, 
                                init_JoIp, 
                                init_ip), 
                          color='gray', linewidth=1, linestyle='dotted')
Tseries_line, = axs[0].plot(v,
                            Tseries(v, 
                                    init_JoTs, 
                                    init_ts), 
                            color='gray', linewidth=1, linestyle='dotted')
Rseries_line, = axs[0].plot(v,
                            Rseries(v, 
                                    init_Rs), 
                            color='gray', linewidth=1, linestyle='dashed')	

# Add each circuit leg to plot
Jdark_exp_line, = axs[0].plot(v, 
                              Jdark_exp(v, 
                                        init_Jo, 
                                        init_n, 
                                        init_JoBd, 
                                        init_b
                                        ), color='white', linewidth=0.8)
Jdark_shunt_line, = axs[0].plot(v, 
                                Jdark_shunt(v, 
                                            init_JoTsh, 
                                            init_tsh, 
                                            init_JoBsh,
                                            init_b
                                            ), color='white', linewidth=0.8)
Jdark_series_line, = axs[0].plot(v, Jdark_series(v, init_JoTs, init_ts, init_JoIp, init_ip), color='white', linewidth=0.8)
Slope_exp_line, = axs[1].plot(v, 
                              Slope_exp(v, 
                                        init_Jo, 
                                        init_n, 
                                        init_JoBd, 
                                        init_b
                                        ), color='white', linewidth=0.8)
Slope_shunt_line, = axs[1].plot(v, 
                                Slope_shunt(v, 
                                            init_JoTsh, 
                                            init_tsh, 
                                            init_JoBsh, init_b
                                            ),color='white', linewidth=0.8)
Slope_series_line, = axs[1].plot(v, 
                                 Slope_series(v, 
                                              init_JoTs, 
                                              init_ts, 
                                              init_JoIp,
                                              init_ip
                                              ),color='white', linewidth=0.8)
Slope_Rseries_line, = axs[1].plot(v, 
                                 Slope_Rseries(v, 
                                              init_Rs,
                                              ), color='gray', linewidth=1, linestyle='dashed')
                
init_params = [init_Rsh, 
                init_Joffset, 
                init_Jo, 
                init_n, 
                init_JoBd, 
                init_JoTsh, 
                init_tsh, 
                init_JoBsh, 
                init_JoTs, 
                init_ts, 
                init_JoIp, 
                init_ip, 
                init_Rs,
                init_b,
]

# Add complete circuit solution to plot
Guess_Jdark_line, = axs[0].plot(v, 
                                Jdark(v, *init_params),
                                linewidth=1.2, c='purple', label='Guess')
Guess_Slope_line, = axs[1].plot(v, 
                                Slope(v, *init_params),
                                linewidth=1.2, c='purple')
Fit_Jdark_line, = axs[0].plot(v, 
                              Jdark(v, *init_params),
                              linewidth=1.2, c='orange', label='Fit')
Fit_Slope_line, = axs[1].plot(v, 
                              Slope(v, *init_params),
                              linewidth=1.2, c='orange')

# Add Rs voltage drops to inset
Guess_Rs_volt_drop_line, = axins.plot(v, 
                                      Rs_volt_drop(v,*init_params),
                                      linewidth=0.5, c='purple')
Fit_Rs_volt_drop_line, = axins.plot(v, 
                                    Rs_volt_drop(v, *init_params),
                                    linewidth=0.5, c='orange')
Rs_volt_drop_func_line, = axins.plot(v, 
                                     Rs_volt_drop_func(v, 
                                        width_slider.val,
                                        Vbi_slider.val,
                                        Vo_slider.val), 
                                     linewidth=0.5, c='red')

# Add labels for individual circuit elements
Rs_box = axs[0].text(0.9*v.min(), Rseries(v.min(), init_Rs), '$R_s$')
Rsh_box = axs[0].text(0.95*v.max(), Rshunt(v.max(), init_Rsh, init_Joffset), '$R_{sh}$')

# Add labels for circuit leg lines
jdiff_box = axs[0].text(0.95*v.max(), Jdark_exp(v.max(), 
                                                init_Jo, 
                                                init_n, 
                                                init_JoBd,
                                                init_b
                                                ), '$j_{diff}$')
jshunt_box = axs[0].text(fsolve(lambda x: Jdark_shunt(x, 
                                                      init_JoTsh, 
                                                      init_tsh, 
                                                      init_JoBsh, 
                                                      init_b
                                                      ) - j.min(),[0.5]), j.min(), '$j_{shunt}$')
jseries_box = axs[0].text(fsolve(lambda x: Jdark_series(x,
                                                        init_JoTs,
                                                        init_ts,
                                                        init_JoIp,
                                                        init_ip
                                                        ) - j.min(),[1]), j.min(), '$j_{series}$')
jdiff_slope_box = axs[1].text(0.95*v.max(), Slope_exp(v, 
                                                      init_Jo, 
                                                        init_n, 
                                                        init_JoBd, 
                                                        init_b
                                                      )[-1], '$j_{diff}$')
jshunt_slope_box = axs[1].text(0.85*v.max(), Slope_shunt(v, 
                                                         init_JoTsh, 
                                                        init_tsh, 
                                                        init_JoBsh,
                                                        init_b
                                                        )[-1], '$j_{shunt}$')
jseries_slope_box = axs[1].text(0.85*v.max(), Slope_series(v, 
                                                            init_JoTs, 
                                                            init_ts, 
                                                            init_JoIp, 
                                                            init_ip
                                                           )[-1], '$j_{series}$')

axs[0].legend(loc='upper left', frameon=False)

# The function to be called anytime a slider's value changes
def update(val):
    
    guess_params = [Rsh_slider.val, 
                    Joffset_slider.val, 
                    Jo_slider.val,
                    n_slider.val,
                    JoBd_slider.val, 
                    JoTsh_slider.val, 
                    tsh_slider.val, 
                    JoBsh_slider.val, 
                    JoTs_slider.val, 
                    ts_slider.val, 
                    JoIp_slider.val, 
                    ip_slider.val, 
                    Rs_slider.val, 
                    b_slider.val,
                    width_slider.val,
                    Vbi_slider.val,
                    Vo_slider.val
    ]
    
    # Update circuit element lines
    Rshunt_line.set_ydata(Rshunt(v,
                                  Rsh_slider.val,
                                  Joffset_slider.val)
    )
    Diode_line.set_ydata(Diode(v,
                                Jo_slider.val,
                                n_slider.val)
    )
    block_Diode_line.set_ydata(block_Diode(v,
                                            JoBd_slider.val)
    )
    Tshunt_line.set_ydata(
        Tshunt(v, 
                JoTsh_slider.val, 
                tsh_slider.val,
                Vbi_slider.val)
    )
    block_Tshunt_line.set_ydata(
        block_Tshunt(v, 
                    JoBsh_slider.val, 
                    b_slider.val,
                    Vbi_slider.val)
    )
    Tseries_line.set_ydata(
        Tseries(v, 
                JoTs_slider.val, 
                ts_slider.val,
                Vo_slider.val)
    )
    Inj_p_line.set_ydata(
        Inj_p(v, 
                JoIp_slider.val, 
                ip_slider.val, 
                Vo_slider.val)
    )
    Rseries_line.set_ydata(Rseries(v, 
                            Rs_slider.val, 
                            width_slider.val,
                            Vbi_slider.val, 
                            Vo_slider.val)
    )
    Slope_Rseries_line.set_ydata(Slope_Rseries(v, 
                                                Rs_slider.val,
                                                width_slider.val,
                                                Vbi_slider.val, 
                                                Vo_slider.val)
                                               )
    
    # Update circuit leg lines
    Jdark_exp_line.set_ydata(Jdark_exp(v,
                                       Jo_slider.val,
                                       n_slider.val,
                                       JoBd_slider.val,
                                       b_slider.val
                                       ))
    Jdark_shunt_line.set_ydata(Jdark_shunt(v, 
                                           JoTsh_slider.val, 
                                           tsh_slider.val, 
                                           JoBsh_slider.val, 
                                           b_slider.val, 
                                            Vbi_slider.val
                                           ))
    Jdark_series_line.set_ydata(Jdark_series(v, 
                                             JoTs_slider.val, 
                                            ts_slider.val, 
                                            JoIp_slider.val, 
                                            ip_slider.val,
                                            Vo_slider.val
                                        ))
    Guess_Jdark_line.set_ydata(
                                Jdark(v, *guess_params)
                              )
    Guess_Slope_line.set_ydata(
                                Slope(v, *guess_params)
                              )
    Slope_exp_line.set_ydata(Slope_exp(v,
                                        Jo_slider.val,
                                        n_slider.val,
                                        JoBd_slider.val,
                                        b_slider.val
                                    ))
    Slope_shunt_line.set_ydata(Slope_shunt(v, 
                                           JoTsh_slider.val, 
                                           tsh_slider.val, 
                                           JoBsh_slider.val, 
                                           b_slider.val,
                                            Vbi_slider.val
                                        ))
    Slope_series_line.set_ydata(Slope_series(v, 
                                            JoTs_slider.val, 
                                            ts_slider.val, 
                                            JoIp_slider.val, 
                                            ip_slider.val,
                                            Vo_slider.val
                                        ))
    
    # Update Rs voltage drop lines
    Guess_Rs_volt_drop_line.set_ydata(
                                        Rs_volt_drop(v, *guess_params)
                                    )
    
    # Update text box postions
    Rs_box.set_position((0.9*v.min(), Rseries(v.min(), Rs_slider.val), '$R_s$'))
    Rsh_box.set_position((0.95*v.max(), Rshunt(v.max(), Rsh_slider.val, Joffset_slider.val), '$R_{sh}$'))
    jdiff_box.set_position((0.8*v.max(), Jdark_exp(v.max(),
                                                   Jo_slider.val,
                                                   n_slider.val,
                                                   JoBd_slider.val, 
                                                   b_slider.val
                                                   )))
    jshunt_box.set_position((fsolve(lambda x: Jdark_shunt(x,
                                                          JoTsh_slider.val,
                                                          tsh_slider.val,
                                                          JoBsh_slider.val, 
                                                          b_slider.val, 
                                                            Vbi_slider.val
                                                          ) - j.min(),[0.5]), j.min()))
    jseries_box.set_position((fsolve(lambda x: Jdark_series(x,
                                                            JoTs_slider.val,
                                                            ts_slider.val,
                                                            JoIp_slider.val,
                                                            ip_slider.val,
                                                            Vo_slider.val
                                                            ) - j.min(),[1]), j.min()))
    jdiff_slope_box.set_position((0.95*v.max(), Slope_exp(v,
                                                            Jo_slider.val,
                                                            n_slider.val,
                                                            JoBd_slider.val,
                                                            b_slider.val
                                                           )[-1]))
    jshunt_slope_box.set_position((0.85*v.max(), Slope_shunt(v,
                                                          JoTsh_slider.val,
                                                          tsh_slider.val,
                                                          JoBsh_slider.val, 
                                                          b_slider.val,
                                                            Vbi_slider.val
                                                            )[-1]))
    jseries_slope_box.set_position((0.85*v.max(), Slope_series(v,
                                                                JoTs_slider.val,
                                                                ts_slider.val,
                                                                JoIp_slider.val,
                                                                ip_slider.val, 
                                                                Vo_slider.val
                                                            )[-1]))
    
    # update Vo & Vbi points
    Vo_point.set_data(Vo_slider.val, fint_j(Vo_slider.val))
    Vbi_point.set_data(Vbi_slider.val, fint_j(Vbi_slider.val))
    Vo_Vbi_point.set_data(Vo_slider.val + Vbi_slider.val, fint_j(Vo_slider.val + Vbi_slider.val))
    Vo_text.set_position((Vo_slider.val, fint_j(Vo_slider.val)))
    Vbi_text.set_position((Vbi_slider.val, fint_j(Vbi_slider.val)))
    Vo_Vbi_text.set_position((Vbi_slider.val + Vo_slider.val, fint_j(Vbi_slider.val+Vo_slider.val)))
    Vbi_slope_point.set_data(Vo_slider.val, fint_jvslope(Vo_slider.val))
    Vo_slope_point.set_data(Vbi_slider.val, fint_jvslope(Vbi_slider.val))
    Vo_Vbi_slope_point.set_data(Vo_slider.val + Vbi_slider.val, fint_jvslope(Vo_slider.val + Vbi_slider.val))
    Vo_slope_text.set_position((Vo_slider.val, fint_jvslope(Vo_slider.val)))
    Vbi_slope_text.set_position((Vbi_slider.val, fint_jvslope(Vbi_slider.val)))
    Vo_Vbi_slope_text.set_position((Vo_slider.val + Vbi_slider.val, fint_jvslope(Vo_slider.val + Vbi_slider.val)))
    
    fig.canvas.draw_idle()
    
# register the update function with each slider
Rsh_slider.on_changed(update)
Joffset_slider.on_changed(update)
Jo_slider.on_changed(update)
n_slider.on_changed(update)
JoBd_slider.on_changed(update)
JoTsh_slider.on_changed(update)
tsh_slider.on_changed(update)
JoBsh_slider.on_changed(update)
JoTs_slider.on_changed(update)
ts_slider.on_changed(update)
JoIp_slider.on_changed(update)
ip_slider.on_changed(update)
Rs_slider.on_changed(update)
b_slider.on_changed(update)
width_slider.on_changed(update)
Vbi_slider.on_changed(update)
Vo_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to fit, accept fit, save
fitax = fig.add_axes([0.6, 0.025, 0.1, 0.04])
button1 = Button(fitax, 'Fit', color='orange')
acceptax = fig.add_axes([0.7, 0.025, 0.1, 0.04])
button2 = Button(acceptax, 'Accept', color='orange')
saveax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button3 = Button(saveax, 'Save', color='orange')

# Function that activates when Fit button is pressed
def fit(event):
    
    def Log_Jdark(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh , JoBsh, JoTs, ts, JoIp, ip, Rs, b, width = width_slider.val, Vbi = Vbi_slider.val, Vo = Vo_slider.val):
        return np.log10(Jdark(x, Rsh, Joffset, Jo, n, JoBd, JoTsh, tsh, JoBsh, JoTs, ts, JoIp, ip, Rs, b, width, Vbi, Vo))
    
    #use slider values as guess values for fit
    init_fit_params = np.array([
        Rsh_slider.val, 
        Jo_slider.val, 
        Jo_slider.val,
        n_slider.val,
        JoBd_slider.val, 
        JoTsh_slider.val, 
        tsh_slider.val, 
        JoBsh_slider.val, 
        JoTs_slider.val, 
        ts_slider.val, 
        JoIp_slider.val, 
        ip_slider.val, 
        Rs_slider.val,
        b_slider.val
    ])
    
    # Fit log of jvcurve
    fit_params, pcov = curve_fit(Log_Jdark, 
                                v, 
                                np.log10(j), 
                                p0=init_fit_params, 
                                method='lm'
    )
    #plot fit
    Fit_Jdark_line.set_ydata(Jdark(v, 
                                    *fit_params, 
                                    width_slider.val, 
                                    Vbi_slider.val, 
                                    Vo_slider.val)
    )
    #plot slope of fit
    Fit_Slope_line.set_ydata(Slope(v, 
                                    *fit_params,
                                    width_slider.val, 
                                    Vbi_slider.val, 
                                    Vo_slider.val)
    )
    #standard deviation of errors
    perr = np.sqrt(np.diag(pcov))
    
    #show fit results
    fit_results = pd.DataFrame(index = ['Rsh', 'Joffset', 'Jo', 'n', 'JoBd', 'JoTsh', 'tsh', 'JoBsh', 'JoTs', 'ts', 'JoIp', 'ip', 'Rs','b'])
    fit_results['Value'] = fit_params
    fit_results['Error'] = perr
    print(fit_results)
    
    fig.canvas.draw_idle()
    
    # Function that activates when Accept button is pressed,
    def accept(event):
        
        # Reset slider values to fit parameters
        Rsh_slider.set_val(fit_params[0])
        Joffset_slider.set_val(fit_params[1])
        Jo_slider.set_val(fit_params[2])
        n_slider.set_val(fit_params[3])
        JoBd_slider.set_val(fit_params[4])
        JoTsh_slider.set_val(fit_params[5])
        tsh_slider.set_val(fit_params[6])
        JoBsh_slider.set_val(fit_params[7])
        JoTs_slider.set_val(fit_params[8])
        ts_slider.set_val(fit_params[9])
        JoIp_slider.set_val(fit_params[10])
        ip_slider.set_val(fit_params[11])
        Rs_slider.set_val(fit_params[12])
        b_slider.set_val(fit_params[13])
        
        # Reset values for inset (bug: not resetting automatically)
        Fit_Rs_volt_drop_line.set_ydata(
            Rs_volt_drop(v, 
                        Rsh_slider.val, 
                        Joffset_slider.val, 
                        Jo_slider.val,
                        n_slider.val,
                        JoBd_slider.val, 
                        JoTsh_slider.val, 
                        tsh_slider.val, 
                        JoBsh_slider.val, 
                        JoTs_slider.val, 
                        ts_slider.val, 
                        JoIp_slider.val, 
                        ip_slider.val, 
                        Rs_slider.val,
                        b_slider.val,
                        width_slider.val,
                        Vbi_slider.val,
                        Vo_slider.val)
        )
        Rs_volt_drop_func_line.set_ydata(
            Rs_volt_drop_func(v, 
                                width_slider.val, 
                                Vbi_slider.val,
                                Vo_slider.val)
        )
        
        fig.canvas.draw_idle()
        
        # Function that activates when Save button is pressed
        def save(event):
            folderpath = files.parent / (files.stem + ' results')			#path of new folder for results
            folderpath.mkdir(parents=True,exist_ok=True)					#make new folder
            fit_results.to_csv(folderpath/'Fit parameters.txt',sep='\t')	#save fit parameters
            jvcurve.to_csv(folderpath/'JV curve.txt',sep='\t')				#save curve
            plt.savefig(folderpath/'Plot.pdf')								#save plot
            print('Results Saved')
            
        button3.on_clicked(save)
    button2.on_clicked(accept)
button1.on_clicked(fit)

print(jvcurve)
#print(Slope_exp(v, init_Jo, init_n, init_JoBd)[-1])
#print(Slope_shunt(v, init_JoTsh, init_tsh, init_JoTsh))
#print(Slope_series(v,JoTs_slider.val,ts_slider.val,JoIp_slider.val,ip_slider.val)[-1])

plt.show()
