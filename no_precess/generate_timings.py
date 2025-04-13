import numpy as np, cupy as cp, os, sys, ast
import matplotlib.pyplot as plt
import kerrgeopy as kg


sma = float(sys.argv[1])
ecc = float(sys.argv[2])
incl = float(sys.argv[3])
a = float(sys.argv[4])
logMbh = int(sys.argv[5])
timing_file = sys.argv[7]
window_file = sys.argv[8]
phi_r0 = 0
phi_theta0 = 0
phi_phi0 = 0
theta_obs = cp.pi/4
n_obs = cp.array([cp.sin(theta_obs), cp.array(0), cp.cos(theta_obs)])
theta_d = cp.radians(0)
phi_d = np.radians(0)
dt = 10 # timestep in seconds
G = 6.6743e-8 # cgs
Msun = 1.989e33 # g
Mbh = cp.array(10**logMbh) * Msun
c = 2.998e10 # cm/s
t_g = cp.array(G * Mbh / c**3)
P_d = 4e5 # disk period in s

windows = np.array(ast.literal_eval(sys.argv[6]), ndmin=2, dtype=np.float64)
np.savetxt(window_file, windows)
from pn_trajectory import trajectory
t, r, (x, y, z), lambd = trajectory(windows, np.array([logMbh]), np.array([sma]), np.array([ecc]), np.array([incl]), np.array([a]), np.array([phi_r0]), np.array([phi_theta0]), np.array([phi_phi0]), dt)

orbit_true = kg.StableOrbit(a, sma * (1 - ecc**2), ecc, np.cos(np.radians(incl)))
tgeo, rgeo, thetageo, phigeo = orbit_true.trajectory((0,np.pi+phi_r0,-np.pi/2+phi_theta0,phi_phi0))
lambd = lambd.get()[0]
tgeo = cp.array(tgeo(lambd)) * t_g
xgeo = cp.array(rgeo(lambd) * np.sin(thetageo(lambd)) * np.cos(phigeo(lambd)))
ygeo = cp.array(rgeo(lambd) * np.sin(thetageo(lambd)) * np.sin(phigeo(lambd)))
zgeo = cp.array(rgeo(lambd) * np.cos(thetageo(lambd)))
dot_product = n_obs[2]
D_t = zgeo
sign_changes = np.where((D_t[:-1]*D_t[1:])<0)[0].get()
model_timings = tgeo[sign_changes].get()
shapiro_delay = -2 * np.log(rgeo(lambd) * (1 + dot_product.get())) * t_g.get()
geometric_delay = -rgeo(lambd) * dot_product.get() * t_g.get()
model_timings += shapiro_delay[sign_changes] + geometric_delay[sign_changes]
np.savetxt(timing_file, model_timings)
