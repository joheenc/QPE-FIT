import numpy as np, cupy as cp, os
import matplotlib.pyplot as plt
import kerrgeopy as kg

sma = 120
a = 0.9
ecc = 0.3
incl = 80
phi_r0 = 0
phi_theta0 = 0
phi_phi0 = 0
theta_obs = cp.pi/4
n_obs = cp.array([cp.sin(theta_obs), cp.array(0), cp.cos(theta_obs)])
theta_d = cp.radians(1e-6)
phi_d = np.radians(0)
dt = 10 # timestep in seconds
G = 6.6743e-8 # cgs
Msun = 1.989e33 # g
logMbh = cp.array([6])
Mbh = cp.array(10**logMbh) * Msun
c = 2.998e10 # cm/s
t_g = cp.array(G * Mbh / c**3)
P_d = 1e8 # disk period in s

windows = np.array([[0, 4.1e5]], ndmin=2)
np.savetxt('windows.dat', windows)

from pn_trajectory import trajectory

t, r, (x, y, z), lambd = trajectory(windows, logMbh, np.array([sma]), np.array([ecc]), np.array([incl]), np.array([a]), np.array([phi_r0]), np.array([phi_theta0]), np.array([phi_phi0]), dt=dt)
t = t[0] * t_g # convert gravitational time to physical time
r = r[0]
x = x[0]
y = y[0]
z = z[0]
n_crs_x = cp.sin(theta_d) * cp.cos(2 * cp.pi * t / P_d + phi_d)
n_crs_y = cp.sin(theta_d) * cp.sin(2 * cp.pi * t / P_d + phi_d)
n_crs_z = cp.cos(theta_d)
dot_product = (n_obs[0]*n_crs_x + n_obs[1]*n_crs_y + n_obs[2]*n_crs_z)
D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z
sign_changes = cp.where((D_t[:-1]*D_t[1:])<0)[0]
model_timings = t[sign_changes]
shapiro_delay = -2 * np.log(r * (1 + dot_product)) * t_g
geometric_delay = -r * dot_product * t_g
model_timings += shapiro_delay[sign_changes] + geometric_delay[sign_changes]
np.savetxt('timings_pn.dat', model_timings.get())

plt.plot(np.diff(model_timings.get()), marker='o', label='PN')

orbit_true = kg.StableOrbit(a, sma * (1 - ecc**2), ecc, np.cos(np.radians(incl)))
tgeo, rgeo, thetageo, phigeo = orbit_true.trajectory((0,np.pi+phi_r0,-np.pi/2+phi_theta0,phi_phi0))
lambd = lambd.get()[0]
tgeo = cp.array(tgeo(lambd)) * t_g
xgeo = cp.array(rgeo(lambd) * np.sin(thetageo(lambd)) * np.cos(phigeo(lambd)))
ygeo = cp.array(rgeo(lambd) * np.sin(thetageo(lambd)) * np.sin(phigeo(lambd)))
zgeo = cp.array(rgeo(lambd) * np.cos(thetageo(lambd)))
n_crs_x = cp.sin(theta_d) * cp.cos(2 * cp.pi * tgeo / P_d + phi_d)
n_crs_y = cp.sin(theta_d) * cp.sin(2 * cp.pi * tgeo / P_d + phi_d)
n_crs_z = cp.cos(theta_d)
dot_product = (n_obs[0]*n_crs_x + n_obs[1]*n_crs_y + n_obs[2]*n_crs_z)
D_t = n_crs_x * xgeo + n_crs_y * ygeo + n_crs_z * zgeo
sign_changes = np.where((D_t[:-1]*D_t[1:])<0)[0].get()
model_timings = tgeo[sign_changes].get()
shapiro_delay = -2 * np.log(rgeo(lambd) * (1 + dot_product.get())) * t_g.get()
geometric_delay = -rgeo(lambd) * dot_product.get() * t_g.get()
model_timings += shapiro_delay[sign_changes] + geometric_delay[sign_changes]
np.savetxt('timings_kg.dat', model_timings)

plt.plot(np.diff(model_timings), marker='o', label='KerrGeo')
plt.legend()
plt.savefig('timings.png')
plt.close()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes[0][0].plot(lambd, xgeo.get(), 'o', label='KerrGeoPy')
axes[0][0].plot(lambd, x.get(), '.', label='PN')
axes[0][0].set_xlabel('lambda')
axes[0][0].set_ylabel('x')

axes[0][1].plot(lambd, ygeo.get(), 'o', label='KerrGeoPy')
axes[0][1].plot(lambd, y.get(), '.', label='PN')
axes[0][1].set_xlabel('lambda')
axes[0][1].set_ylabel('y')

axes[1][0].plot(lambd, zgeo.get(), 'o', label='KerrGeoPy')
axes[1][0].plot(lambd, z.get(), '.', label='PN')
axes[1][0].set_xlabel('lambda')
axes[1][0].set_ylabel('z')

axes[1][1].plot(lambd, tgeo.get(), 'o', label='KerrGeoPy')
axes[1][1].plot(lambd, t.get(), '.', label='PN')
axes[1][1].set_xlabel('lambda')
axes[1][1].set_ylabel('t')

axes[0][0].legend()
axes[0][1].legend()
axes[1][0].legend()
axes[1][1].legend()

plt.savefig('PNversusexact_cartesian.png')
plt.close()