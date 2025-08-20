import numpy as np
import sys
import ast
import kerrgeopy as kg
from pn_trajectory import trajectory

# Parse command line arguments
sma = float(sys.argv[1])
ecc = float(sys.argv[2])
incl = float(sys.argv[3])
phi_r0 = float(sys.argv[4])
phi_theta0 = float(sys.argv[5])
phi_phi0 = float(sys.argv[6])
spin = float(sys.argv[7])
logMbh = float(sys.argv[8])
theta_obs = float(sys.argv[9])
theta_d = float(sys.argv[10])
P_d = float(sys.argv[11])
phi_d = float(sys.argv[12])
windows = np.array(ast.literal_eval(sys.argv[13]), ndmin=2, dtype=np.float64)
timing_file = sys.argv[14]
window_file = sys.argv[15]

# Physical constants
G = 6.6743e-8  # cgs
Msun = 1.989e33  # g
c = 2.998e10  # cm/s
dt = 10  # timestep in seconds

# Derived quantities
Mbh = 10**logMbh * Msun
t_g = G * Mbh / c**3

# Save windows
np.savetxt(window_file, windows)

# Get orbital period from PN trajectory
_, _, _, lambd, P_orb = trajectory(
    windows, 
    np.array([logMbh]), np.array([sma]), np.array([ecc]), 
    np.array([incl]), np.array([spin]), np.array([phi_r0]), 
    np.array([phi_theta0]), np.array([phi_phi0]), dt
)
P_d *= P_orb[0].get()
lambd = lambd.get()[0]

# Compute exact trajectory using KerrGeoPy
orbit_true = kg.StableOrbit(spin, sma * (1 - ecc**2), ecc, np.cos(np.radians(incl)))
tgeo_func, rgeo_func, thetageo_func, phigeo_func = orbit_true.trajectory(
    (0, np.pi + phi_r0, -np.pi/2 + phi_theta0, phi_phi0)
)

# Evaluate trajectory
tgeo = tgeo_func(lambd) * t_g
rgeo = rgeo_func(lambd)
thetageo = thetageo_func(lambd)
phigeo = phigeo_func(lambd)

# Convert to Cartesian coordinates
sin_theta = np.sin(thetageo)
xgeo = rgeo * sin_theta * np.cos(phigeo)
ygeo = rgeo * sin_theta * np.sin(phigeo)
zgeo = rgeo * np.cos(thetageo)

# Compute disk normal vector (time-dependent)
theta_d_rad = np.radians(theta_d)
disk_phase = 2 * np.pi * tgeo / P_d + phi_d
n_crs_x = np.sin(theta_d_rad) * np.cos(disk_phase)
n_crs_y = np.sin(theta_d_rad) * np.sin(disk_phase)
n_crs_z = np.cos(theta_d_rad)

# Compute distance from disk
D_t = n_crs_x * xgeo + n_crs_y * ygeo + n_crs_z * zgeo

# Find disk crossings with interpolation
crossings_mask = (D_t[:-1] * D_t[1:]) < 0
alpha = -D_t[:-1] / (D_t[1:] - D_t[:-1])

# Interpolate time and position at crossings
t_cross = tgeo[:-1] + alpha * (tgeo[1:] - tgeo[:-1])
x_cross = xgeo[:-1] + alpha * (xgeo[1:] - xgeo[:-1])
y_cross = ygeo[:-1] + alpha * (ygeo[1:] - ygeo[:-1])
z_cross = zgeo[:-1] + alpha * (zgeo[1:] - zgeo[:-1])
r_cross = rgeo[:-1] + alpha * (rgeo[1:] - rgeo[:-1])

# Calculate relativistic/geometrical delays
r_mag = np.sqrt(x_cross**2 + y_cross**2 + z_cross**2)
theta_obs_rad = np.radians(theta_obs)
cos_angle = (np.sin(theta_obs_rad) * x_cross + np.cos(theta_obs_rad) * z_cross) / r_mag

shapiro_delay = -2 * t_g * np.log(r_cross * (1 - cos_angle))
geometric_delay = r_cross * cos_angle * t_g

# Get observed crossing times & save results
crossings = np.where(crossings_mask, t_cross + shapiro_delay + geometric_delay, np.nan)
np.savetxt(timing_file, crossings[~np.isnan(crossings)])
