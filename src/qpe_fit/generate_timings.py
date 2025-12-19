import numpy as np
import argparse
import json
import kerrgeopy as kg
from . import trajectory as pn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True, help='JSON file with EMRI/MBH parameters')
    parser.add_argument('--windows', required=True, help='Observation windows file')
    parser.add_argument('--output-timings', required=True)
    parser.add_argument('--one-per-orbit', action='store_true', help='Only one QPE per orbit (not two).')
    parser.add_argument('--dt', type=float, default=10.0)
    args = parser.parse_args()
    
    with open(args.params) as f:
        p = json.load(f)
    
    windows = np.loadtxt(args.windows, ndmin=2)    
    t_g = 4.926580927874239e-06 * 10**p['logMbh']

    _, _, _, lambd, P_orb = pn.trajectory(
        windows,
        np.array([p['logMbh']]), np.array([p['sma']]), np.array([p['e']]),
        np.array([p['incl']]), np.array([p['spin']]), np.array([p['phi_r0']]),
        np.array([p['phi_theta0']]), np.array([p['phi_phi0']]), args.dt
    )
    P_d = p['P_d'] * P_orb[0]
    lambd = lambd[0]
    
    orbit = kg.StableOrbit(
        p['spin'],
        p['sma'] * (1 - p['e']**2),
        p['e'],
        np.cos(np.radians(p['incl']))
    )
    tgeo_func, rgeo_func, thetageo_func, phigeo_func = orbit.trajectory(
        (0, np.pi + p['phi_r0'], -np.pi/2 + p['phi_theta0'], p['phi_phi0'])
    )
    
    tgeo = tgeo_func(lambd) * t_g
    rgeo = rgeo_func(lambd)
    thetageo = thetageo_func(lambd)
    phigeo = phigeo_func(lambd)
    
    sin_theta = np.sin(thetageo)
    xgeo = rgeo * sin_theta * np.cos(phigeo)
    ygeo = rgeo * sin_theta * np.sin(phigeo)
    zgeo = rgeo * np.cos(thetageo)
    
    theta_d_rad = np.radians(p['theta_d'])
    disk_phase = 2 * np.pi * tgeo / P_d + p['phi_d']
    n_crs_x = np.sin(theta_d_rad) * np.cos(disk_phase)
    n_crs_y = np.sin(theta_d_rad) * np.sin(disk_phase)
    n_crs_z = np.cos(theta_d_rad)
    
    D_t = n_crs_x * xgeo + n_crs_y * ygeo + n_crs_z * zgeo
    
    if args.one_per_orbit:
        crossings_mask = (D_t[:-1] < 0) & (D_t[1:] >= 0)
    else:
        crossings_mask = (D_t[:-1] * D_t[1:]) < 0
    alpha = -D_t[:-1] / (D_t[1:] - D_t[:-1])
    
    t_cross = tgeo[:-1] + alpha * (tgeo[1:] - tgeo[:-1])
    x_cross = xgeo[:-1] + alpha * (xgeo[1:] - xgeo[:-1])
    y_cross = ygeo[:-1] + alpha * (ygeo[1:] - ygeo[:-1])
    z_cross = zgeo[:-1] + alpha * (zgeo[1:] - zgeo[:-1])
    r_cross = rgeo[:-1] + alpha * (rgeo[1:] - rgeo[:-1])
    
    r_mag = np.sqrt(x_cross**2 + y_cross**2 + z_cross**2)
    cos_angle = (np.sin(p['theta_obs']) * x_cross + np.cos(p['theta_obs']) * z_cross) / r_mag
    
    shapiro_delay = -2 * t_g * np.log(r_cross * (1 - cos_angle))
    geometric_delay = r_cross * cos_angle * t_g
    
    crossings = np.where(crossings_mask, t_cross + shapiro_delay + geometric_delay, np.nan)
    np.savetxt(args.output_timings, crossings[~np.isnan(crossings)])

if __name__ == '__main__':
    main()
