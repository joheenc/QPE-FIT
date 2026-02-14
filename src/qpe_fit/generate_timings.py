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
    
    for key in p:
        p[key] = np.atleast_1d(p[key])
    n_sets = len(p['logMbh'])
    
    windows = np.loadtxt(args.windows, ndmin=2)    
    t_g = 4.926580927874239e-06 * 10**p['logMbh']

    _, _, _, lambd, P_orb = pn.trajectory(
        windows,
        p['logMbh'], p['sma'], p['e'],
        p['incl'], p['spin'], p['phi_r0'],
        p['phi_theta0'], p['phi_phi0'], args.dt
    )
    P_d = p['P_d'] * P_orb
    
    all_crossings = []
    for i in range(n_sets):
        orbit = kg.StableOrbit(
            p['spin'][i],
            p['sma'][i] * (1 - p['e'][i]**2),
            p['e'][i],
            np.cos(np.radians(p['incl'][i]))
        )
        tgeo_func, rgeo_func, thetageo_func, phigeo_func = orbit.trajectory(
            (0, np.pi + p['phi_r0'][i], -np.pi/2 + p['phi_theta0'][i], p['phi_phi0'][i])
        )
        
        tgeo = tgeo_func(lambd[i]) * t_g[i]
        rgeo = rgeo_func(lambd[i])
        thetageo = thetageo_func(lambd[i])
        phigeo = phigeo_func(lambd[i])
        
        sin_theta = np.sin(thetageo)
        xgeo = rgeo * sin_theta * np.cos(phigeo)
        ygeo = rgeo * sin_theta * np.sin(phigeo)
        zgeo = rgeo * np.cos(thetageo)
        
        theta_d_rad = np.radians(p['theta_d'][i])
        disk_phase = 2 * np.pi * tgeo / P_d[i] + p['phi_d'][i]
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
        cos_angle = (np.sin(p['theta_obs'][i]) * x_cross + np.cos(p['theta_obs'][i]) * z_cross) / r_mag
        
        shapiro_delay = -2 * t_g[i] * np.log(r_cross * (1 - cos_angle))
        geometric_delay = r_cross * cos_angle * t_g[i]
        
        crossings = np.where(crossings_mask, t_cross + shapiro_delay + geometric_delay, np.nan)
        all_crossings.append(crossings[~np.isnan(crossings)])
    
    if n_sets == 1:
        np.savetxt(args.output_timings, all_crossings[0])
    else:
        with open(args.output_timings, 'w') as f:
            for crossings in all_crossings:
                f.write(' '.join(map(str, crossings)) + '\n')

if __name__ == '__main__':
    main()
