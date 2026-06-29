import numpy as np
import argparse
import json
from . import trajectory as pn

# Note on the generator model:
# The original QPE-FIT generator used a single kerrgeopy.StableOrbit as an
# independent high-accuracy "truth".  A StableOrbit has fixed constants of
# motion, so it cannot represent a secularly evolving (adot/edot != 0) orbit.
# We therefore generate timings from the same adiabatic PN trajectory used in
# the likelihood, so injected data and the recovery model are self-consistent.
# An independent kerrgeopy cross-check for the evolving case (stitching together
# StableOrbits along the a(t), e(t) track) is left as a separate validation.

T_G_FACTOR = 4.926580927874239e-06  # seconds per M, for log10(M/Msun) = 0


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

    # Optional physics: params.json may include any of the disk / secular-
    # evolution parameters in pn.OPTIONAL_DEFAULTS (theta_d, P_d, phi_d, r_warp,
    # adot, edot).  Anything absent falls back to its disabling default, so the
    # generator reproduces the simpler model.  Scalars broadcast across sets.
    for opt, default in pn.OPTIONAL_DEFAULTS.items():
        if opt not in p:
            p[opt] = np.full(n_sets, default)
        elif len(p[opt]) == 1 and n_sets > 1:
            p[opt] = np.full(n_sets, p[opt][0])

    windows = np.loadtxt(args.windows, ndmin=2)

    t, r, (x, y, z), _, P_orb = pn.trajectory(
        windows,
        p['logMbh'], p['sma'], p['e'],
        p['incl'], p['spin'], p['phi_r0'],
        p['phi_theta0'], p['phi_phi0'], args.dt,
        adot=p['adot'], edot=p['edot'],
    )

    M = T_G_FACTOR * 10**p['logMbh']            # seconds per M  (n_sets,)
    t = t * M[:, None]                          # geometric -> seconds
    P_d = p['P_d'] * P_orb                       # seconds (inf if non-precessing)

    # Identical disk geometry to the likelihood (pn.disk_normal); with r_warp=0
    # this is a flat tilted+precessing disk, matching the original generator.
    n_crs_x, n_crs_y, n_crs_z = pn.disk_normal(
        t, r, np.radians(p['theta_d']), P_d, p['phi_d'], p['r_warp'])

    D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z

    if args.one_per_orbit:
        crossings_mask = (D_t[:, :-1] < 0) & (D_t[:, 1:] >= 0)
    else:
        crossings_mask = (D_t[:, :-1] * D_t[:, 1:]) < 0
    alpha = -D_t[:, :-1] / (D_t[:, 1:] - D_t[:, :-1])

    t_cross = t[:, :-1] + alpha * (t[:, 1:] - t[:, :-1])
    x_cross = x[:, :-1] + alpha * (x[:, 1:] - x[:, :-1])
    y_cross = y[:, :-1] + alpha * (y[:, 1:] - y[:, :-1])
    z_cross = z[:, :-1] + alpha * (z[:, 1:] - z[:, :-1])
    r_cross = r[:, :-1] + alpha * (r[:, 1:] - r[:, :-1])

    r_mag = np.sqrt(x_cross**2 + y_cross**2 + z_cross**2)
    cos_angle = (np.sin(p['theta_obs'])[:, None] * x_cross
                 + np.cos(p['theta_obs'])[:, None] * z_cross) / r_mag

    shapiro_delay = -2 * M[:, None] * np.log(r_cross * (1 - cos_angle))
    geometric_delay = r_cross * cos_angle * M[:, None]

    crossings = np.where(crossings_mask, t_cross + shapiro_delay + geometric_delay, np.nan)
    all_crossings = [crossings[i][~np.isnan(crossings[i])] for i in range(n_sets)]

    if n_sets == 1:
        np.savetxt(args.output_timings, all_crossings[0])
    else:
        with open(args.output_timings, 'w') as f:
            for c in all_crossings:
                f.write(' '.join(map(str, c)) + '\n')


if __name__ == '__main__':
    main()
