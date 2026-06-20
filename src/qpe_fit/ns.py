import numpy as np
import argparse
import json
import ultranest
import matplotlib.pyplot as plt
import corner
from ultranest.popstepsampler import PopulationSimpleSliceSampler, PopulationSliceSampler, PopulationRandomWalkSampler, generate_region_oriented_direction, generate_random_direction, generate_mixture_random_direction, generate_cube_oriented_direction
import trajectory as pn

DIRECTION_FUNCS = {
    'region': generate_region_oriented_direction,
    'random': generate_random_direction,
    'mixture': generate_mixture_random_direction,
    'cube': generate_cube_oriented_direction,
}

REGION_CLASSES = {
    'mlfriends': ultranest.mlfriends.MLFriends,
    'simple': ultranest.mlfriends.SimpleRegion,
    'ellipsoid': ultranest.mlfriends.RobustEllipsoidRegion,
}

SAMPLERS = {
    'none': None,
    'slice': PopulationSimpleSliceSampler,
    'harm': PopulationSliceSampler,
    'rwalk': PopulationRandomWalkSampler,
}

REQUIRED_MODEL_PARAMS = [
    'sma', 'e', 'incl', 'phi_r0', 'phi_theta0', 'phi_phi0',
    'spin', 'logMbh', 'theta_obs',
]
# Optional physics.  Present in priors.json -> sampled; absent -> the disabling
# default in pn.residuals is used and the effect is simply not explored.  The
# set (and the disabling defaults) are defined in trajectory.OPTIONAL_DEFAULTS:
#   theta_d (tilt), P_d + phi_d (precession), r_warp (warp), adot/edot (decay).
OPTIONAL_MODEL_PARAMS = list(pn.OPTIONAL_DEFAULTS.keys())

def load_priors(path):
    with open(path) as f:
        return json.load(f)

def make_prior_transform(priors, param_names):
    """Per-parameter prior transform.

    Each prior entry may set "type":
      - "uniform"     (default): uniform over [lo, hi].
      - "log"                  : log-uniform over [lo, hi]  (lo, hi > 0).
      - "signed_log"           : two-sided log-uniform over +/-[lo, hi], where
                                 lo, hi are positive magnitudes.  Half the prior
                                 mass sits on each sign and the magnitude is
                                 log-uniform.  The cube->value map is monotonic
                                 (u=0 -> -hi, u->0.5- -> -lo, u=0.5+ -> +lo,
                                 u=1 -> +hi), so there is no discrete sign jump
                                 for the sampler to fight -- only a gap of width
                                 2*lo straddling zero (the smallest resolvable
                                 magnitude).

                                 Optional "zero_frac" (default 0): reserve a
                                 central cube slice of this width that maps to
                                 exactly 0, putting a probability atom of mass
                                 zero_frac on the "no effect" hypothesis (e.g. the
                                 no-secular-evolution point adot=0 / edot=0).  The
                                 two log-uniform wings keep their shape on the
                                 remaining 1 - zero_frac of the mass, and the map
                                 stays monotonic non-decreasing, so the static
                                 hypothesis becomes explicitly samplable.  With
                                 zero_frac = 0 the transform is unchanged.

                                 NB: for an evolving-vs-static *model* comparison
                                 the cleaner route is two runs (with / without the
                                 parameter) compared by evidence Z, which nested
                                 sampling yields directly; the atom is the
                                 single-run alternative.
    """
    specs = [(priors[name].get('type', 'uniform'), float(priors[name]['bounds'][0]),
              float(priors[name]['bounds'][1]), float(priors[name].get('zero_frac', 0.0)))
             for name in param_names]

    def transform(cube):
        cube = np.atleast_2d(cube)
        if cube.shape[1] != len(specs) and cube.shape[0] == len(specs):
            cube = cube.T
        out = np.empty_like(cube)
        for i, (ptype, lo, hi, zf) in enumerate(specs):
            u = cube[:, i]
            if ptype == 'uniform':
                out[:, i] = u * (hi - lo) + lo
            elif ptype == 'log':
                llo, lhi = np.log10(lo), np.log10(hi)
                out[:, i] = 10.0**(llo + u * (lhi - llo))
            elif ptype == 'signed_log':
                llo, lhi = np.log10(lo), np.log10(hi)
                if zf > 0.0:
                    lo_u, hi_u = 0.5 - zf / 2.0, 0.5 + zf / 2.0    # central [lo_u, hi_u] -> 0
                    mag_neg = lhi + (u / lo_u) * (llo - lhi)               # |.|: hi -> lo on [0, lo_u)
                    mag_pos = llo + ((u - hi_u) / (1.0 - hi_u)) * (lhi - llo)  # |.|: lo -> hi on (hi_u, 1]
                    val = np.zeros_like(u)                                  # central slice stays 0
                    val = np.where(u < lo_u, -10.0**mag_neg, val)
                    val = np.where(u > hi_u, 10.0**mag_pos, val)
                    out[:, i] = val
                else:
                    mag_neg = lhi + (u / 0.5) * (llo - lhi)        # |.|: hi -> lo  on u in [0, 0.5)
                    mag_pos = llo + ((u - 0.5) / 0.5) * (lhi - llo)    # |.|: lo -> hi  on u in [0.5, 1]
                    out[:, i] = np.where(u < 0.5, -10.0**mag_neg, 10.0**mag_pos)
            else:
                raise ValueError(f"unknown prior type {ptype!r} for parameter {param_names[i]!r}")
        return out
    return transform

def make_log_likelihood(timings, windows, errs, dt, one_per_orbit, param_names):
    missing = [name for name in REQUIRED_MODEL_PARAMS if name not in param_names]
    if missing:
        raise ValueError(f"Missing required prior(s): {', '.join(missing)}")
    unknown = [name for name in param_names
               if name not in REQUIRED_MODEL_PARAMS and name not in OPTIONAL_MODEL_PARAMS]
    if unknown:
        raise ValueError(f"Unknown prior parameter(s): {', '.join(unknown)}. "
                         f"Optional physics params are: {', '.join(OPTIONAL_MODEL_PARAMS)}.")

    def loglike(params):
        params = np.atleast_2d(params).astype(np.float32)
        if params.shape[1] != len(param_names) and params.shape[0] == len(param_names):
            params = params.T

        # Map by name (independent of JSON key order).  Required params are
        # passed positionally; optional physics params are forwarded only if the
        # user put them in priors.json -- anything omitted falls back to the
        # disabling default in pn.residuals, so that physics is simply not sampled.
        param_by_name = {name: params[:, i] for i, name in enumerate(param_names)}
        base_args = [param_by_name[name] for name in REQUIRED_MODEL_PARAMS]
        opt_kwargs = {name: param_by_name[name] for name in OPTIONAL_MODEL_PARAMS if name in param_by_name}
        resid = pn.residuals(timings, windows, errs, *base_args, dt, one_per_orbit, **opt_kwargs)
        ll = -0.5 * pn.to_numpy(resid)
        return np.where(np.isfinite(ll), ll, -1e30).astype(np.float64)
    return loglike

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output', help='Sampling output directory')
    parser.add_argument('--timings', default='timings.txt', help='.txt file containing QPE timings (one per line)')
    parser.add_argument('--windows', default='windows.txt', help='.txt file containing observation windows (one start-stop pair per line separated by a space)')
    parser.add_argument('--errors', default='errors.txt', help='.txt file containing QPE timing errors (one per line)')
    parser.add_argument('--priors', default='priors.json', help='.json file containing sampling priors')
    parser.add_argument('--dt', type=float, default=10.0, help='Time step size for likelihood evaluations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU-accelerated likelihood evaluation')
    parser.add_argument('--stepsampler', choices=SAMPLERS.keys(), default='slice', help='Which step sampler to use. Options are none, slice, harm, rwalk.')
    parser.add_argument('--direction', choices=DIRECTION_FUNCS.keys(), default='region', help='Which direction function to use for step sampling. Options are region, random, mixture, cube.')
    parser.add_argument('--region', choices=REGION_CLASSES.keys(), default='simple', help='Which region class to use. Options are mlfriends, simple, ellipsoid.')
    parser.add_argument('--popsize', type=int, default=256, help='Number of walkers to maintain for step sampling.')
    parser.add_argument('--nsteps', type=int, default=256, help='Number of steps for step sampling.')
    parser.add_argument('--nlive', type=int, default=600, help='Minimum number of live points throughout the run')
    parser.add_argument('--dkl', type=float, default=0.5, help='Target posterior uncertainty (KL divergence b/w bootstrapped integrators, in nats)')
    parser.add_argument('--frac-remain', type=float, default=0.01, help='Integrate until this fraction of the integral is left in the remainder.')
    parser.add_argument('--min-ess', type=int, default=400, help='Minimum effective sample size for nested sampling.')
    parser.add_argument('--one-per-orbit', action='store_true', help='Only one QPE per orbit (not two).')
    args = parser.parse_args()

    pn.set_backend(args.gpu)
    
    timings = pn.xp.asarray(np.loadtxt(args.timings), dtype=pn.xp.float32)
    windows = np.loadtxt(args.windows, ndmin=2)
    errs = pn.xp.asarray(np.loadtxt(args.errors), dtype=pn.xp.float32)
    priors = load_priors(args.priors)
    
    param_names = list(priors.keys())
    wrapped = [priors[p].get('wrapped', False) for p in param_names]

    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        make_log_likelihood(timings, windows, errs, args.dt, args.one_per_orbit, param_names),
        make_prior_transform(priors, param_names),
        log_dir=args.output,
        resume='resume',
        vectorized=True,
        wrapped_params=wrapped,
    )
    
    if SAMPLERS[args.stepsampler]:
        sampler.stepsampler = SAMPLERS[args.stepsampler](
            popsize=args.popsize,
            nsteps=args.nsteps,
            generate_direction=DIRECTION_FUNCS[args.direction]        )
    
    sampler.run(
        dKL=args.dkl,
        min_num_live_points=args.nlive,
        frac_remain=args.frac_remain,
        min_ess=args.min_ess,
        region_class=REGION_CLASSES[args.region],
    )
    
    sampler.print_results()

    # For signed_log parameters the posterior spans decades and both signs, so
    # plot them as sign(x)*log10|x| to keep the corner readable.
    samples = sampler.results['samples'].copy()
    labels = list(param_names)
    for i, name in enumerate(param_names):
        if priors[name].get('type') == 'signed_log':
            s = samples[:, i]
            samples[:, i] = np.sign(s) * np.log10(np.abs(s))
            labels[i] = f'sign*log10|{name}|'

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
        bins=50
    )
    fig.savefig(f"{args.output}/corner.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
