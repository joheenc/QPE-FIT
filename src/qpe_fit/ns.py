import numpy as np
import argparse
import json
import ultranest
import matplotlib.pyplot as plt
import corner
from ultranest.popstepsampler import PopulationSimpleSliceSampler, PopulationSliceSampler, PopulationRandomWalkSampler, generate_region_oriented_direction, generate_random_direction, generate_mixture_random_direction, generate_cube_oriented_direction
from . import trajectory as pn

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

def load_priors(path):
    with open(path) as f:
        return json.load(f)

def make_prior_transform(param_lims):
    lims = np.array(list(param_lims.values()))
    def transform(cube):
        cube = np.atleast_2d(cube)
        if cube.shape[1] != len(lims) and cube.shape[0] == len(lims):
            cube = cube.T
        return cube * (lims[:, 1] - lims[:, 0]) + lims[:, 0]
    return transform

def make_log_likelihood(timings, windows, errs, dt, one_per_orbit):
    def loglike(params):
        params = np.atleast_2d(params).astype(np.float32)
        resid = pn.residuals(timings, windows, errs, *params.T, dt, one_per_orbit)
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
    param_lims = {p: priors[p]['bounds'] for p in param_names}
    
    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        make_log_likelihood(timings, windows, errs, args.dt, args.one_per_orbit),
        make_prior_transform(param_lims),
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
    
    fig = corner.corner(
        sampler.results['samples'],
        labels=param_names,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
        bins=50
    )
    fig.savefig(f"{args.output}/corner.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
