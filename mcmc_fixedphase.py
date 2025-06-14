import numpy as np, cupy as cp, emcee, matplotlib.pyplot as plt, h5py, heapq, corner, sys
from pn_trajectory import residuals
from emcee.backends import HDFBackend

def log_prior(theta, param_lims):
    valid = np.logical_and.reduce([(theta[:, i] >= lim[0]) & (theta[:, i] <= lim[1]) for i, lim in enumerate(param_lims)])
    lp = np.where(valid, 0, -np.inf)
    return lp

def log_likelihood(theta, timings, windows, errs, dt):
    resid = residuals(timings, windows, errs, *theta.T, dt)
    ll = -np.log(resid.get())
    return ll

def log_prob(theta, param_lims, timings, windows, errs, dt):
    lp = log_prior(theta, param_lims)
    ll = log_likelihood(theta, timings, windows, errs, dt)
    result = lp + ll
    result[~np.isfinite(result)] = -np.inf
    return result

if __name__ == '__main__':
    output_file = sys.argv[1]
    timings = cp.loadtxt(sys.argv[2])
    windows = np.loadtxt(sys.argv[3], ndmin=2)
    errs = cp.loadtxt(sys.argv[4])
    nsteps = int(sys.argv[5])
    nwalkers = int(sys.argv[6])
    dt = float(sys.argv[7])

    sma_lims = (30, 500)
    e_lims = (0, 0.9)
    incl_lims = (0, 90)
    a_lims = (0., 0.998)
    logMbh_lims = (4, 8)
    theta_obs_lims = (0, np.pi)
    theta_d_lims = (0, 60)
    P_d_lims = (2, 100)
    phi_d_lims = (0, np.pi)
    t0_lims = (-50000, 50000)
    param_lims = [sma_lims, e_lims, incl_lims, a_lims, logMbh_lims, theta_obs_lims, theta_d_lims, P_d_lims, phi_d_lims, t0_lims]
    ndim = len(param_lims)
    rng = np.random.default_rng()
    p0 = np.column_stack([rng.uniform(lim[0], lim[1], size=nwalkers) for lim in param_lims])

    backend = HDFBackend(output_file)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[param_lims, timings, windows, errs, dt], backend=backend, vectorize=True)
    backend.reset(nwalkers, ndim)
    sampler.run_mcmc(p0, nsteps, progress=True)
