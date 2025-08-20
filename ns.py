import numpy as np, cupy as cp, ultranest, matplotlib.pyplot as plt, corner, sys
from pn_trajectory import residuals
from ultranest.popstepsampler import PopulationSimpleSliceSampler, generate_region_oriented_direction

cp.cuda.MemoryPool().free_all_blocks()
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def prior_transform(cube):
    cube = np.atleast_2d(cube)
    if cube.shape[1] != len(param_lims):
        if cube.shape[0] == len(param_lims):
            cube = cube.T
    
    lows = np.array([lim[0] for lim in param_lims])
    highs = np.array([lim[1] for lim in param_lims])
    params = cube * (highs - lows) + lows

    return params

def log_likelihood(params):
    params = np.atleast_2d(params)
    resid = residuals(timings, windows, errs, *params.T, dt)
    ll = -0.5 * resid.get()
    
    ll = np.where(np.isfinite(ll), ll, -1e30)    
    return ll

if __name__ == '__main__':
    output_dir = sys.argv[1]
    timings = cp.asarray(np.loadtxt(sys.argv[2]), dtype=cp.float64)
    windows = np.loadtxt(sys.argv[3], ndmin=2).astype(np.float64)
    errs = cp.asarray(np.loadtxt(sys.argv[4]), dtype=cp.float64)
    dt = float(sys.argv[5])
    
    sma_lims = (50, 100)
    e_lims = (0, 0.3)
    incl_lims = (0, 180)
    phi_r0_lims = (0, 2*np.pi)
    phi_theta0_lims = (0, 2*np.pi)
    phi_phi0_lims = (0, 2*np.pi)
    a_lims = (0., 0.998)
    logMbh_lims = (5.6, 6)
    theta_obs_lims = (0, 2*np.pi)
    theta_d_lims = (0, 80)
    P_d_lims = (200, 800)
    phi_d0_lims = (0, 2*np.pi)
    param_lims = [sma_lims, e_lims, incl_lims, phi_r0_lims, phi_theta0_lims, phi_phi0_lims,
                  a_lims, logMbh_lims, theta_obs_lims, theta_d_lims, P_d_lims, phi_d0_lims]
    
    param_names = ['$a$', '$e$', '$i$', r'$\phi_{r,0}$', r'$\phi_{\theta,0}$',  r'$\phi_{\phi,0}$', 
                   r'$\chi_\bullet$', r'$\log M_\bullet$', r'$\theta_{\rm obs}$', 
                   r'$\theta_{\rm disk}$', r'$T_{\rm disk}/P_{\rm orb}$', r'$\phi_{\rm disk}$']

    sampler = ultranest.ReactiveNestedSampler(
        param_names, 
        log_likelihood,
        prior_transform,
        log_dir=output_dir,
        resume='resume',
        vectorized=True,
        draw_multiple=True,
        num_test_samples=100,
        wrapped_params=[False,False,False,True,True,True,False,False,True,False,False,True]
    )
    
    sampler.stepsampler = PopulationSimpleSliceSampler(
        popsize=512,
        nsteps=256,
        generate_direction=generate_region_oriented_direction
    )
    
    result = sampler.run(
        min_num_live_points=1000,
        dlogz=0.1,
        show_status=True,
        viz_callback=False
    )
    
    sampler.print_results()
    print(f"\nParameter estimates:")
    for i, name in enumerate(param_names):
        samples = result['samples'][:, i]
        q16, q50, q84 = np.percentile(samples, [16, 50, 84])
        print(f"{name:12s}: {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")
    
    corner_fig = corner.corner(
        result['samples'], 
        labels=param_names,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
    )
    corner_fig.savefig(f"{output_dir}/corner.png", dpi=150)
    plt.close(corner_fig)
