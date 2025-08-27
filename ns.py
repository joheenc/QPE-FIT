import numpy as np, cupy as cp, ultranest, matplotlib.pyplot as plt, corner, sys, os
from pn_trajectory import residuals
from ultranest.popstepsampler import PopulationSimpleSliceSampler, generate_region_oriented_direction
from ultranest.stepsampler import SliceSampler

cp.cuda.MemoryPool().free_all_blocks()
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def prior_transform(cube):
    """
    Transform from unit cube to physical parameters.
    Using uniform in cos(inclination) for isotropic orientations.
    """
    cube = np.atleast_2d(cube).astype(np.float64)
    if cube.shape[1] != len(param_lims):
        if cube.shape[0] == len(param_lims):
            cube = cube.T
    
    params = np.zeros_like(cube, dtype=np.float64)
    
    # Semi-major axis: log-uniform prior
    params[:, 0] = 10**(cube[:, 0] * (np.log10(param_lims[0][1]) - np.log10(param_lims[0][0])) + np.log10(param_lims[0][0]))
    
    # Eccentricity: uniform
    params[:, 1] = cube[:, 1] * (param_lims[1][1] - param_lims[1][0]) + param_lims[1][0]
    
    # Orbital inclination: uniform in cos(i) for isotropic orientation
    cos_i = 1 - 2*cube[:, 2]  # Maps [0,1] to [1,-1] (cos(0°) to cos(180°))
    params[:, 2] = np.arccos(cos_i) * 180/np.pi
    
    # Phase parameters: uniform
    params[:, 3] = cube[:, 3] * 2*np.pi  # phi_r0
    params[:, 4] = cube[:, 4] * 2*np.pi  # phi_theta0
    params[:, 5] = cube[:, 5] * 2*np.pi  # phi_phi0
    
    # Spin: uniform
    params[:, 6] = cube[:, 6] * (param_lims[6][1] - param_lims[6][0]) + param_lims[6][0]
    
    # Log mass: uniform
    params[:, 7] = cube[:, 7] * (param_lims[7][1] - param_lims[7][0]) + param_lims[7][0]
    
    # Observer angle: uniform
    params[:, 8] = cube[:, 8] * 2*np.pi
    
    # Disk inclination: uniform in cos(theta_d) for isotropic orientation
    cos_theta_d_min = np.cos(param_lims[9][1] * np.pi/180)  # cos(80°)
    cos_theta_d = cube[:, 9] * (1 - cos_theta_d_min) + cos_theta_d_min
    params[:, 9] = np.arccos(cos_theta_d) * 180/np.pi
    
    # Disk period: log-uniform (scale-invariant)
    params[:, 10] = 10**(cube[:, 10] * (np.log10(param_lims[10][1]) - np.log10(param_lims[10][0])) + np.log10(param_lims[10][0]))
    
    # Disk phase: uniform
    params[:, 11] = cube[:, 11] * 2*np.pi
    
    return params

def log_likelihood(params):
    params = np.atleast_2d(params).astype(np.float64)
    batch_size = params.shape[0]
    params_gpu = params.astype(np.float32)
    
    # Compute residuals on GPU with float32 for efficiency
    resid = residuals(timings, windows, errs, *params_gpu.T, dt, dtype=cp.float32)
    ll = -0.5 * resid.get()
    ll = np.where(np.isfinite(ll), ll, -1e30)
    # Return as float64 for UltraNest
    return ll.astype(np.float64)

if __name__ == '__main__':
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
    os.environ['CUPY_TF32'] = '1'

    output_dir = sys.argv[1]
    
    # Load data as float32 for GPU efficiency
    timings = cp.asarray(np.loadtxt(sys.argv[2]), dtype=cp.float32)
    windows = np.loadtxt(sys.argv[3], ndmin=2).astype(np.float64)
    errs = cp.asarray(np.loadtxt(sys.argv[4]), dtype=cp.float32)
    dt = float(sys.argv[5])
    
    # Prior limits
    sma_lims = (50, 1000)
    e_lims = (0, 0.5)
    incl_lims = (0, 180)
    phi_r0_lims = (0, 2*np.pi)
    phi_theta0_lims = (0, 2*np.pi)
    phi_phi0_lims = (0, 2*np.pi)
    a_lims = (0., 0.998)
    logMbh_lims = (4, 7)
    theta_obs_lims = (0, 2*np.pi)
    theta_d_lims = (0, 80)
    P_d_lims = (2, 1000)
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
    
    # Run with settings optimized for multi-modal posteriors (you may want to tweak them!)
    result = sampler.run(
        min_num_live_points=2000,
        dlogz=0.05,
        dKL=0.5,
        frac_remain=0.01,
        max_num_improvement_loops=3,
        show_status=True,
        viz_callback=False,
        cluster_num_live_points=50,
    )
    
    sampler.print_results()
    
    # Report if multiple modes found
    if hasattr(result, 'modes') and len(result.get('modes', [])) > 1:
        print(f"\n⚠ WARNING: Found {len(result['modes'])} modes in posterior!")
        print("This suggests multiple solutions fit the data similarly well.")
        for i, mode in enumerate(result['modes']):
            print(f"\nMode {i+1} (log Z = {mode['logz']:.2f}):")
            for j, name in enumerate(param_names):
                samples = mode['samples'][:, j]
                q50 = np.median(samples)
                print(f"  {name:12s}: {q50:.4f}")
    
    print(f"\nParameter estimates (global):")
    for i, name in enumerate(param_names):
        samples = result['samples'][:, i]
        q16, q50, q84 = np.percentile(samples, [16, 50, 84])
        print(f"{name:12s}: {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")
    
    # Generate corner plot
    corner_fig = corner.corner(
        result['samples'], 
        labels=param_names,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
        truth_color='red',
        bins=50
    )
    corner_fig.savefig(f"{output_dir}/corner.png", dpi=150, bbox_inches='tight')
    plt.close(corner_fig)
    
    # Generate convergence diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Log likelihood trace
    axes[0, 0].plot(result['weighted_samples']['logl'], alpha=0.5)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Log Likelihood')
    axes[0, 0].set_title('Likelihood Evolution')
    
    # Posterior mass evolution
    if 'logz' in result:
        axes[0, 1].axhline(result['logz'], color='red', linestyle='--', label=f"Final: {result['logz']:.2f}")
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Log Evidence')
        axes[0, 1].set_title('Evidence Estimate')
        axes[0, 1].legend()
    
    # Number of live points
    axes[1, 0].hist(result['samples'][:, 0], bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Semi-major axis [GM/c²]')
    axes[1, 0].set_ylabel('Posterior samples')
    axes[1, 0].set_title('Marginal distribution of a')
    
    # Effective sample size
    if 'ess' in result:
        axes[1, 1].bar(range(len(param_names)), result['ess'])
        axes[1, 1].set_xlabel('Parameter')
        axes[1, 1].set_ylabel('Effective Sample Size')
        axes[1, 1].set_title('ESS by Parameter')
        axes[1, 1].set_xticklabels(param_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSampling complete. Results saved to {output_dir}/")
    print(f"Log evidence: {result['logz']:.2f} ± {result['logzerr']:.2f}")
    print(f"Number of likelihood evaluations: {sampler.ncall}")
    
