import numpy as np, cupy as cp, emcee, matplotlib.pyplot as plt, h5py, heapq, corner
from pn_trajectory import residuals
from scipy.stats import binned_statistic_2d
from emcee.backends import HDFBackend

def log_prior(theta, param_lims):
    valid = np.logical_and.reduce([(theta[:, i] >= lim[0]) & (theta[:, i] <= lim[1]) for i, lim in enumerate(param_lims)])
    lp = np.where(valid, 0, -np.inf)
    return lp

def log_likelihood(theta, timings, windows, errs):
    resid = residuals(timings, windows, errs, *theta.T)
    ll = -np.log(resid.get())
    return ll

def log_prob(theta, param_lims, timings, windows, errs):
    lp = log_prior(theta, param_lims)
    ll = log_likelihood(theta, timings, windows, errs)
    result = lp + ll
    result[~np.isfinite(result)] = -np.inf
    return result

if __name__ == '__main__':
    timings = cp.loadtxt('timings_kg.dat')
    windows = np.loadtxt('windows.dat', ndmin=2)
    errs = cp.ones_like(timings)

    print(log_likelihood(np.array([[120, 0.3, 80, 0.9, 6, np.pi/4, 0, 0, 1e8],
                                   [180, 0.3, 80, 0.9, 6, np.pi/4, 0, 0, 1e8]]).reshape(-1, 9), timings, windows, errs))
    # lls = [log_likelihood(np.vstack([360, 0.3, 80, 0.5, phir0, 0, 0, 10, 0, 2e6]*2).reshape(-1, 10), timings, windows, errs)[0] for phir0 in np.linspace(0, 2*np.pi, 30)]
    # plt.plot(np.linspace(0, 2*np.pi, 30), lls)
    # plt.savefig('lls.png')
    # plt.close()
    # print(lls)

    sma_lims = (100, 400)
    e_lims = (0, 0.9)
    incl_lims = (30, 90)
    a_lims = (0., 0.998)
    logMbh_lims = (5.6, 6.4)
    theta_obs_lims = (0, 2*np.pi)
    theta_d_lims = (0, 30)
    phi_d_lims = (-np.pi, np.pi)
    P_d_lims = (.999e8, 1.001e8)
    param_lims = [sma_lims, e_lims, incl_lims, a_lims, logMbh_lims, theta_obs_lims, theta_d_lims, phi_d_lims, P_d_lims]
    ndim = len(param_lims)
    nsteps = 2000
    topk = 5000
    nwalkers = 1000
    rng = np.random.default_rng()
    p0 = np.column_stack([rng.uniform(lim[0], lim[1], size=nwalkers) for lim in param_lims])

    filename = "samples_20.h5"
    backend = HDFBackend(filename)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[param_lims, timings, windows, errs], backend=backend, vectorize=True)
    # p0 = sampler.chain[-1]
    backend.reset(nwalkers, ndim)
    sampler.run_mcmc(p0, nsteps, progress=True)

    chunk_size = 100
    with h5py.File(filename, "r") as f:
        lnprob_dset = f['mcmc']["log_prob"]    # shape: (nsteps, nwalkers)
        nsteps, nwalkers = lnprob_dset.shape
        discard = int(nsteps * 0.5)             # discard first 50% as burn-in
        min_heap = []
        for i in range(discard, nsteps, chunk_size):
            chunk = lnprob_dset[i:min(i + chunk_size, nsteps), :]
            flat_chunk = chunk.flatten()
            for val in flat_chunk:
                if len(min_heap) < topk:
                    heapq.heappush(min_heap, val)
                else:
                    if val > min_heap[0]:
                        heapq.heapreplace(min_heap, val)
        threshold = min_heap[0]

    print("Threshold for top-k lnprob samples:", threshold)
    discard = 1000

    samples_list = []
    lnprob_kept_list = []
    with h5py.File(filename, "r") as f:
        chain_dset = f['mcmc']["chain"]      # shape: (nsteps, nwalkers, ndim)
        lnprob_dset = f['mcmc']["log_prob"]    # shape: (nsteps, nwalkers)
        nsteps, nwalkers, ndim = chain_dset.shape

        for i in range(discard, nsteps, chunk_size):
            # Load a chunk of chain and lnprob data
            chain_chunk = chain_dset[i:min(i + chunk_size, nsteps), :, :]
            lnprob_chunk = lnprob_dset[i:min(i + chunk_size, nsteps), :]
            
            # Flatten over iterations and walkers
            chain_flat = chain_chunk.reshape(-1, ndim)
            lnprob_flat = lnprob_chunk.reshape(-1)
            
            # Create a mask for samples above the threshold
            mask = lnprob_flat > threshold
            if np.any(mask):
                samples_list.append(chain_flat[mask])
                lnprob_kept_list.append(lnprob_flat[mask])
        
        posterior_topk = np.concatenate(samples_list, axis=0)
        lnprob_topk = np.concatenate(lnprob_kept_list, axis=0)

    print("Posterior samples shape:", posterior_topk.shape)

    num_bins = 500
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        posterior_topk[:, 0], posterior_topk[:, 1], lnprob_topk,
        statistic='mean', bins=num_bins
    )

    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(x_edges, y_edges, stat.T, shading='auto', cmap='viridis')
    plt.xlabel('posterior_sma')
    plt.ylabel('posterior_e')
    plt.colorbar(mesh) 
    plt.savefig('lnprob_map.png')
    plt.close()

    plt.figure()
    plt.hist(lnprob_topk, bins=50)
    plt.savefig('lnprob_hist.png')
    plt.close()

    labels = ["sma", "e", "incl", "a", "$\\log M_{\\rm BH}$", "$\\theta_{\\rm obs}$", "$\\theta_d$", "$\\phi_d$", "P_d"]
    fig = corner.corner(posterior_topk, labels=labels, smooth=2, show_titles=True)
    fig.savefig("corner_plot.png")
    plt.close()