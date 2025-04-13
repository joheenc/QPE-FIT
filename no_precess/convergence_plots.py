import h5py, matplotlib.pyplot as plt, heapq, numpy as np

topk = 10000
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
nqpes = [0]
nbins = 10
for nqpe in nqpes:
    # filename = f'sma=100_e=0.3_a=0.9_Mbh=5_windows={nqpe}.h5'
    filename = f'tmp.h5'
    print(filename)
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
    
    axes[0].hist(posterior_topk[:, 0], label=f'n={nqpe}', bins=nbins, histtype='step', linewidth=1.5)
    axes[1].hist(posterior_topk[:, 1], label=f'n={nqpe}', bins=nbins, histtype='step', linewidth=1.5)
    axes[2].hist(posterior_topk[:, 3], label=f'n={nqpe}', bins=nbins, histtype='step', linewidth=1.5)
    axes[3].hist(posterior_topk[:, 4], label=f'n={nqpe}', bins=nbins, histtype='step', linewidth=1.5)
    print(posterior_topk.shape)
params = ['sma', 'ecc', 'a', 'log(Mbh)']
true_vals = [100, 0.7, 0.5, 6]
for param, true_val, ax in zip(params, true_vals, axes):
    ax.set_title(param)
    ax.vlines(true_val, *ax.get_ylim(), ls='--', color='k')
    ax.legend()
plt.tight_layout()
plt.savefig('posteriors.png')
plt.close()

import corner

labels = ['sma', 'ecc', 'incl', 'a', 'log(Mbh)', "$\\theta_{\\rm obs}$"]
fig = corner.corner(posterior_topk, labels=labels, show_titles=True)
fig.savefig("corner_plot.png")
plt.close()
