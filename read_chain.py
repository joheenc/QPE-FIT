import h5py, heapq, numpy as np, corner, sys

filename = sys.argv[1]
topk = float(sys.argv[2])
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

labels = ['$a$', '$e$', '$i$', r'$a_\bullet$', r'$\log M_\bullet$', r'$\theta_{\rm obs}$', r'$\theta_{\rm disk}$', r'$T_{\rm disk}/P_{\rm orb}$', r'$\phi_{\rm disk, 0}$', '$t_0$']
fig = corner.corner(posterior_topk, labels=labels, show_titles=True, smooth=2)
fig.savefig("corner_plot.png")
