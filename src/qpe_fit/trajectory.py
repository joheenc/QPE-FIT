import numpy as np

xp = np
USE_GPU = False

def to_numpy(arr):
    return arr.get() if USE_GPU else arr

def _alpha(e, v, q, Y):
	alpha_0 = 1 + e**2 * (1/2 - 1/2 * v**2 + q * Y * v**3 + (-3 + (1/2 - Y**2) * q**2) * v**4 \
		+ 10 * q * Y * v**5 + (-18 + (11/2 - 18 * Y**2) * q**2) * v**6) \
		+ e**4 * (3/8 - 3/8 * v**2 + 3/4 * q * Y * v**3 + (-33/16 + (5/16 - 11/16 * Y**2) * q**2) * v**4 \
		+ 27/4 * q * Y * v**5 + (-189/16 + (-185/16 * Y**2 + 61/16) * q**2) * v**6) \
		+ e**6 * (5/16 - 5/16 * v**2 + 5/8 * q * Y * v**3 + (-27/16 + (1/4 - 9/16 * Y**2) * q**2) * v**4 \
		+ 11/2 * q * Y * v**5 + (-19/2 + (-75/8 * Y**2 + 49/16) * q**2) * v**6)

	alpha_1 = e + e**3 * (3/4 - 1/2 * v**2 + Y * q * v**3 + (-51/16 + (7/16 - 15/16 * Y**2) * q**2) * v**4 \
		+ 43/4 * Y * q * v**5 + (-81/4 + (-153/8 * Y**2 + 11/2) * q**2) * v**6) \
		+ e**5 * (5/8 - 1/2 * v**2 + Y * q * v**3 + (-93/32 + (13/32 - 29/32 * Y**2) * q**2) * v**4 \
		+ 77/8 * Y * q * v**5 + (-277/16 + (-133/8 * Y**2 + 83/16) * q**2) * v**6)

	alpha_2 = e**2 * (0.5 + 0.5 * v**2 - Y * q * v**3 + (3 + (Y**2 - 1/2) * q**2) * v**4 \
		- 10 * Y * q * v**5 + (18 + (18 * Y**2 - 11/2) * q**2) * v**6) \
		+ e**4 * (1/2 - 1/2 * v**4 + 2 * Y * q * v**5 + (-11/2 + (-4 * Y**2 + 1/2) * q**2) * v**6) \
		+ e**6 * (15/32 - 5/32 * v**2 + 5/16 * Y * q * v**3 + (-39/32 + (1/8 - 9/32 * Y**2) * q**2) * v**4 \
		+ 17/4 * Y * q * v**5 + (-279/32 + (-243/32 * Y**2 + 2) * q**2) * v**6)

	alpha_3 = e**3 * (1/4 + v**2 / 2 - Y * q * v**3 + (51/16 + (-7/16 + 15/16 * Y**2) * q**2) * v**4 \
		- 43/4 * Y * q * v**5 + (81/4 + (153/8 * Y**2 - 11/2) * q**2) * v**6) \
		+ e**5 * (5/16 + v**2 / 4 - Y * q * v**3 / 2 + (69/64 + (-13/64 + 29/64 * Y**2) * q**2) * v**4 \
		- 53/16 * Y * q * v**5 + (135/32 + (43/8 * Y**2 - 69/32) * q**2) * v**6)

	alpha_4 = e**4 * (1/8 + 3/8 * v**2 - 3/4 * Y * q * v**3 + (41/16 + (-5/16 + 11/16 * Y**2) * q**2) * v**4 \
		- 35/4 * Y * q * v**5 + (277/16 + (249/16 * Y**2 - 69/16) * q**2) * v**6) \
		+ e**6 * (3/16 + 5/16 * v**2 - 5/8 * Y * q * v**3 + (27/16 + (-1/4 + 9/16 * Y**2) * q**2) * v**4 \
		- 11/2 * Y * q * v**5 + (9 + (75/8 * Y**2 - 49/16) * q**2) * v**6)

	alpha_5 = e**5 * (1/16 + v**2 / 4 - Y * q * v**3 / 2 + (117/64 + (-13/64 + 29/64 * Y**2) * q**2) * v**4 \
		- 101/16 * Y * q * v**5 + (419/32 + (45/4 * Y**2 - 97/32) * q**2) * v**6)

	alpha_6 = e**6 * (1/32 + 5/32 * v**2 - 5/16 * Y * q * v**3 + (39/32 + (-1/8 + 9/32 * Y**2) * q**2) * v**4 \
		- 17/4 * Y * q * v**5 + (295/32 + (243/32 * Y**2 - 2) * q**2) * v**6)

	return alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6

def beta(e, v, q, Y, nparams):
	beta_1 = 1 + (1/16 - 9/16 * Y**2) * q**2 * v**4 + (-1/4 + 9/4 * Y**2) * q**2 * v**6 \
		+ e**2 * ((-1/16 + 9/16 * Y**2) * q**2 * v**4 + (-9/4 * Y**2 + 1/4) * q**2 * v**6)

	beta_3 = (1 - Y**2) / 16 * q**2 * v**4 - (1-Y**2) / 4 * q**2 * v**6 + e**2 * (-(1 - Y**2) / 16 * q**2 * v**4 + (1 - Y**2) / 4 * q**2 * v**6)
	_z = xp.zeros_like(beta_1)
	return _z, beta_1, _z, beta_3

def _v_p_t(e, v, q, Y):
	v_p_t1_r = e*(2 + 4*v**2 - 6*Y*q*v**3 + (17 + (4*Y**2 - 1)*q**2) * v**4 - 54*Y*q*v**5 \
		+ (88 + (84*Y**2 - 20)*q**2)*v**6) \
		+ e**3 * (3 + 3*v**2 - 4*Y*q*v**3 + (77/8 + (21/8*Y**2 - 5/8)*q**2) * v**4 - 57/2*Y*q*v**5 \
		+ (173/4 + (42*Y**2 - 51/4) * q**2) * v**6) \
		+ e**5*(15/4 + 5/2*v**2 - 13/4*Y*q*v**3 + (15/2 + (17/8*Y**2 - 1/2)*q**2)*v**4 - 45/2*Y*q*v**5 \
		+ (67/2 + (133/4*Y**2 - 10)*q**2)*v**6)

	v_p_t2_r = e**2 * (3/4 + 7/4*v**2 - 13/4*Y*q*v**3 + (81/8 + (5/2*Y**2 - 7/8) * q**2) * v**4 - 135/4*Y*q*v**5 \
		+ (499/8 + (55*Y**2 - 113/8) * q**2) * v**6) \
		+ e**4 * (5/4 + 7/4*v**2 - 3*Y*q*v**3 + (131/16 + (37/16*Y**2 - 13/16)*q**2)*v**4 - 103/4*Y*q*v**5 \
		+ (691/16 + (655/16*Y**2 - 197/16) * q**2) * v**6) \
		+ e**6 * (105/64 + 105/64*v**2 - 175/64*Y*q*v**3 + (905/128 + (135/64*Y**2 - 95/128)*q**2)*v**4 - 1413/64*Y*q*v**5 \
		+ (4591/128 + (2241/64*Y**2 - 1389/128)*q**2)*v**6)

	v_p_t3_r = e**3 * (1/3 + v**2 - 2*Y*q*v**3 + (53/8 + (13/8 * Y**2 - 5/8) * q**2) * v**4 - 45/2*Y*q*v**5 \
		+ (523/12 + (38*Y**2 - 39/4)*q**2)*v**6) \
		+ e**5 * (5/8 + 5/4*v**2 - 19/8*Y*q*v**3 + (7 + (31/16*Y**2 - 3/4)*q**2)*v**4 - 91/4*Y*q*v**5 \
		+ (647/16 + (601/16*Y**2 - 175/16)*q**2)*v**6)

	v_p_t4_r = e**4 * (5/32 + 19/32*v**2 - 39/32*Y*q*v**3 + (137/32 + (65/64*Y**2 - 13/32)*q**2)*v**4 - 473/32*Y*q*v**5 \
		+ (957/32 + (1631/64*Y**2 - 207/32)*q**2)*v**6) \
		+ e**6 * (21/64 + 57/64*v**2 - 113/64 * Y*q*v**3 + (89/16 + (189/128*Y**2 - 19/32)*q**2)*v**4 - 1185/64*Y*q*v**5 \
		+ (553/16 + (4005/128*Y**2 - 141/16) * q**2) * v**6)

	v_p_t5_r = e**5 * (3/40 + 7/20*v**2 - 29/40*Y*q*v**3 + (27/10 + (49/80*Y**2 - 1/4)*q**2)*v**4 - 189/20*Y*q*v**5 \
		+ (319/16 + (1321/80*Y**2 - 331/80)*q**2)*v**6)

	v_p_t6_r = e**6 * (7/192 + 13/64*v**2 - 27/64*Y*q*v**3 + (213/128 + (23/64*Y**2 - 19/128)*q**2)*v**4 - 377/64*Y*q*v**5 \
		+ (4969/384 + (665/64*Y**2 - 329/128)*q**2)*v**6)
	return v_p_t1_r, v_p_t2_r, v_p_t3_r, v_p_t4_r, v_p_t5_r, v_p_t6_r

def t_theta_p(e, v, q, Y, nparams):
	t2_theta_p = (Y**2-1)/4 * q**2*v**3 - (Y**2-1)/2 * q**2*v**5 + Y*(Y**2-1)/4 * (3+e**2)*q**3*v**6
	return xp.zeros_like(t2_theta_p), t2_theta_p

def _phi_r(e, v, q, Y):
	phi1_r = e*(-2*q*v**3 + 2*Y*q**2*v**4 - 10*q*v**5 + 18*Y*q**2*v**6)

	phi2_r = e**2 * (-1/4 *Y*q**2*v**4 + 1/2*q*v**5 - 3/4*Y*q**2*v**6)
	return phi1_r, phi2_r

def X_R(e, v, q, Y, nparams):
	X0_R = ((1+Y)/2 - (9*Y-1)*(Y**2-1)/32 * q**2*v**4 + (9*Y-1)*(Y**2-1)/8 * q**2*v**6) \
		+ e**2 * ((9*Y-1)*(Y**2-1)/32 * q**2*v**4 - (9*Y-1)*(Y**2-1)/8 * q**2*v**6)

	X2_R = ((1-Y)/2 + Y*(Y**2-1)/4 * q**2*v**4 - Y*(Y**2-1)*q**2*v**6) \
		+ e**2*(-Y*(Y**2-1)/4 * q**2*v**4 + Y*(Y**2-1)*q**2*v**6)

	X4_R = ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6)

	return X0_R, xp.zeros_like(X0_R), X2_R, xp.zeros_like(X0_R), X4_R

def X_J(e, v, q, Y, nparams):
	X2_J = ((Y-1)/2 - (5*Y+1)*(Y**2-1)/16 * q**2*v**4 + (5*Y+1)*(Y**2-1) / 4 * q**2*v**6) \
		+ e**2*((5*Y+1)*(Y**2-1) / 16 * q**2*v**4 - (5*Y+1)*(Y**2-1) / 4 * q**2*v**6)

	X4_J = (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6)
	_z = xp.zeros_like(X2_J)
	return _z, _z, X2_J, _z, X4_J

def _Omegas(p, e, v, q, Y):
	Omega_t = p**2*(1 + 3/2*e**2 + 15/8*e**4 + 35/16*e**6 + (3/2 - 1/4*e**2 - 15/16*e**4 - 45/32*e**6)*v**2 \
		+ (2*Y*q*e**2 + 3*Y*q*e**4 + 15/4*Y*q*e**6)*v**3 \
		+ (27/8 - 1/2*Y**2*q**2 + 1/2*q**2 + (-99/16 + q**2 - 2*Y**2*q**2)*e**2 \
		+ (-567/64 + 21/16*q**2 - 45/16*Y**2*q**2)*e**4 + (-1371/128 + 25/16*q**2 - 55/16*Y**2*q**2)*e**6)*v**4 \
		+ (-3*Y*q + 43/2*Y*q*e**2 + 231/8*Y*q*e**4 + 555/16*Y*q*e**6)*v**5 \
		+ (135/16 - 1/4*q**2 + 3/4*Y**2*q**2 + (-1233/32 + 47/4*q**2 - 75/2*Y**2*q**2)*e**2 \
		+ (-6567/128 + 499/32*q**2 - 1577/32*Y**2*q**2)*e**4 \
		+ (-15565/256 + 75/4*q**2 - 1887/32*Y**2*q**2)*e**6)*v**6)

	Omega_r = p*v*(1 + (-3/2 + 1/2*e**2)*v**2 + (3*Y*q - Y*q*e**2)*v**3 \
		+ (-45/8 + 1/2*q**2 - 2*Y**2*q**2 + (1/4*q**2 + 1/4*Y**2*q**2)*e**2 + 3/8*e**4)*v**4 \
		+ (33/2*Y*q + 2*Y*q*e**2 - 3/2*Y*q*e**4)*v**5 \
		+ (-351/16 - 51/2*Y**2*q**2 + 33/4*q**2 + (-135/16 + 7/8*q**2 - 39/8 * Y**2*q**2)*e**2 \
		+ (21/16 + 1/8*q**2 + 13/8 * Y**2*q**2)*e**4 + 5/16*e**6) * v**6)

	Omega_theta = p*v*(1 + (3/2 + 1/2*e**2)*v**2 - (3*Y*q + Y*q*e**2)*v**3 \
		     + (27/8 + 7/4*Y**2*q**2 - 1/4*q**2 + (9/4 + 1/4*q**2 + 1/4*Y**2*q**2)*e**2 + 3/8*e**4)*v**4 \
		     - (15/2*Y*q + 7*Y*q*e**2 + 3/2*Y*q*e**4)*v**5 \
		     + (135/16 + 57/8*Y**2*q**2 - 27/8*q**2 + (135/16 - 19/4*q**2 + 45/4*Y**2*q**2)*e**2 \
		        + (45/16 + 1/8*q**2 + 13/8*Y**2*q**2)*e**4 + 5/16*e**6)*v**6)
		        
	Omega_phi = p*v*(1 + (3/2 + 1/2*e**2)*v**2 + (2*q - 3*Y*q - Y*q*e**2)*v**3 \
		+ (-3/2*Y*q**2 + 7/4*Y**2*q**2 - 1/4*q**2 + 27/8 + (9/4 + 1/4*q**2 + 1/4*Y**2*q**2)*e**2 + 3/8*e**4)*v**4 \
		+ (3*q - 15/2*Y*q + (4*q - 7*Y*q)*e**2 - 3/2*Y*q*e**4)*v**5 \
		+ (-9/4*Y*q**2 + 57/8*Y**2*q**2 + 135/16 - 27/8*q**2 + (135/16 - 19/4*q**2 - 35/4*Y*q**2 + 45/4*Y**2*q**2)*e**2 \
		+ (45/16 + 1/8*q**2 + 13/8*Y**2*q**2)*e**4 + 5/16*e**6)*v**6)

	return Omega_t, Omega_r, Omega_theta, Omega_phi

def _cartesian(r, cos_theta, phi):
    sin_theta = (1-cos_theta**2)**(1/2)
    return r * sin_theta * xp.cos(phi), r * sin_theta * xp.sin(phi), r * cos_theta

alpha = _alpha
v_p_t = _v_p_t
phi_r = _phi_r
Omegas = _Omegas
cartesian = _cartesian

def set_backend(gpu=False):
    global xp, USE_GPU, alpha, v_p_t, phi_r, Omegas, cartesian
    if gpu:
        import cupy
        xp = cupy
        USE_GPU = True
        alpha = cupy.fuse(_alpha)
        v_p_t = cupy.fuse(_v_p_t)
        phi_r = cupy.fuse(_phi_r)
        Omegas = cupy.fuse(_Omegas)
        cartesian = cupy.fuse(_cartesian)


def _cumtrapz_time(y, x):
    """Cumulative trapezoidal integral of ``y`` w.r.t. ``x`` along the time axis.

    ``y`` and ``x`` both have shape ``(nset, ntime)``.  Returns an array of the
    same shape whose first column is zero (integral measured from the first
    sample).  Implemented with ``cumsum`` so it behaves identically on the NumPy
    and CuPy backends.
    """
    seg = 0.5 * (y[:, 1:] + y[:, :-1]) * (x[:, 1:] - x[:, :-1])
    out = xp.zeros_like(y)
    out[:, 1:] = xp.cumsum(seg, axis=1)
    return out


def trajectory(windows, logMbh, sma, ecc, incl, spin, phi_r0, phi_theta0, phi_phi0, dt, adot=0.0, edot=0.0):
    """Adiabatic (slowly-evolving) post-Newtonian Kerr trajectory.

    The semi-major axis and eccentricity are promoted to slowly-varying
    functions of *observer* time,

        a(t) = sma + adot * t ,   e(t) = ecc + edot * t ,

    with ``t`` in seconds measured from ``t = 0`` (so ``sma``/``ecc`` are the
    values at ``t = 0``).  ``adot`` has units of Rg/s and ``edot`` units of 1/s.
    With ``adot = edot = 0`` the model reduces exactly to the original
    fixed-geodesic QPE-FIT trajectory.

    Self-consistency: the fundamental frequencies and the Fourier amplitudes are
    re-evaluated at the *instantaneous* (a(t), e(t)), and the orbital phases are
    obtained by integrating the instantaneous frequencies,

        Phi_i(t) = phi_i0 + \\int_0^t (Omega_i / Omega_t) dt' ,

    so the apsidal and Lense-Thirring (nodal) precessions accelerate
    self-consistently as the orbit evolves -- unlike a timing-level
    period-derivative remap, which leaves the precession frequencies frozen.
    """
    sma = xp.asarray(sma)
    ecc = xp.asarray(ecc)
    logMbh = xp.asarray(logMbh)
    q = xp.asarray(spin)[:, None]
    Y = xp.cos(xp.radians(xp.asarray(incl)))[:, None]
    phi_r0 = xp.asarray(phi_r0)[:, None]
    phi_theta0 = xp.asarray(phi_theta0)[:, None]
    phi_phi0 = xp.asarray(phi_phi0)
    nparams = sma.shape[0]

    adot = xp.atleast_1d(xp.asarray(adot, dtype=sma.dtype))
    edot = xp.atleast_1d(xp.asarray(edot, dtype=sma.dtype))
    if adot.shape[0] == 1 and nparams != 1:
        adot = xp.full(nparams, adot[0])
    if edot.shape[0] == 1 and nparams != 1:
        edot = xp.full(nparams, edot[0])

    M = 4.926580927874239e-06 * 10**logMbh                       # seconds per M  (nset,)

    # --- observer-time grid (phases integrated from the common absolute origin) ---
    # Each observation window is sampled uniformly at `dt`.  Consecutive windows
    # are separated by long gaps (up to ~1e7 s).  The orbital phases come from
    # integrating Omega_i / Omega_t over time; a single trapezoid spanning a whole
    # gap mis-accumulates that phase once the orbit evolves secularly
    # (adot/edot != 0), because the frequency ratios then drift across the gap --
    # so a later window's trajectory would depend on whether earlier windows are
    # present (errors of tens of Rg in r for realistic rates).  The ratios are
    # smooth (a(t), e(t) are linear in t; no orbital oscillation enters them), so a
    # fixed handful of sub-samples per gap integrates them accurately.  These gap
    # samples exist only to keep the cumulative phase integral correct across the
    # gaps and are dropped from the returned trajectory via `real_mask`.
    #
    # When the orbit does NOT evolve (adot = edot = 0, the default) the ratios are
    # exactly constant, a single trapezoid is exact, and gap sub-sampling buys
    # nothing -- so it is skipped and the grid (and cost) are bit-identical to the
    # original fixed-geodesic model.  The sub-sampling is paid for only when
    # secular evolution is actually being explored.
    secular = bool(xp.any(adot != 0)) or bool(xp.any(edot != 0))
    if secular:
        N_GAP = 256                                             # sub-samples per gap
        grid_parts, mask_parts = [], []
        prev_stop = None
        for start, stop in windows:
            wg = xp.arange(start, stop, dt)
            if prev_stop is not None:
                fill = xp.linspace(prev_stop, wg[0], N_GAP + 2)[1:-1]   # gap interior only
                grid_parts.append(fill)
                mask_parts.append(xp.zeros(fill.shape[0], dtype=bool))
            grid_parts.append(wg)
            mask_parts.append(xp.ones(wg.shape[0], dtype=bool))
            prev_stop = wg[-1]
        t_sec = xp.concatenate(grid_parts)
        real_mask = xp.concatenate(mask_parts)
    else:
        t_sec = xp.concatenate([xp.arange(start, stop, dt) for start, stop in windows])
        real_mask = None                                        # no gap-fill to drop
    t_g = t_sec[None, :] / M[:, None]                            # geometric coord time (nset, ntime)

    # --- agnostic secular evolution of the orbital elements (linear in observer time) ---
    e = xp.clip(ecc[:, None] + edot[:, None] * t_sec[None, :], 1e-6, 0.999)
    a = xp.clip(sma[:, None] + adot[:, None] * t_sec[None, :], 1e-3, None)
    p = a * (1 - e**2)
    v = (1 / p)**(1/2)

    Omega_t, Omega_r, Omega_theta, Omega_phi = Omegas(p, e, v, q, Y)

    # Accumulated orbital phases.  These replace the static-model products
    # Omega_i * lambda; with constant frequencies cumtrapz(Omega_i/Omega_t, t_g)
    # reduces to (Omega_i/Omega_t) * t_g, recovering the original expressions.
    Psi_r = phi_r0 + _cumtrapz_time(Omega_r / Omega_t, t_g)
    Psi_theta = phi_theta0 + _cumtrapz_time(Omega_theta / Omega_t, t_g)
    Psi_phi = _cumtrapz_time(Omega_phi / Omega_t, t_g)
    lambd = _cumtrapz_time(1.0 / Omega_t, t_g)                   # Mino time (returned for reference)

    # --- coordinate time: secular part is t_g, plus the oscillatory r/theta corrections ---
    n_r = xp.arange(1, 7, dtype=xp.float32).reshape(1, 6, 1)
    sin_r_terms = xp.sin(n_r * Psi_r[:, None, :])
    amp_vpt = (p / v)[:, None, :] * xp.stack(v_p_t(e, v, q, Y), axis=1)
    summation_r = xp.einsum('ijk,ijk->ik', amp_vpt, sin_r_terms)
    n_theta = xp.arange(1, 3).reshape(1, 2, 1)
    sin_theta_terms = xp.sin(n_theta * Psi_theta[:, None, :])
    amp_ttheta = p[:, None, :] * xp.stack(t_theta_p(e, v, q, Y, nparams), axis=1)
    summation_theta = xp.einsum('ijk,ijk->ik', amp_ttheta, sin_theta_terms)
    t = t_g + summation_r + summation_theta
    t = t - t[:, 0][:, None]

    # --- radial coordinate r(t) ---
    n_r = xp.arange(7, dtype=xp.float32).reshape(1, 7, 1)
    cos_terms = xp.cos(n_r * Psi_r[:, None, :])
    summation = xp.einsum('ijk,ijk->ik', xp.stack(alpha(e, v, q, Y), axis=1), cos_terms)
    r = p * summation

    # --- polar coordinate cos(theta)(t) ---
    n_theta = xp.arange(4).reshape(1, 4, 1)
    sin_terms = xp.sin(n_theta * Psi_theta[:, None, :])
    summation = xp.einsum('ijk,ijk->ik', xp.stack(beta(e, v, q, Y, nparams), axis=1), sin_terms)
    cos_theta = xp.sqrt(1 - Y**2) * summation

    # --- azimuthal coordinate phi(t) ---
    n_r = xp.arange(1, 3).reshape(1, 2, 1)
    n_theta = xp.arange(5).reshape(1, 5, 1)
    sin_r_terms2 = xp.sin(n_r * Psi_r[:, None, :])
    summation_r3 = xp.einsum('ijk,ijk->ik', xp.stack(phi_r(e, v, q, Y), axis=1), sin_r_terms2)
    sin_theta_terms2 = xp.sin(n_theta * Psi_theta[:, None, :])
    cos_theta_terms2 = xp.cos(n_theta * Psi_theta[:, None, :])
    XR_stack = xp.stack(X_R(e, v, q, Y, nparams), axis=1)
    XJ_stack = xp.stack(X_J(e, v, q, Y, nparams), axis=1)
    summation_theta3 = p * (xp.einsum('ijk,ijk->ik', XR_stack, cos_theta_terms2) + xp.einsum('ijk,ijk->ik', 1j * XJ_stack, sin_theta_terms2))
    phi = Psi_phi + summation_r3 + xp.angle(summation_theta3) + phi_phi0[:, None]

    P_orb = 2*xp.pi / Omega_phi[:, 0] * Omega_t[:, 0] * M

    if real_mask is not None:
        t = t[:, real_mask]
        r = r[:, real_mask]
        cos_theta = cos_theta[:, real_mask]
        phi = phi[:, real_mask]
        lambd = lambd[:, real_mask]
    return t, r, cartesian(r, cos_theta, phi), lambd, P_orb


# ---------------------------------------------------------------------------
# Optional physics.
#
# Every effect below is OFF by default and switched on only when its parameter
# is supplied.  The value listed here is the *disabling* default: passing it
# (or simply omitting the parameter) reproduces the simpler model exactly, so a
# parameter absent from priors.json (sampling) or params.json (injection) is
# just not explored.  ns.py and generate_timings.py both read this registry.
#   theta_d  disk inclination  [deg]   -> 0    : disk in the equatorial plane
#   P_d      precession period [Porb]  -> inf  : non-precessing disk
#   phi_d    disk azimuthal phase [rad]-> 0
#   r_warp   warp radius       [Rg]    -> 0    : flat (unwarped) disk
#   adot     da/dt             [Rg/s]  -> 0    : no semimajor-axis evolution
#   edot     de/dt             [1/s]   -> 0    : no eccentricity evolution
OPTIONAL_DEFAULTS = {
    'theta_d': 0.0,
    'P_d': float('inf'),
    'phi_d': 0.0,
    'r_warp': 0.0,
    'adot': 0.0,
    'edot': 0.0,
}


def _as_param_array(x, n, dtype):
    """Broadcast a scalar / length-1 / length-n value to a ``(n,)`` backend array."""
    x = xp.atleast_1d(xp.asarray(x, dtype=dtype))
    if x.shape[0] == 1 and n != 1:
        x = xp.full(n, x[0], dtype=dtype)
    return x


def disk_normal(t_sec, r, theta_d, P_d_sec, phi_d, r_warp):
    """Unit normal of the accretion disk along the orbit, with optional tilt,
    precession, and warp.

    ``theta_d`` [rad], ``P_d_sec`` [s], ``phi_d`` [rad] and ``r_warp`` [Rg] are
    per-set arrays of shape ``(nset,)``; ``t_sec`` and ``r`` are the
    ``(nset, ntime)`` time [s] and radius [Rg] grids.  Returns the three
    Cartesian components of the unit normal, each ``(nset, ntime)``.

    Warp model (a Bardeen-Petterson-like radial twist + alignment):
        gamma_w = 2 sqrt(r_warp / r)                  radius-dependent warp angle
        sin(beta) = sin(theta_d) e^{-gamma_w}         local tilt (-> 0 at small r)
        azimuth = gamma_w + 2 pi t / P_d + phi_d       warp twist + nodal precession
    With ``r_warp = 0`` gamma_w vanishes and this reduces *exactly* to a flat,
    rigidly precessing disk; with ``P_d_sec = inf`` the precession term drops and
    the disk is static; with ``theta_d = 0`` the normal is the spin (z) axis.

    Provenance & scope.  The equal-magnitude alignment/twist coupling of the
    complex tilt W = beta e^{i gamma} ~ e^{-(1-i) gamma_w}, gamma_w ~ r^{-1/2}, is
    the canonical diffusive steady-state warp of Scheuer & Feiler (1996) (see also
    Kumar & Pringle 1985), not an ad hoc profile.  Caveats:
      - ``r_warp`` is a free *phenomenological* scale; it is not derived from the
        spin / alpha / (H/R) as in first-principles Bardeen-Petterson theory.
      - The prefactor 2 in gamma_w carries no independent meaning -- it is exactly
        degenerate with the fitted ``r_warp`` (2 sqrt(r_warp/r) = sqrt(4 r_warp/r)).
      - This layers a *steady* warp on *rigid* precession (an effective hybrid in
        the diffusive regime); it does not model bending-wave, anti-alignment, or
        disk-tearing regimes.
      - ``cos_beta = sqrt(1 - sin_beta**2) >= 0``, so the warped normal stays in
        the +z hemisphere; the "reduces exactly to a flat disk" statement and the
        supported tilt range are therefore ``theta_d in [0, 90 deg]`` (the prior
        caps theta_d at 80 deg).
    """
    gamma_w = 2.0 * xp.sqrt(xp.maximum(r_warp[:, None] / r, 0.0))
    azimuth = gamma_w + 2.0 * xp.pi * t_sec / P_d_sec[:, None] + phi_d[:, None]
    sin_beta = xp.sin(theta_d[:, None]) * xp.exp(-gamma_w)
    cos_beta = xp.sqrt(1.0 - sin_beta**2)
    return sin_beta * xp.cos(azimuth), sin_beta * xp.sin(azimuth), cos_beta


def residuals(timings, windows, errs, sma, e, incl, phi_r0, phi_theta0, phi_phi0,
              a, logMbh, theta_obs, dt, one_per_orbit,
              theta_d=0.0, P_d=float('inf'), phi_d=0.0, r_warp=0.0, adot=0.0, edot=0.0):
    """QPE-timing chi-square residuals.

    The orbit (``sma, e, incl, a, logMbh`` + initial phases, with optional
    ``adot``/``edot`` secular evolution) is always modelled.  The disk geometry
    is optional: ``theta_d``, ``P_d``, ``phi_d`` and ``r_warp`` each default to a
    value that switches the corresponding effect off (see ``OPTIONAL_DEFAULTS``),
    so an aligned / static / flat disk is obtained simply by not supplying them.
    """
    sma = xp.asarray(sma)
    nparams = sma.shape[0]
    dtype = sma.dtype
    t, r, (x, y, z), _, P_orb = trajectory(windows, logMbh, sma, e, incl, a, phi_r0, phi_theta0, phi_phi0, dt, adot=adot, edot=edot)
    theta_d = xp.radians(_as_param_array(theta_d, nparams, dtype))
    P_d = _as_param_array(P_d, nparams, dtype) * P_orb
    phi_d = _as_param_array(phi_d, nparams, dtype)
    r_warp = _as_param_array(r_warp, nparams, dtype)
    t_g = 4.926580927874239e-06 * 10**xp.asarray(logMbh)
    t = t * t_g[:, None]
    theta_obs = xp.asarray(theta_obs)
    n_obs = xp.column_stack((xp.sin(theta_obs), xp.zeros_like(theta_obs), xp.cos(theta_obs)))
    n_crs_x, n_crs_y, n_crs_z = disk_normal(t, r, theta_d, P_d, phi_d, r_warp)
    D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z
    if one_per_orbit:
        crossings_mask = (D_t[:, :-1] < 0) & (D_t[:, 1:] >= 0)
    else:
        crossings_mask = (D_t[:, :-1] * D_t[:, 1:]) < 0
    num_valid = xp.sum(crossings_mask, axis=1)
    max_num_crossings = int(xp.max(num_valid))
    t_cross = t[:, :-1] - D_t[:, :-1] * (t[:, 1:] - t[:, :-1]) / (D_t[:, 1:] - D_t[:, :-1])
    alpha = -D_t[:, :-1] / (D_t[:, 1:] - D_t[:, :-1])
    x_cross = x[:, :-1] + alpha * (x[:, 1:] - x[:, :-1])
    y_cross = y[:, :-1] + alpha * (y[:, 1:] - y[:, :-1])
    z_cross = z[:, :-1] + alpha * (z[:, 1:] - z[:, :-1])
    r_cross = r[:, :-1] + alpha * (r[:, 1:] - r[:, :-1])
    r_mag = xp.sqrt(x_cross**2 + y_cross**2 + z_cross**2)
    r_unit_x, r_unit_y, r_unit_z = x_cross / r_mag, y_cross / r_mag, z_cross / r_mag
    cos_angle = n_obs[:, 0][:, None] * r_unit_x + n_obs[:, 1][:, None] * r_unit_y + n_obs[:, 2][:, None] * r_unit_z
    shapiro_delay = -2 * t_g[:, None] * xp.log(r_cross * (1 - cos_angle))
    geometric_delay = r_cross * cos_angle * t_g[:, None]
    crossings = xp.where(crossings_mask, t_cross + shapiro_delay + geometric_delay, xp.nan)
    sorted_indices = xp.argsort(xp.isnan(crossings), axis=1)
    all_crossings = xp.sort(xp.take_along_axis(crossings, sorted_indices, axis=1)[:, :max_num_crossings], axis=1)
    resid = xp.zeros_like(sma)
    for window in windows:
        crossings_in_window = xp.where((all_crossings >= window[0]) & (all_crossings <= window[1]) & xp.isfinite(all_crossings), all_crossings, xp.inf)
        idx = (timings >= window[0]) & (timings <= window[1])
        timings_in_window = timings[idx]
        errs_in_window = errs[idx]
        crossings_in_window = xp.pad(crossings_in_window, ((0, 0), (0, max(0, len(timings_in_window) - crossings_in_window.shape[1]))), constant_values=window[1])
        crossings_in_window = xp.sort(crossings_in_window, axis=1)[:, :len(timings_in_window)]
        crossings_in_window = xp.where(crossings_in_window == xp.inf, window[1], crossings_in_window)
        resid += xp.nansum((timings_in_window - crossings_in_window[:, :len(timings_in_window)])**2 / errs_in_window**2, axis=1)
    return resid
