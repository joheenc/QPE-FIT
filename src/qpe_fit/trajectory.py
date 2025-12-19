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
	return xp.zeros(nparams), beta_1, xp.zeros(nparams), beta_3

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
	return xp.zeros(nparams), t2_theta_p

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

	return X0_R, xp.zeros(nparams), X2_R, xp.zeros(nparams), X4_R

def X_J(e, v, q, Y, nparams):
	X2_J = ((Y-1)/2 - (5*Y+1)*(Y**2-1)/16 * q**2*v**4 + (5*Y+1)*(Y**2-1) / 4 * q**2*v**6) \
		+ e**2*((5*Y+1)*(Y**2-1) / 16 * q**2*v**4 - (5*Y+1)*(Y**2-1) / 4 * q**2*v**6)

	X4_J = (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6)
	return xp.zeros(nparams), xp.zeros(nparams), X2_J, xp.zeros(nparams), X4_J

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

def trajectory(windows, logMbh, sma, ecc, incl, spin, phi_r0, phi_theta0, phi_phi0, dt):
    e = xp.asarray(ecc)
    p = xp.asarray(sma) * (1-e**2)
    v = (1/p)**(1/2)
    Y = xp.cos(xp.radians(xp.asarray(incl)))
    q = xp.asarray(spin)
    phi_r0 = xp.asarray(phi_r0)
    phi_theta0 = xp.asarray(phi_theta0)
    phi_phi0 = xp.asarray(phi_phi0)
    nparams = len(e)
    Omega_t, Omega_r, Omega_theta, Omega_phi = Omegas(p, e, v, q, Y)
    t_g = xp.concatenate([xp.arange(start, stop, dt) for start, stop in windows]) / (4.926580927874239e-06 * 10**xp.asarray(logMbh)[:, None])
    lambd = t_g / Omega_t[:, None]
    lambd_shifted_r = lambd + (phi_r0 / Omega_r)[:, None]
    lambd_shifted_theta = lambd + (phi_theta0 / Omega_theta)[:, None]
    n_r = xp.arange(1, 7, dtype=xp.float32).reshape(1, 6, 1)
    sin_r_terms = xp.sin(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
    summation_r = xp.einsum('ik,ikj->ij', (p / v)[:, None] * xp.vstack(v_p_t(e, v, q, Y)).T, sin_r_terms)
    n_theta = xp.arange(1, 3).reshape(1, 2, 1)
    sin_theta_terms = xp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
    summation_theta = xp.einsum('ik,ikj->ij', p[:, None] * xp.vstack(t_theta_p(e, v, q, Y, nparams)).T, sin_theta_terms)
    t = Omega_t[:, None] * lambd + summation_r + summation_theta
    t = t - t[:, 0][:, None]
    n_r = xp.arange(7, dtype=xp.float32).reshape(1, 7, 1)
    cos_terms = xp.cos(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
    summation = xp.einsum('ij,ijk->ik', xp.vstack(alpha(e, v, q, Y)).T, cos_terms)
    r = p[:, None] * summation
    n_theta = xp.arange(4).reshape(1, 4, 1)
    sin_terms = xp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
    summation = xp.einsum('ij,ijk->ik', xp.vstack(beta(e, v, q, Y, nparams)).T, sin_terms)
    cos_theta = xp.sqrt(1 - Y**2)[:, None] * summation
    n_r = xp.arange(1, 3).reshape(1, 2, 1)
    n_theta = xp.arange(5).reshape(1, 5, 1)
    sin_r_terms2 = xp.sin(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
    summation_r3 = xp.einsum('ij,ijk->ik', xp.vstack(phi_r(e, v, q, Y)).T, sin_r_terms2)
    sin_theta_terms2 = xp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
    cos_theta_terms2 = xp.cos(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
    XR_stack = xp.vstack(X_R(e, v, q, Y, nparams)).T
    XJ_stack = xp.vstack(X_J(e, v, q, Y, nparams)).T
    summation_theta3 = p[:, None] * (xp.einsum('ij,ijk->ik', XR_stack, cos_theta_terms2) + xp.einsum('ij,ijk->ik', 1j * XJ_stack, sin_theta_terms2))
    phi = Omega_phi[:, None] * lambd + summation_r3 + xp.angle(summation_theta3) + phi_phi0[:, None]
    P_orb = 2*xp.pi / Omega_phi * Omega_t * (4.926580927874239e-06 * 10**xp.asarray(logMbh))
    return t, r, cartesian(r, cos_theta, phi), lambd, P_orb

def residuals(timings, windows, errs, sma, e, incl, phi_r0, phi_theta0, phi_phi0, a, logMbh, theta_obs, theta_d, P_d, phi_d, dt):
    t, r, (x, y, z), _, P_orb = trajectory(windows, logMbh, sma, e, incl, a, phi_r0, phi_theta0, phi_phi0, dt)
    P_d = xp.asarray(P_d) * P_orb
    theta_d = xp.radians(xp.asarray(theta_d))
    phi_d = xp.asarray(phi_d)
    t_g = 4.926580927874239e-06 * 10**xp.asarray(logMbh)
    t = t * t_g[:, None]
    theta_obs = xp.asarray(theta_obs)
    n_obs = xp.column_stack((xp.sin(theta_obs), xp.zeros_like(theta_obs), xp.cos(theta_obs)))
    n_crs_x = xp.sin(theta_d[:, None]) * xp.cos(2 * xp.pi * t / P_d[:, None] + phi_d[:, None])
    n_crs_y = xp.sin(theta_d[:, None]) * xp.sin(2 * xp.pi * t / P_d[:, None] + phi_d[:, None])
    n_crs_z = xp.cos(theta_d[:, None])
    D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z
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
        resid += xp.nansum((timings_in_window - crossings_in_window[:, :len(timings_in_window)])**2 / errs_in_window**2, axis=1)
    return resid
