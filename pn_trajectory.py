import cupy as cp, numpy as np

@cp.fuse
def alpha(e, v, q, Y):
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
	return cp.zeros(nparams), beta_1, cp.zeros(nparams), beta_3

@cp.fuse
def v_p_t(e, v, q, Y):
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
	t2_theta_p = (Y**2-1)/4 * q**2*v**3 - (Y**2-1)/2 * q**2+v**5 + Y*(Y**2-1)/4 * (3+e**2)*q**3*v**6
	return cp.zeros(nparams), t2_theta_p

@cp.fuse
def phi_r(e, v, q, Y):
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

	return X0_R, cp.zeros(nparams), X2_R, cp.zeros(nparams), X4_R

def X_J(e, v, q, Y, nparams):
	X2_J = ((Y-1)/2 - (5*Y+1)*(Y**2-1)/16 * q**2*v**4 + (5*Y+1)*(Y**2-1) / 4 * q**2*v**6) \
		+ e**2*((5*Y+1)*(Y**2-1) / 16 * q**2*v**4 - (5*Y+1)*(Y**2-1) / 4 * q**2*v**6)

	X4_J = (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6)
	return cp.zeros(nparams), cp.zeros(nparams), X2_J, cp.zeros(nparams), X4_J

@cp.fuse
def Omegas(p, e, v, q, Y): # Mino frequencies
	Omega_t = p**2*(1 + 3/2*e**2 + 15/8*e**4 + 35/16*e**6 + (3/2 - 1/4*e**2 - 15/16*e**4 - 45/32*e**6)*v**2 \
		+ (2*Y*q*e**2 + 3*Y*q*e**4 + 15/4*Y*q*e**6)*v**3 \
		+ (27/8 - 1/2*Y**2*q**2 + 1/2*q**2 + (-99/16 + q**2 - 2*Y**2*q**2)*e**2 \
		+ (-567/64 + 21/16*q**2 - 45/16*Y**2*q**2)*e**4 + (-1371/128 + 25/16*q**2 - 55/16*Y**2*q**2)*e**6)*v**4 \
		+ (-3*Y*q + 43/2*Y*q*e**2 + 231/8*Y*q*e**4 + 555/16*Y*q*e**6)*v**5 \
		+ (135/16 - 1/4*q**2 + 3/4*Y**2*q**2 + (-1233/32 + 47/4*q**2 - 75/2*Y**2*q**2)*e**2 \
		+ (-6567/128 + 499/32*q**2 - 1577/32*Y**2*q**2)*e**2 \
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

@cp.fuse
def cartesian(r, cos_theta, phi):
    sin_theta = (1-cos_theta**2)**(1/2)
    x = r * sin_theta * cp.cos(phi)
    y = r * sin_theta * cp.sin(phi)
    z = r * cos_theta
    return x, y, z

def trajectory(windows, logMbh, sma, ecc, incl, spin, phi_r0, phi_theta0, phi_phi0, dt):
	# change to variables consistent with arXiv:1505.01600v3
	e = cp.array(ecc)
	p = cp.array(sma) * (1-e**2) # semilatus rectum
	v = (1/p)**(1/2)
	Y = cp.cos(cp.radians(cp.array(incl)))
	q = cp.array(spin)
	phi_r0 = cp.array(phi_r0)
	phi_theta0 = cp.array(phi_theta0)
	phi_phi0 = cp.array(phi_phi0)
	nparams = int(len(e))
	Omega_t, Omega_r, Omega_theta, Omega_phi = Omegas(p, e, v, q, Y)
	t_g = cp.concatenate([cp.arange(start, stop, dt) for start, stop in windows]) / (4.926580927874239e-06 * 10**cp.array(logMbh)[:, None])
	lambd = t_g / Omega_t[:, None]
	lambd_shifted_r = lambd + (phi_r0 / Omega_r)[:, None]
	lambd_shifted_theta = lambd + (phi_theta0 / Omega_theta)[:, None]

	n_r = cp.arange(1, 7, dtype=cp.float32).reshape(1, 6, 1)
	sin_r_terms = cp.sin(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])  # Shape (k, 6, N*n)
	summation_r = cp.einsum('ik,ikj->ij', (p / v)[:, None] * cp.vstack([v_p_t(e, v, q, Y)]).T, sin_r_terms)  # Shape (k, N*n)
	n_theta = cp.arange(1, 3).reshape(1, 2, 1)  # Shape (2, 1)
	sin_theta_terms = cp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])  # Shape (k, 2, N*n)
	summation_theta = cp.einsum('ik,ikj->ij', p[:, None] * cp.vstack([t_theta_p(e, v, q, Y, nparams)]).T, sin_theta_terms)  # Shape (k, N*n)
	t = Omega_t[:, None] * lambd + summation_r + summation_theta  # Shape (k, N*n)
	t = t - t[:, 0][:, None]

	n_r = cp.arange(7, dtype=cp.float32).reshape(1, 7, 1) 
	cos_terms = cp.cos(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
	summation = cp.einsum('ij,ijk->ik', cp.vstack([alpha(e, v, q, Y)]).T, cos_terms)
	r = p[:, None] * summation

	n_theta = cp.arange(4).reshape(1, 4, 1)  # Shape (1, 4, 1)
	sin_terms = cp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])  # Shape (k, 4, N*n)
	summation = cp.einsum('ij,ijk->ik', cp.vstack([beta(e, v, q, Y, nparams)]).T, sin_terms)  # Shape (k, N*n)
	cos_theta = cp.sqrt(1 - Y**2)[:, None] * summation  # Shape (k, N*n)
	
	n_r = cp.arange(1, 3).reshape(1, 2, 1)
	n_theta = cp.arange(5).reshape(1, 5, 1)
	sin_r_terms2 = cp.sin(n_r * Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
	phi_stack = cp.vstack(phi_r(e, v, q, Y)).T # shape (n_traj, 2)
	summation_r3 = cp.einsum('ij,ijk->ik', phi_stack, sin_r_terms2)
	sin_theta_terms2 = cp.sin(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
	cos_theta_terms2 = cp.cos(n_theta * Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
	XR_stack = cp.vstack(X_R(e, v, q, Y, nparams)).T # shape (n_traj, 5)
	XJ_stack = cp.vstack(X_J(e, v, q, Y, nparams)).T # shape (n_traj, 5)
	summation_theta3 = p[:, None] * (cp.einsum('ij,ijk->ik', XR_stack, cos_theta_terms2)+cp.einsum('ij,ijk->ik', 1j * XJ_stack, sin_theta_terms2))
	phi = Omega_phi[:, None] * lambd + summation_r3 + cp.angle(summation_theta3) + phi_phi0[:, None]
	P_orb = 2*cp.pi / Omega_phi * Omega_t * (4.926580927874239e-06 * 10**cp.array(logMbh))

	return t, r, cartesian(r, cos_theta, phi), lambd, P_orb

# def residuals(timings, windows, errs, sma, e, incl, a, logMbh, theta_obs, phi_r0, phi_theta0, phi_phi0, theta_d, phi_d, P_d):
def residuals(timings, windows, errs, sma, e, incl, a, logMbh, theta_obs, theta_d, P_d, phi_d, t0, dt):
	t, r, (x, y, z), _, P_orb = trajectory(windows, logMbh, sma, e, incl, a, np.zeros_like(a), np.zeros_like(a), np.zeros_like(a), dt)
	P_d = cp.array(P_d) * P_orb
	theta_d = cp.radians(cp.array(theta_d))
	phi_d = cp.array(phi_d)
	t_g = 4.926580927874239e-06 * 10**cp.array(logMbh)
	t = t * t_g[:, None] # convert t from gravitational time to seconds
	theta_obs = cp.array(theta_obs)
	n_obs = cp.column_stack((cp.sin(theta_obs), cp.zeros_like(theta_obs), cp.cos(theta_obs)))
	n_crs_x = cp.sin(theta_d[:, None]) * cp.cos(2 * cp.pi * t / P_d[:, None] + phi_d[:, None])
	n_crs_y = cp.sin(theta_d[:, None]) * cp.sin(2 * cp.pi * t / P_d[:, None] + phi_d[:, None])
	n_crs_z = cp.cos(theta_d[:, None])
	dot_product = n_obs[:, 0][:, None] * n_crs_x + n_obs[:, 1][:, None] * n_crs_y + n_obs[:, 2][:, None] * n_crs_z
	shapiro_delay = -2 * np.log(r * (1 + dot_product)) * t_g[:, None]
	geometric_delay = -r * dot_product * t_g[:, None]
	D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z
	crossings_mask = (D_t[:, :-1] * D_t[:, 1:]) < 0
	num_valid = cp.sum(crossings_mask, axis=1)
	max_num_crossings = int(cp.max(num_valid))
	crossings = cp.where(crossings_mask, t[:, :-1] + shapiro_delay[:, :-1] + geometric_delay[:, :-1], cp.nan)
	sorted_indices = cp.argsort(cp.isnan(crossings), axis=1)
	all_crossings = cp.take_along_axis(crossings, sorted_indices, axis=1)[:, :max_num_crossings]
	all_crossings = all_crossings - cp.array(t0[:, None])
	resid = cp.zeros_like(sma)
	for window in windows:
		crossings_in_window = cp.where((all_crossings >= window[0]) & (all_crossings <= window[1]) & (cp.isfinite(all_crossings)), all_crossings, cp.inf)
		idx = (timings >= window[0]) & (timings <= window[1])
		timings_in_window = timings[idx]
		crossings_in_window = cp.take_along_axis(crossings_in_window, cp.argsort(~cp.isfinite(crossings_in_window), axis=1)[:, :len(timings_in_window)], axis=1)
		errs_in_window = errs[idx]
		resid += cp.nansum((timings_in_window - crossings_in_window)**2 / errs_in_window**2, axis=1)
	resid[~cp.isfinite(resid)] = cp.nanmax(resid)
	return resid
