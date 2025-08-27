# pn_trajectory.py
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

def beta(e, v, q, Y, nparams, dtype=cp.float32):
	beta_1 = 1 + (1/16 - 9/16 * Y**2) * q**2 * v**4 + (-1/4 + 9/4 * Y**2) * q**2 * v**6 \
		+ e**2 * ((-1/16 + 9/16 * Y**2) * q**2 * v**4 + (-9/4 * Y**2 + 1/4) * q**2 * v**6)

	beta_3 = (1 - Y**2) / 16 * q**2 * v**4 - (1-Y**2) / 4 * q**2 * v**6 + e**2 * (-(1 - Y**2) / 16 * q**2 * v**4 + (1 - Y**2) / 4 * q**2 * v**6)
	return cp.zeros(nparams, dtype=dtype), beta_1, cp.zeros(nparams, dtype=dtype), beta_3

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

def t_theta_p(e, v, q, Y, nparams, dtype=cp.float32):
	t2_theta_p = (Y**2-1)/4 * q**2*v**3 - (Y**2-1)/2 * q**2*v**5 + Y*(Y**2-1)/4 * (3+e**2)*q**3*v**6
	return cp.zeros(nparams, dtype=dtype), t2_theta_p

@cp.fuse
def phi_r(e, v, q, Y):
	phi1_r = e*(-2*q*v**3 + 2*Y*q**2*v**4 - 10*q*v**5 + 18*Y*q**2*v**6)

	phi2_r = e**2 * (-1/4 *Y*q**2*v**4 + 1/2*q*v**5 - 3/4*Y*q**2*v**6)
	return phi1_r, phi2_r

def X_R(e, v, q, Y, nparams, dtype=cp.float32):
	X0_R = ((1+Y)/2 - (9*Y-1)*(Y**2-1)/32 * q**2*v**4 + (9*Y-1)*(Y**2-1)/8 * q**2*v**6) \
		+ e**2 * ((9*Y-1)*(Y**2-1)/32 * q**2*v**4 - (9*Y-1)*(Y**2-1)/8 * q**2*v**6)

	X2_R = ((1-Y)/2 + Y*(Y**2-1)/4 * q**2*v**4 - Y*(Y**2-1)*q**2*v**6) \
		+ e**2*(-Y*(Y**2-1)/4 * q**2*v**4 + Y*(Y**2-1)*q**2*v**6)

	X4_R = ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6)

	return X0_R, cp.zeros(nparams, dtype=dtype), X2_R, cp.zeros(nparams, dtype=dtype), X4_R

def X_J(e, v, q, Y, nparams, dtype=cp.float32):
	X2_J = ((Y-1)/2 - (5*Y+1)*(Y**2-1)/16 * q**2*v**4 + (5*Y+1)*(Y**2-1) / 4 * q**2*v**6) \
		+ e**2*((5*Y+1)*(Y**2-1) / 16 * q**2*v**4 - (5*Y+1)*(Y**2-1) / 4 * q**2*v**6)

	X4_J = (-(Y+1)*(Y-1)**2 / 32 * q**2*v**4 + (Y+1)*(Y-1)**2 / 8 * q**2*v**6) \
		+ e**2 * ((Y+1)*(Y-1)**2 / 32 * q**2*v**4 - (Y+1)*(Y-1)**2 / 8 * q**2*v**6)
	return cp.zeros(nparams, dtype=dtype), cp.zeros(nparams, dtype=dtype), X2_J, cp.zeros(nparams, dtype=dtype), X4_J

@cp.fuse
def Omegas(p, e, v, q, Y): # Mino frequencies
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

@cp.fuse
def cartesian(r, cos_theta, phi): # convert polar to Cartesian coordinates
    sin_theta = (1-cos_theta**2)**(1/2)
    x = r * sin_theta * cp.cos(phi)
    y = r * sin_theta * cp.sin(phi)
    z = r * cos_theta
    return x, y, z

def trajectory(windows, logMbh, sma, ecc, incl, spin, phi_r0, phi_theta0, phi_phi0, dt, dtype=cp.float32):
    # Input preparation with optimal padding for TensorCore tiles (works best w/ multiples of 16)
    n_params_original = len(sma)
    n_params = ((n_params_original + 15) // 16) * 16
    
    def pad_array(arr):
        arr = cp.asarray(arr, dtype=dtype)
        if len(arr) < n_params:
            return cp.pad(arr, (0, n_params - len(arr)), mode='edge')
        return arr
    
    e = pad_array(ecc)
    sma_padded = pad_array(sma)
    p = sma_padded * (1 - e**2)
    v = (1/p)**(1/2)
    Y = cp.cos(cp.radians(pad_array(incl)))
    q = pad_array(spin)
    phi_r0 = pad_array(phi_r0)
    phi_theta0 = pad_array(phi_theta0)
    phi_phi0 = pad_array(phi_phi0)
    logMbh_padded = pad_array(logMbh)
    
    # Compute Mino frequencies
    Omega_t, Omega_r, Omega_theta, Omega_phi = Omegas(p, e, v, q, Y)
    
    # Create ime grid
    t_g = cp.concatenate([cp.arange(start, stop, dt, dtype=dtype) for start, stop in windows])
    t_g = t_g / (4.926580927874239e-06 * 10**logMbh_padded[:, None])
    
    lambd = t_g / Omega_t[:, None]
    lambd_shifted_r = lambd + (phi_r0 / Omega_r)[:, None]
    lambd_shifted_theta = lambd + (phi_theta0 / Omega_theta)[:, None]
    
    # Pre-compute coefficient matrices and ensure they're memory-contiguous for optimal performance
    vpt_coeffs = cp.ascontiguousarray((p / v)[:, None] * cp.vstack([v_p_t(e, v, q, Y)]).T)
    alpha_coeffs = cp.ascontiguousarray(cp.vstack([alpha(e, v, q, Y)]).T)
    theta_t_coeffs = cp.ascontiguousarray(p[:, None] * cp.vstack([t_theta_p(e, v, q, Y, n_params, dtype)]).T)
    beta_coeffs = cp.ascontiguousarray(cp.vstack([beta(e, v, q, Y, n_params, dtype)]).T)
    phi_r_coeffs = cp.ascontiguousarray(cp.vstack(phi_r(e, v, q, Y)).T)
    XR_coeffs = cp.ascontiguousarray(cp.vstack(X_R(e, v, q, Y, n_params, dtype)).T)
    XJ_coeffs = cp.ascontiguousarray(cp.vstack(X_J(e, v, q, Y, n_params, dtype)).T)
    
    # Compute r harmonics (0-6)
    n_r = cp.arange(7, dtype=dtype).reshape(1, 7, 1)
    angles_r = n_r * (Omega_r[:, None, None] * lambd_shifted_r[:, None, :])
    sin_r_full = cp.sin(angles_r)
    cos_r_full = cp.cos(angles_r)
    
    # Compute theta harmonics (0-4)
    n_theta = cp.arange(5, dtype=dtype).reshape(1, 5, 1)
    angles_theta = n_theta * (Omega_theta[:, None, None] * lambd_shifted_theta[:, None, :])
    sin_theta_full = cp.sin(angles_theta)
    cos_theta_full = cp.cos(angles_theta)
    
    # Compute time component of r summation
    summation_r = cp.matmul(vpt_coeffs[:, None, :], sin_r_full[:, 1:7, :]).squeeze(1)
    
    # Compute time component of theta summation  
    summation_theta = cp.matmul(theta_t_coeffs[:, None, :], sin_theta_full[:, 1:3, :]).squeeze(1)
    
    t = Omega_t[:, None] * lambd + summation_r + summation_theta
    t = t - t[:, 0:1]  # Preserve dimensions
    
    # Compute radial coordinate
    r = p[:, None] * cp.matmul(alpha_coeffs[:, None, :], cos_r_full).squeeze(1)
    
    # Compute theta coordinate
    cos_theta = cp.sqrt(1 - Y**2)[:, None] * cp.matmul(beta_coeffs[:, None, :], sin_theta_full[:, :4, :]).squeeze(1)
    
    # Compute phi coordinate
    summation_r_phi = cp.matmul(phi_r_coeffs[:, None, :], sin_r_full[:, 1:3, :]).squeeze(1)
    real_part = p[:, None] * cp.matmul(XR_coeffs[:, None, :], cos_theta_full).squeeze(1)
    imag_part = p[:, None] * cp.matmul(XJ_coeffs[:, None, :], sin_theta_full).squeeze(1)
    
    summation_theta_phi = real_part + 1j * imag_part
    phi = Omega_phi[:, None] * lambd + summation_r_phi + cp.angle(summation_theta_phi) + phi_phi0[:, None]
    
    P_orb = 2*cp.pi / Omega_phi * Omega_t * (4.926580927874239e-06 * 10**logMbh_padded)
    
    # Remove padding and return original size
    t = t[:n_params_original]
    r = r[:n_params_original]
    cos_theta = cos_theta[:n_params_original]
    phi = phi[:n_params_original]
    lambd = lambd[:n_params_original]
    P_orb = P_orb[:n_params_original]
    
    return t, r, cartesian(r, cos_theta, phi), lambd, P_orb


def residuals(timings, windows, errs, sma, e, incl, phi_r0, phi_theta0, phi_phi0, a, logMbh, theta_obs, theta_d, P_d, phi_d, dt, dtype=cp.float32):
    # Compute trajectory for input parameters
    t, r, (x, y, z), _, P_orb = trajectory(
        windows, logMbh, sma, e, incl, a, phi_r0, phi_theta0, phi_phi0, dt, dtype
    )
    
    # Ensure all arrays are contiguous and correct dtype
    P_d = cp.ascontiguousarray(cp.array(P_d, dtype=dtype) * P_orb)
    theta_d = cp.ascontiguousarray(cp.radians(cp.array(theta_d, dtype=dtype)))
    phi_d = cp.ascontiguousarray(cp.array(phi_d, dtype=dtype))
    t_g = cp.ascontiguousarray(4.926580927874239e-06 * 10**cp.array(logMbh, dtype=dtype))
    theta_obs = cp.ascontiguousarray(cp.array(theta_obs, dtype=dtype))
    
    # Convert time to seconds (physical)
    t = t * t_g[:, None]
    
    # Compute observer direction vector (constant for each trajectory)
    n_obs = cp.column_stack((
        cp.sin(theta_obs), 
        cp.zeros_like(theta_obs, dtype=dtype), 
        cp.cos(theta_obs)
    ))
    
    # Compute disk normal vector components (time-varying)
    phase = 2 * cp.pi * t / P_d[:, None] + phi_d[:, None]
    sin_theta_d = cp.sin(theta_d[:, None])
    cos_theta_d = cp.cos(theta_d[:, None])
    
    n_crs_x = sin_theta_d * cp.cos(phase)
    n_crs_y = sin_theta_d * cp.sin(phase)
    n_crs_z = cos_theta_d * cp.ones_like(t, dtype=dtype)
    
    # Compute disk crossings
    D_t = n_crs_x * x + n_crs_y * y + n_crs_z * z
    
    # Find where sign changes occur (disk crossings)
    crossings_mask = (D_t[:, :-1] * D_t[:, 1:]) < 0
    
    # Compute interpolation factor alpha for all potential crossings
    denominator = D_t[:, 1:] - D_t[:, :-1]
    safe_denom = cp.where(cp.abs(denominator) > 1e-10, denominator, 1e-10)
    alpha = -D_t[:, :-1] / safe_denom
    
    # Interpolate all quantities at crossing times
    t_cross = t[:, :-1] + alpha * (t[:, 1:] - t[:, :-1])
    x_cross = x[:, :-1] + alpha * (x[:, 1:] - x[:, :-1])
    y_cross = y[:, :-1] + alpha * (y[:, 1:] - y[:, :-1])
    z_cross = z[:, :-1] + alpha * (z[:, 1:] - z[:, :-1])
    r_cross = r[:, :-1] + alpha * (r[:, 1:] - r[:, :-1])
    
    # Compute relativistic/geometric time delays
    r_mag = cp.sqrt(x_cross**2 + y_cross**2 + z_cross**2)
    
    # Unit vectors
    r_unit_x = x_cross / r_mag
    r_unit_y = y_cross / r_mag
    r_unit_z = z_cross / r_mag
    
    # Dot product with observer direction
    cos_angle = (n_obs[:, 0][:, None] * r_unit_x + 
                n_obs[:, 1][:, None] * r_unit_y + 
                n_obs[:, 2][:, None] * r_unit_z)
    
    # Shapiro delay: -2GM/c³ * ln(r(1 - n·r̂))
    shapiro_arg = cp.maximum(r_cross * (1 - cos_angle), 1e-10)
    shapiro_delay = -2 * t_g[:, None] * cp.log(shapiro_arg)
    
    # Geometric delay: r * (n·r̂) * GM/c³
    geometric_delay = r_cross * cos_angle * t_g[:, None]
    
    total_delay = shapiro_delay + geometric_delay
    
    # Apply delays only where crossings occur
    crossings = cp.where(crossings_mask, t_cross + total_delay, cp.nan)
    
    # Count valid crossings per trajectory
    num_valid = cp.sum(crossings_mask, axis=1)
    max_num_crossings = int(cp.max(num_valid)) if cp.any(crossings_mask) else 0
    
    if max_num_crossings == 0:
        return cp.full_like(sma, cp.inf, dtype=dtype)
    
    # Sort to move valid crossings to the front
    sorted_indices = cp.argsort(cp.isnan(crossings), axis=1)
    all_crossings = cp.take_along_axis(crossings, sorted_indices, axis=1)[:, :max_num_crossings]
    
    # Pre-allocate residual array
    resid = cp.zeros(len(sma), dtype=dtype)
    
    # Convert timings and errors to GPU arrays if not already
    timings = cp.asarray(timings, dtype=dtype)
    errs = cp.asarray(errs, dtype=dtype)
    
    # Process windows
    for window in windows:
        window_start, window_end = dtype(window[0]), dtype(window[1])
        
        # Find crossings in this window
        valid_mask = (all_crossings >= window_start) & (all_crossings <= window_end) & cp.isfinite(all_crossings)
        
        # Get timings for this window
        timing_mask = (timings >= window_start) & (timings <= window_end)
        if not cp.any(timing_mask):
            continue
            
        timings_in_window = timings[timing_mask]
        errs_in_window = errs[timing_mask]
        n_timings = len(timings_in_window)
        
        # Extract relevant crossings for this window
        crossings_in_window = cp.where(valid_mask, all_crossings, cp.inf)
        
        # Sort to get the first n_timings valid crossings
        if crossings_in_window.shape[1] > 0:
            sort_idx = cp.argsort(crossings_in_window, axis=1)
            crossings_sorted = cp.take_along_axis(crossings_in_window, sort_idx, axis=1)
            
            # Take only the number of crossings we need
            if crossings_sorted.shape[1] >= n_timings:
                crossings_sorted = crossings_sorted[:, :n_timings]
            else:
                # Pad if we don't have enough crossings
                padding = cp.full((crossings_sorted.shape[0], n_timings - crossings_sorted.shape[1]), 
                                 window_end, dtype=dtype)
                crossings_sorted = cp.hstack([crossings_sorted, padding])
            
            # Replace inf values with window_end
            crossings_sorted = cp.where(cp.isfinite(crossings_sorted), crossings_sorted, window_end)
            
            # Compute residuals for this window
            diff = timings_in_window[None, :] - crossings_sorted
            chi2 = diff**2 / errs_in_window[None, :]**2
            resid += cp.nansum(chi2, axis=1)
    
    # Handle any infinite or NaN residuals
    finite_mask = cp.isfinite(resid)
    if cp.any(~finite_mask):
        max_finite = cp.max(resid[finite_mask]) if cp.any(finite_mask) else 1e10
        resid = cp.where(finite_mask, resid, max_finite * 10)
    
    return resid
