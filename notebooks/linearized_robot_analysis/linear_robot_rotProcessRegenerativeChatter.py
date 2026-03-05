import numpy as np
import matplotlib.pyplot as plt

def Phi_mat(omega, M, C, K):
    D = -omega**2 * M + 1j*omega*C + K
    return np.linalg.inv(D)

def alpha_from_angles(phi_st, phi_ex, Kr):
    # same closed-form you used, but this returns alpha (no tooth count scaling)
    def F(phi):
        return np.array([
            [0.5*(np.cos(2*phi) - 2*Kr*phi + Kr*np.sin(2*phi)),
             0.5*(-np.sin(2*phi) - 2*phi + Kr*np.cos(2*phi))],
            [0.5*(-np.sin(2*phi) + 2*phi + Kr*np.cos(2*phi)),
             0.5*(-np.cos(2*phi) - 2*Kr*phi - Kr*np.sin(2*phi))]
        ])
    return F(phi_ex) - F(phi_st)

def rotation_theta_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def alim_altintas_zero_order(M, C, K, Ktc, Krc,
                            phi_st, phi_ex,
                            rpm, N_teeth,
                            freqs_hz, theta=0.0,
                            real_tol=2e-3):
    """
    Altintas-style zero-order milling, returns a_lim at rpm, orientation theta.

    Uses the loop: B(ω) = 0.5 * Kt * alpha(φ) * (1 - e^{-iωT}) * Φ(iω)
    borderline when det(I - a B)=0  => a candidates = 1/λ(B)
    """
    # units: Ktc/Krc in Pa if M,C,K in SI
    Kr =  Krc / Ktc #Krc 
    alpha = alpha_from_angles(phi_st, phi_ex, Kr)

    # tooth period (delay)
    T = 60.0 / (N_teeth * rpm)

    # rotate directional term if needed
    R = rotation_theta_z(theta)
    alpha_th = R @ alpha @ R.T

    alim_best = np.inf

    for f in freqs_hz:
        w = 2*np.pi*f
        Phi = Phi_mat(w, M, C, K)

        Delta = 1.0 - np.exp(-1j*w*T)
        if abs(Delta) < 1e-12:
            continue

        # loop matrix (2x2)
        B = 0.5 * Ktc * (alpha_th @ (Delta * Phi))
        lam = np.linalg.eigvals(B)

        a_cand = 1.0 / lam

        # filter reasonable candidates
        re = np.real(a_cand)
        im = np.imag(a_cand)
        good = np.isfinite(a_cand) & (re > 0) & (np.abs(im) <= real_tol*np.maximum(1.0, np.abs(re)))

        if np.any(good):
            alim_best = min(alim_best, float(np.min(re[good])))

    return alim_best if np.isfinite(alim_best) else np.nan


def alim_altintas_smoother(M, C, K, Ktc, Krc,
                          phi_st, phi_ex, rpm, N_teeth,
                          freqs_hz, theta=0.0,
                          real_tol=5e-2,
                          spike_quantile=0.02):
    Kr = Krc / Ktc
    alpha = alpha_from_angles(phi_st, phi_ex, Kr)

    T = 60.0 / (N_teeth * rpm)

    R = rotation_theta_z(theta)
    alpha_th = R @ alpha @ R.T

    a_best_per_w = []

    for f in freqs_hz:
        w = 2*np.pi*f
        Phi = Phi_mat(w, M, C, K)
        Delta = 1.0 - np.exp(-1j*w*T)
        if abs(Delta) < 1e-12:
            continue

        B = 0.5 * Ktc * (alpha_th @ (Delta * Phi))
        lam = np.linalg.eigvals(B)
        a = 1.0 / lam

        re = np.real(a)
        im = np.imag(a)
        good = np.isfinite(a) & (re > 0) & (np.abs(im) <= real_tol*np.maximum(1.0, np.abs(re)))
        if np.any(good):
            a_best_per_w.append(np.min(re[good]))  # best at this ω

    if len(a_best_per_w) == 0:
        return np.nan

    a_best_per_w = np.array(a_best_per_w)

    # robust "minimum": use a low percentile instead of absolute min to kill spikes
    return float(np.quantile(a_best_per_w, spike_quantile))


def compute_entry_exit_angles_downmilling(
    
    r_engagement : float,
    v : float,
    D : float
    
):
    
    # Current tool entry / exit angles: --> https://www.sciencedirect.com/science/article/pii/S0888327020307044#b0160
    # Downmilling setup
    R = D/2
    b =  r_engagement  # radial depth of cut 
    v =  0.0  # We assume full entry of the tool thus distance from the edge is zero. (No partial entry)
    arg = np.clip((R - b) / R , -1.0, 1.0) # np.clip(1.0 - 2.0*b/D, -1.0, 1.0)
    arg1 = np.clip(v / (D/2) ,0.0, 1.0)
    phi_st = np.pi / 2 + np.arcsin(arg)
    phi_ex = np.pi / 2 + np.arccos(arg1)
        
    return phi_st , phi_ex

def load_macro_from_npz(npz_path, axes=(0, 1)):
    data = np.load(npz_path, allow_pickle=False)

    Lambda = data["Lambda_xyz"]  # (3,3)
    Dx     = data["Dx_xyz"]      # (3,3)
    Kx     = data["Kx_xyz"]      # (3,3)

    a0, a1 = axes

    M2 = Lambda[np.ix_([a0, a1], [a0, a1])]
    C2 = Dx[np.ix_([a0, a1], [a0, a1])]
    K2 = Kx[np.ix_([a0, a1], [a0, a1])]

    extras = {k: data[k] for k in data.files if k not in ["Lambda_xyz", "Dx_xyz", "Kx_xyz"]}
    return M2, C2, K2, extras

M2, C2, K2, extras = load_macro_from_npz("./data/linearized_operational_space_xyz.npz", axes=(0, 1))  # (x,y)
print("M2:\n", M2)
print("C2:\n", C2)
print("K2:\n", K2)
Ktc = 1930.4
Krc = 1159.6

# c_feed = 0.3
b_radial = 5.0
D = 20.0
N_teeth = 8

feed = 5.0 # mm/s 
rpm_fixed = 8000.0

phi_st, phi_ex = compute_entry_exit_angles_downmilling(r_engagement=b_radial, v=0.0, D=D)

# convert cutting coeffs to SI if M,C,K are SI
Ktc_SI = Ktc  * 1e6
Krc_SI = Krc * 1e6

# 2) choose frequency sweep based on your system modes (same approach you used)
w2 = np.linalg.eigvals(np.linalg.inv(M2) @ K2)
fn = np.sqrt(np.real(w2)) / (2*np.pi)
fmin = 0.2 * float(np.min(fn))
fmax = 5.0 * float(np.max(fn))
freqs = np.linspace(fmin, fmax, 6000)

# 3) evaluate a_lim(theta) for a set of orientations
theta_deg = np.arange(0, 361, 5)         # 0..360 every 5 deg (minimal)
thetas = np.deg2rad(theta_deg)

alim_m = np.array([
    alim_altintas_smoother(M2, C2, K2, Ktc_SI, Krc_SI,
                             phi_st, phi_ex,
                             rpm=rpm_fixed, N_teeth=N_teeth,
                             freqs_hz=freqs, theta=th,
                             real_tol=2e-2)
    for th in thetas
])

print("a_lim [mm] for each theta:")
for d, a in zip(theta_deg, alim_m):
    print(f"{d:3d} deg : {1000*a:.6g} mm")

# 4) plot vs theta (Cartesian)
plt.figure()
plt.plot(theta_deg, 1000*alim_m, marker="o", linewidth=1)
plt.xlabel("theta [deg]")
plt.ylabel("a_lim [mm]")
plt.title(f"a_lim vs orientation at {rpm_fixed:.0f} rpm")
plt.grid(True)
plt.show()

# 5) optional: polar plot (like your earlier map, but only the boundary)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="polar")
ax.plot(thetas, 1000*alim_m, marker="o", linewidth=1)
ax.set_title(f"a_lim boundary (mm) at {rpm_fixed:.0f} rpm")
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
plt.show()