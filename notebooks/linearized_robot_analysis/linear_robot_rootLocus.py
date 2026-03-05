import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

@dataclass
class MillingParams:
    # Tool / engagement
    D_mm: float = 20.0
    N_teeth: int = 4
    b_radial_mm: float = 5.0
    a_axial_mm: float = 2.0

    # Cutting coefficients (as you have them)
    Ktc_N_per_mm2: float = 1930.4
    Krc_N_per_mm2: float = 1159.6

    # Spindle + feed definition
    n_rpm: float = 8000.0          # spindle speed (rpm)  <-- tune
    fz_mm_per_tooth: float = 0.15  # feed per tooth (mm/tooth) <-- tune

    # Optional: if you prefer defining feed rate directly:
    Vf_mm_per_min: float | None = None  # set to a number to override fz

def derived_feed(params: MillingParams):
    """
    Returns:
      omega (rad/s),
      tooth_pass_Hz,
      Vf_mm_per_min,
      fz_mm_per_tooth
    """
    n = params.n_rpm
    N = params.N_teeth

    omega = 2*np.pi * n/60.0
    tooth_pass_Hz = N * n/60.0

    if params.Vf_mm_per_min is None:
        fz = params.fz_mm_per_tooth
        Vf = fz * N * n  # mm/min
    else:
        Vf = params.Vf_mm_per_min
        fz = Vf / (N * n)

    return omega, tooth_pass_Hz, Vf, fz

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


def zero_order_dF(
                    c: float,  
                    a_axial: float, 
                    b_radial: float,
                    D: float,
                    N_teeth: float, 
                    phi_st: float, 
                    phi_ex: float, 
                    Ktc: float, 
                    Krc: float):
    """
    Average force gradients
    Inputs:
        c  : feed per tooth
        a  : axial depth of cut
        b  : width of cut / radial depth of cut
        v  : feed direction distance from tool center to cut start --> range(-R, 0)
        D  : tool diameter
        N  : number of flutes
        omega : angular velocity
        phi_st  : start angle
        phi_ex  : exit angle
        Ktc, Krc : specific force coefficients
    Returns:
       Jacobian [dF_dx, dF_dy] in space coordinate system.
    """

    v = 0.0  # We assume full entry of the tool thus distance from the edge is zero. (No partial entry)
    # dphi/dv and dphi/db
    dphi_st_db = -1.0 / np.sqrt(b_radial*(D-b_radial))
    dphi_ex_dv = -2.0 / np.sqrt(D**2 - 4*v**2)

    def q(phi):
        h = c*np.sin(phi)
        T = np.array([[-np.cos(phi), -np.sin(phi)],
                      [ np.sin(phi), -np.cos(phi)]])
        K = np.array([Ktc, Krc])
        return a_axial * (T @ K) * h 

    dF_dv = N_teeth/(2*np.pi) * q(phi_ex) * dphi_ex_dv
    dF_db = N_teeth/(2*np.pi) * (-q(phi_st) * dphi_st_db)
    
    dv_dx, dv_dy = -1.0, 0.0
    db_dx, db_dy =  0.0, -1.0

    dF_dx = dF_dv*dv_dx + dF_db*db_dx
    dF_dy = dF_dv*dv_dy + dF_db*db_dy

    return np.column_stack((dF_dx, dF_dy))


def rotation_theta_z(theta):
    """
    Rotation matrix for rotation about the z-axis by angle theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])
    
    
def state_matrix(M, C, K_eff):
    # Build A for z=[x; xdot], zdot = A z
    zeros = np.zeros((2, 2))
    I = np.eye(2)
    Minv = np.linalg.inv(M)
    A = np.block([
        [zeros,             I],
        [-Minv @ K_eff, -Minv @ C]
    ])
    return A

def stability_metric(M, C, K, Kp0, theta):
    R = rotation_theta_z(theta)
    Kp = R @ Kp0 @ R.T
    K_eff = K - Kp
    A = state_matrix(M, C, K_eff)
    eigvals = np.linalg.eigvals(A)
    max_real = np.max(np.real(eigvals))
    return max_real, eigvals, Kp, K_eff

def sweep_orientations(M, C, K, Kp0, n=361):
    thetas = np.linspace(0.0, 2*np.pi, n, endpoint=True)
    max_reals = np.empty_like(thetas)

    for i, th in enumerate(thetas):
        max_real, _, _, _ = stability_metric(M, C, K, Kp0, th)
        max_reals[i] = max_real

    stable = max_reals < 0.0
    return thetas, max_reals, stable

def max_real_eig(M, C, K, Kp0, theta):
    R = rotation_theta_z(theta)
    Kp = R @ Kp0 @ R.T
    K_eff = K - Kp
    A = state_matrix(M, C, K_eff)
    eigvals = np.linalg.eigvals(A)
    return np.max(np.real(eigvals))


def eigvals_for_ap_ae(M2, C2, K2,
                      ap_mm, ae_mm,
                      c_feed_mm_per_tooth,
                      D_mm, N_teeth,
                      Ktc, Krc,
                      theta=0.0):
    """
    Returns eigenvalues of A for given axial ap and radial ae.
    Kp is computed in N/mm, then converted to N/m to match K2 (assumed SI).
    """
    phi_st, phi_ex = compute_entry_exit_angles_downmilling(r_engagement=ae_mm, v=0.0, D=D_mm)

    Kp0 = zero_order_dF(
        c=c_feed_mm_per_tooth,
        a_axial=ap_mm,
        b_radial=ae_mm,
        D=D_mm,
        N_teeth=N_teeth,
        phi_st=phi_st,
        phi_ex=phi_ex,
        Ktc=Ktc,
        Krc=Krc
    )

    # Unit fix: N/mm -> N/m
    Kp0 = 1000.0 * Kp0

    # rotate workpiece if desired (optional)
    R = rotation_theta_z(theta)
    Kp = R @ Kp0 @ R.T

    K_eff = K2 - Kp
    A = state_matrix(M2, C2, K_eff)
    return np.linalg.eigvals(A)

def max_real_part(eigs):
    return float(np.max(np.real(eigs)))

# ------------------------------------------------------------
# 1) 2D MAP over (ae, ap): max Re(lambda)
# ------------------------------------------------------------
def stability_map(M2, C2, K2,
                  ap_vals_mm, ae_vals_mm,
                  c_feed, D_mm, N_teeth, Ktc, Krc,
                  theta=0.0):

    grid = np.zeros((len(ap_vals_mm), len(ae_vals_mm)))
    for i, ap in enumerate(ap_vals_mm):
        for j, ae in enumerate(ae_vals_mm):
            eigs = eigvals_for_ap_ae(M2, C2, K2, ap, ae, c_feed, D_mm, N_teeth, Ktc, Krc, theta=theta)
            grid[i, j] = max_real_part(eigs)
    return grid  # shape (len(ap), len(ae))

# ------------------------------------------------------------
# 2) Eigenvalue motion vs ONE parameter (locus)
# ------------------------------------------------------------
def eigen_locus_vs_ap(M2, C2, K2,
                      ap_vals_mm, ae_fixed_mm,
                      c_feed, D_mm, N_teeth, Ktc, Krc,
                      theta=0.0):
    # returns array shape (len(ap), 4)
    E = np.zeros((len(ap_vals_mm), 4), dtype=complex)
    for i, ap in enumerate(ap_vals_mm):
        eigs = eigvals_for_ap_ae(M2, C2, K2, ap, ae_fixed_mm, c_feed, D_mm, N_teeth, Ktc, Krc, theta=theta)
        # simple ordering for nicer tracks: sort by imag then real
        idx = np.lexsort((np.real(eigs), np.imag(eigs)))
        E[i, :] = eigs[idx]
    return E

def eigen_locus_vs_ae(M2, C2, K2,
                      ae_vals_mm, ap_fixed_mm,
                      c_feed, D_mm, N_teeth, Ktc, Krc,
                      theta=0.0):
    E = np.zeros((len(ae_vals_mm), 4), dtype=complex)
    for i, ae in enumerate(ae_vals_mm):
        eigs = eigvals_for_ap_ae(M2, C2, K2, ap_fixed_mm, ae, c_feed, D_mm, N_teeth, Ktc, Krc, theta=theta)
        idx = np.lexsort((np.real(eigs), np.imag(eigs)))
        E[i, :] = eigs[idx]
    return E

# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------
def plot_stability_map(ap_vals, ae_vals, grid):
    # grid is (len(ap), len(ae))
    AE, AP = np.meshgrid(ae_vals, ap_vals)

    plt.figure(figsize=(8, 5))
    # Do not set explicit colors; let matplotlib choose defaults
    cs = plt.contourf(AE, AP, grid, levels=30)
    plt.colorbar(cs, label="max Re(eig(A))  [1/s]")
    plt.contour(AE, AP, grid, levels=[0.0], linewidths=2)  # stability boundary
    plt.xlabel("Radial engagement ae [mm]")
    plt.ylabel("Axial depth ap [mm]")
    plt.title("Stability map using max Re(eig(A)) (stable if < 0)")
    plt.tight_layout()

def plot_eigen_locus(E, param_vals, param_name):
    """
    E: (N, 4) complex eigenvalues ordered per step.
    """
    plt.figure(figsize=(7, 6))
    for k in range(E.shape[1]):
        plt.plot(np.real(E[:, k]), np.imag(E[:, k]), marker="o", markersize=3, linewidth=1)

    # vertical stability boundary Re=0
    plt.axvline(0.0, linewidth=2, linestyle="--")

    plt.xlabel("Re(λ) [1/s]")
    plt.ylabel("Im(λ) [rad/s]")
    plt.title(f"Eigenvalue locus vs {param_name}")
    plt.grid(True)
    plt.tight_layout()

def plot_max_real_vs_param(E, param_vals, param_name):
    max_re = np.max(np.real(E), axis=1)
    plt.figure(figsize=(7, 4))
    plt.plot(param_vals, max_re, marker="o", linewidth=1)
    plt.axhline(0.0, linewidth=2, linestyle="--")
    plt.xlabel(param_name)
    plt.ylabel("max Re(λ) [1/s]")
    plt.title(f"Stability metric vs {param_name}")
    plt.grid(True)
    plt.tight_layout()

# ------------------------------------------------------------
# EXAMPLE USAGE (adapt to your setup)
# ------------------------------------------------------------
if __name__ == "__main__":

    # Load your macro model
    M2, C2, K2, extras = load_macro_from_npz("./data/linearized_operational_space_xyz.npz", axes=(0, 1))

    # Cutting coefficients
    Ktc = 1930.4
    Krc = 1159.6

    # Tool / process
    D_mm = 20.0
    N_teeth = 4

    # Use feed per tooth here (Altintas chip thickness uses fz)
    # If you want to compute from feed rate and rpm:
    feed_rate_mm_per_s = 5.0
    rpm = 800.0
    fz = feed_rate_mm_per_s / (N_teeth * (rpm / 60.0))  # mm/tooth
    c_feed = fz

    # Sweep ranges
    ap_vals = np.linspace(0.0, 40.0, 81)  # axial depth [mm]
    ae_vals = np.linspace(0.5, 12.0, 48)  # radial engagement [mm] (avoid 0)

    # 1) Stability map
    grid = stability_map(M2, C2, K2, ap_vals, ae_vals, c_feed, D_mm, N_teeth, Ktc, Krc, theta=0.0)
    plot_stability_map(ap_vals, ae_vals, grid)

    # 2) Eigenvalue locus vs ap (fix ae)
    ae_fixed = 8.0
    E_ap = eigen_locus_vs_ap(M2, C2, K2, ap_vals, ae_fixed, c_feed, D_mm, N_teeth, Ktc, Krc, theta=0.0)
    plot_eigen_locus(E_ap, ap_vals, param_name=f"ap [mm] (ae fixed = {ae_fixed} mm)")
    plot_max_real_vs_param(E_ap, ap_vals, param_name="ap [mm]")

    # 3) Eigenvalue locus vs ae (fix ap)
    ap_fixed = 20.0
    E_ae = eigen_locus_vs_ae(M2, C2, K2, ae_vals, ap_fixed, c_feed, D_mm, N_teeth, Ktc, Krc, theta=0.0)
    plot_eigen_locus(E_ae, ae_vals, param_name=f"ae [mm] (ap fixed = {ap_fixed} mm)")
    plot_max_real_vs_param(E_ae, ae_vals, param_name="ae [mm]")

    plt.show()
