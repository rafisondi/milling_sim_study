import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def eigenvalues_for_a(M2, C2, K2, theta, a_vals,
                      c_feed, b_radial, D, N_teeth,
                      phi_st, phi_ex, Ktc, Krc, n_hat,
                      scale_Kp=1000.0):
    """
    Returns:
      eigvals_all: shape (len(a_vals), 4) complex eigenvalues for each a
      max_real:    shape (len(a_vals),) max real part for each a
    """
    eigvals_all = np.zeros((len(a_vals), 4), dtype=complex)
    max_real = np.zeros(len(a_vals), dtype=float)

    for i, aax in enumerate(a_vals):
        Kp0 = zero_order_dF(
            c=c_feed, a_axial=aax, b_radial=b_radial, D=D, N_teeth=N_teeth,
            phi_st=phi_st, phi_ex=phi_ex, Ktc=Ktc, Krc=Krc, n_hat=n_hat
        )
        Kp0 = scale_Kp * Kp0

        # Build rotated Kp and state matrix exactly like your code
        R = rotation_theta_z(theta)
        Kp = R @ Kp0 @ R.T
        K_eff = K2 - Kp
        A = state_matrix(M2, C2, K_eff)

        eigvals = np.linalg.eigvals(A)

        # Sort for nicer tracking (by imaginary part then real part)
        idx = np.lexsort((np.real(eigvals), np.imag(eigvals)))
        eigvals = eigvals[idx]

        eigvals_all[i, :] = eigvals
        max_real[i] = np.max(np.real(eigvals))

    return eigvals_all, max_real



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
                    n_hat : np.ndarray, # normal vector of the edge surface (unit vector)
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
        n_hat : normal vector of the edge surface (unit vector)
        Ktc, Krc : specific force coefficients
    Returns:
       Jacobian [dF_dx, dF_dy] in space coordinate system.
    """
    nx, ny = n_hat
    tx, ty = -ny, nx  # tangent vector (90 deg CCW rotation of normal)
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

    dF_dx = dF_db * (-nx) + dF_dv * (-tx)
    dF_dy = dF_db * (-ny) + dF_dv * (-ty)


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

if __name__ == "__main__":
    
    M2, C2, K2, extras = load_macro_from_npz("./data/linearized_operational_space_xyz.npz", axes=(0, 1))  # (x,y)
    C2 = C2  
    
    Ktc = 1930.4
    Krc = 1159.6

    # c_feed = 0.3
    b_radial =9.0
    D = 20.0
    N_teeth = 8
    
    feed = 10.0 # mm/s 
    rpm = 400.0
    
    fz = feed /( N_teeth * (rpm/60.0))
    print(fz)
    c_feed = fz
    phi_st, phi_ex = compute_entry_exit_angles_downmilling(r_engagement=b_radial, v=0.0, D=D)

    # Sweep setup
    theta_deg = np.arange(0, 361, 1)
    thetas = np.deg2rad(theta_deg)

    a_max = 100         # mm (adjust as needed)
    delta_a = 1     # mm step size for axial depth
    n_a = int(a_max / delta_a) + 1  # number of axial depth samples
    # n_a = 61             # 0..a_max inclusive
    a_vals = np.linspace(0.0, a_max, n_a)

    # Build polar scatter points
    TH, AAX = np.meshgrid(thetas, a_vals, indexing="xy")  # shape (n_a, n_theta)
    TH_flat = TH.ravel()
    A_flat = AAX.ravel()

    stable_mask = np.empty_like(A_flat, dtype=bool)

    # Evaluate stability on the grid
    for i, (th, aax) in enumerate(zip(TH_flat, A_flat)):
        Kp0 = zero_order_dF(
            c=c_feed, a_axial=aax, b_radial=b_radial, D=D, N_teeth=N_teeth,
            phi_st=phi_st, phi_ex=phi_ex, Ktc=Ktc, Krc=Krc, n_hat=np.array([0.0, 1.0])
        )
        Kp0 = 1000.0 * Kp0
        stable_mask[i] = (max_real_eig(M2, C2, K2, Kp0, th) < 0.0)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    # Use different markers (not explicit colors) to denote stable vs unstable
    ax.scatter(TH_flat[stable_mask], A_flat[stable_mask], s=10, marker="o", alpha=0.6, label="Stable")
    ax.scatter(TH_flat[~stable_mask], A_flat[~stable_mask], s=14, marker="x", alpha=0.9, label="Unstable")

    ax.set_title("Stability map vs orientation (angle) and axial depth (radius)")
    ax.set_rlabel_position(22.5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)  # CCW
    ax.set_rlim(0, a_max)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.show()
    out_path = "./data/viz/stability_polar_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    out_path
    
    
    

    # ---------- Choose one orientation to inspect ----------
    theta_deg = 0.0   # pick the angle where you saw the weird stable/unstable/stable
    theta = np.deg2rad(theta_deg)

    eigvals_all, max_real = eigenvalues_for_a(
        M2, C2, K2, theta, a_vals,
        c_feed=c_feed, b_radial=b_radial, D=D, N_teeth=N_teeth,
        phi_st=phi_st, phi_ex=phi_ex, Ktc=Ktc, Krc=Krc, n_hat=np.array([0.0, 1.0]),
        scale_Kp=1000.0
    )
    
    
    # ---------- Plot 1: max real part vs axial depth ----------
    plt.figure()
    plt.plot(a_vals, max_real)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Axial depth a [mm]")
    plt.ylabel("max Re(eig(A))")
    plt.title(f"Stability metric vs axial depth (theta = {theta_deg:.1f} deg)")
    plt.grid(True)
    plt.show()

    # ---------- Plot 2: eigenvalue trajectories in complex plane ----------
    plt.figure()
    for k in range(eigvals_all.shape[1]):
        plt.plot(np.real(eigvals_all[:, k]), np.imag(eigvals_all[:, k]), marker=".", markersize=3, linestyle="-")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.title(f"Eigenvalue trajectories as a increases (theta = {theta_deg:.1f} deg)")
    plt.grid(True)
    plt.show()

    # ---------- Plot 3 (optional): real parts of each eigenvalue vs a ----------
    plt.figure()
    for k in range(eigvals_all.shape[1]):
        plt.plot(a_vals, np.real(eigvals_all[:, k]), marker=".", markersize=3, linestyle="-")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Axial depth a [mm]")
    plt.ylabel("Re(lambda)")
    plt.title(f"Real parts of eigenvalues vs a (theta = {theta_deg:.1f} deg)")
    plt.grid(True)
    plt.show()
    
    # # standard model steel
    # Ktc = 1930.4
    # Krc = 1159.6
    # Kac = 200.6

    # # Macro parameters from Pinocchio operational-space projection in xy plane
    # M2, C2, K2, extras = load_macro_from_npz("./data/linearized_operational_space_xyz.npz", axes=(0, 1))  # (x,y)
    # print(f"Inertia matrix: {M2}")
    # print(f"Damping matrix: {C2}")
    # print(f"Stiffnesss matrix: {K2}")
    
    # theta = np.radians(120)  
    # R_IT = rotation_theta_z(theta)  
    # print("Rotation matrix R_IT:\n", R_IT)
    
    # phi_st, phi_ex = compute_entry_exit_angles_downmilling(
    #     r_engagement=5.0,
    #     v=0.0,
    #     D=20.0
    # )

    # # process stiffness in the "base" frame you’re treating as local
    # Kp0 = zero_order_dF(
    #     c=0.15,
    #     a_axial=2.0,
    #     b_radial=5.0,
    #     D=20.0,
    #     N_teeth=4,
    #     phi_st=phi_st,
    #     phi_ex=phi_ex,
    #     Ktc=Ktc,
    #     Krc=Krc
    # )

    # thetas, max_reals, stable = sweep_orientations(M2, C2, K2, Kp0, n=361)

    # # Report unstable ranges / worst case
    # worst_i = np.argmax(max_reals)
    # worst_theta_deg = np.degrees(thetas[worst_i])
    # print(f"Worst orientation: {worst_theta_deg:.1f} deg, max Re(lambda) = {max_reals[worst_i]:.6g}")

    # # Simple listing of unstable angles (if any)
    # unstable_deg = np.degrees(thetas[~stable])
    # if unstable_deg.size == 0:
    #     print("All sampled orientations stable (max Re(lambda) < 0).")
    # else:
    #     print(f"Unstable at {unstable_deg.size} / {thetas.size} samples.")
    #     print("First few unstable angles (deg):", unstable_deg[:20])