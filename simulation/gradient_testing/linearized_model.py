import math as m
import numpy as np
import matplotlib.pyplot as plt
from eraser_of_matter import milling_workpiece

def dF2_dy2(
                c: float,
                b_radial: float,
                D: float,
                Ktc: float,
                Krc: float,
                a_axial: float = 1.0,
                db_dy: float = -1.0,
                eps: float = 1e-12,
            ):

    # ---- clip b to open interval ----
    b = float(np.clip(b_radial, eps, D - eps))

    # ---- phi_st(b) ----
    # a = 1 - 2b/D in [-1,1] when b in [0,D]
    a_arg = 1.0 - 2.0 * b / D
    a_arg = float(np.clip(a_arg, -1.0 + 1e-15, 1.0 - 1e-15))
    phi = float(np.pi - np.arccos(a_arg))

    # ---- phi derivatives wrt b (interior) ----
    g = b * (D - b)                       # g = b(D-b)
    phi_b = -1.0 / np.sqrt(g)             # dphi/db
    phi_bb = (D - 2.0 * b) / (2.0 * g**(3.0/2.0))  # d2phi/db2

    # ---- chain to y ----
    b_y = float(db_dy)
    phi_y = phi_b * b_y                   # dphi/dy
    phi_yy = phi_bb * (b_y**2)            # d2phi/dy2 (since d2b/dy2=0)

    # ---- define K, T(phi), dT/dphi ----
    K = np.array([Ktc, Krc], dtype=float)

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    T = np.array([[-cosphi, -sinphi],
                  [ sinphi, -cosphi]], dtype=float)

    Tp = np.array([[ sinphi, -cosphi],
                   [ cosphi,  sinphi]], dtype=float)  # dT/dphi

    # ---- h and derivatives wrt phi ----
    h = c * sinphi
    hp = c * cosphi
    hpp = -c * sinphi

    # ---- t = T@K and derivatives wrt phi ----
    t = T @ K
    tp = Tp @ K

    # ---- s(phi) = a_axial * h * t ----
    # s' = a * (hp*t + h*tp)
    s_p = a_axial * (hp * t + h * tp)

    # s'' = a * (hpp*t + 2*hp*tp + h*t'')
    # but for this T, t'' = -t, and hpp = -h, so:
    # s'' = a * ( (-h)*t + 2*hp*tp + h*(-t) ) = a * (2*(hp*tp - h*t))
    s_pp = a_axial * (2.0 * (hp * tp - h * t))

    # ---- chain rule: d2/dy2 s(phi(y)) = s''*(phi_y)^2 + s' * phi_yy ----
    d2s_dy2 = s_pp * (phi_y**2) + s_p * phi_yy
    return d2s_dy2 # [d2Fx_dy2, d2Fy_dy2] (2,) 

def zero_order_dF(
                    c: float,  
                    a_axial: float, 
                    b_radial: float,
                    v: float,
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
    eps = 1e-9  # in mm; pick something like 1e-6..1e-9 depending on your scale
    b_radial = np.clip(b_radial, eps, D - eps)
    v = np.clip(v, -D/2 + eps, 0.0 - eps)

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



def analytical_force(phi,
                     phi_st,
                     phi_ex,
                     b,
                     fz,
                     Kt,
                     Kr):
    
    """
    Analytical force calculation for milling.

    Inputs:
      phi: current angle
      phi_st: start angle
      phi_ex: exit angle
      b: width of cut
      fz: feed per tooth
      Kt: tangential cutting force coefficient
      Kr: radial cutting force coefficient

    Returns:
      np.array([Fx, Fy]) force vector in the space coordinate system.
    """
    
    if phi_ex - phi_st < 1e-12:
        return np.zeros(2)
    
    if phi < phi_st or phi > phi_ex:
        return np.array([0.0, 0.0], dtype=float)

    h = fz * np.sin(phi)
    Ft = b * (Kt * h)
    Fr = b * (Kr * h)

    # Tx^t from Eq. (6)
    T = np.array([[-np.cos(phi), -np.sin(phi)],
                    [np.sin(phi), -np.cos(phi)]], dtype=float)
    return T @ np.array([Ft, Fr], dtype=float)

def zero_order_force_analytical(
                                c: float,  
                                a: float, 
                                N: float, 
                                phi_st: float, 
                                phi_ex: float, 
                                Ktc: float, 
                                Krc: float):
    """
    Average forces [Fx0, Fy0] for downmilling. https://www.sciencedirect.com/science/article/pii/S0888327020307044#b0160
    Equation (9) (10)
    Inputs:
      c  : feed per tooth
      a  : axial depth of cut
      N  : number of flutes
      phi_st  : start angle
      phi_ex  : exit angle
      Ktc, Krc, Kte, Kre : specific force coefficients
    Returns:
      np.array([Fx0, Fy0]) in space coordinate system.
    """
    
    # # Downmilling start/exit angles (Eq. 8)
    # arg = np.clip(1.0 - 2.0*b/D, -1.0, 1.0) # Argument clipping to avoid NaNs for edge cases b=D/2
    # phi_st = np.pi - np.arccos(arg)
    # phi_ex = np.pi
    
    # Coefficients in Eq. (10) we neglect as we do not consider edge force components Kte and Kre
    # ax0 = (N*a)/(2*np.pi) * bracket(lambda p: -Kte*np.sin(p) + Kre*np.cos(p))
    # ay0 = -(N*a)/(2*np.pi) * bracket(lambda p: Kte*np.cos(p) + Kre*np.sin(p))

    ax1_func = lambda phi: ( Ktc*m.cos(2*phi) - Krc*(2*phi - m.sin(2*phi)))
    ay1_func = lambda phi: ( Ktc*(2*phi - m.sin(2*phi) )+ Krc*m.cos(2*phi)) 
    
    ax1 = (N*a)/ (8*np.pi) *  ( ax1_func(phi_ex) - ax1_func(phi_st) )
    ay1 = (N*a)/ (8*np.pi) *  ( ay1_func(phi_ex) - ay1_func(phi_st) )
    
    Fx0 = ax1*c # + ax0
    Fy0 = ay1*c # + ay0

    return np.array([Fx0, Fy0], dtype=float)

def compute_entry_exit_angles_downmilling(
    r_xy_current: np.ndarray,
    xy_workpiece: np.ndarray,
    radius_tool: float,
    diameter_end_mill: float,
):
    
    # Current tool entry / exit angles: --> https://www.sciencedirect.com/science/article/pii/S0888327020307044#b0160
    # Downmilling setup
    
    b = - r_xy_current[1] + radius_tool + xy_workpiece[1,-1]  # radial depth of cut 
    v = -(r_xy_current[0] - xy_workpiece[0,-1]) 
    arg = np.clip(1.0 - 2.0*b/diameter_end_mill, -1.0, 1.0)
    arg1 = np.clip(v / radius_tool ,0.0, 1.0)
    phi_st = np.pi - np.arccos(arg)
    phi_ex = np.pi / 2 + np.arccos(arg1)

    # pT = r_xy_current.copy()
    # pE = xy_workpiece[: , -1]
    
    # n1 = np.array([0,-1])
    # s1 = pE - pT - 
    
    # d1 = np.array( [pT[0] - pE[0] , 0])
    # h_vert = np.array([0, np.sqrt(milling.radius_tool**2 - d1[0]**2)])
    # s1 =  -pT- d1 - h_vert
    
    # n2 = np.array([0,1])
    # s2 = pE - pT
    # phi_st = np.arccos(np.dot(n2, s2) / (milling.radius_tool))
    
    # d = pE[0] - pT[0]
    # arg1 = np.clip( (pE[0] - pT[0]) / milling.radius_tool,0.0, 1.0)
    # phi_ex = np.arccos(arg1) + np.pi/2
        
        
    return phi_st , phi_ex

def simulate(        
            xy_workpiece: np.ndarray,
            xy_tool_center_0: np.ndarray,
            tool_diameter: float,
            N_tool_teeth: int,
            axial_cutting_depth: float,
            rev_per_min: float,
            feed_per_tooth: float,
            rot_angle_per_time_step: float,
            T_end: float,
            orientation_0: float = 0.0):
    
    # ---------------------------
    # Init milling geometry/model
    # ---------------------------
    milling = milling_workpiece(xy_workpiece)
    milling.diameter_end_mill = tool_diameter
    milling.number_of_teeth = N_tool_teeth
    milling.radius_tool = tool_diameter / 2.
    milling.slice_height = axial_cutting_depth
    
    Kt = milling.cutting_force_coefficient_Ktc
    Kr = milling.cutting_force_coefficient_Krc
    
    # ---------------------------
    # Simulation and Process parameters
    # ---------------------------
    omega = -rev_per_min * 2 * m.pi / 60  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * milling.number_of_teeth  # mm/s
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60
    
    dt = 1.0 / sample_freq
    t = np.arange(0.0, T_end, dt)
    N = len(t)
    
    rev_period = 1/(rev_per_min / 60.0)
    samples_per_rev = int( sample_freq * rev_period )
    
    # ---------------------------
    # Lists
    # ---------------------------
    forces_sim = np.zeros((3, N))                   # simulated milling forces (from milling model)
    forces_analytical_avg = np.zeros((2, N))        # analytical average F returns [Fx, Fy] 
    forces_zoe = np.zeros((2, N))                   # zero-order average F returns [Fx, Fy]
    forces_analytical = np.zeros((2, N))            # analytical F returns [Fx, Fy] (per your usage)
    dF_dx_analytical = np.zeros((2, N))            # analytical dF/dx 
    dF_dy_analytical = np.zeros((2, N))            # analytical dF/dy
    Hessian_vector_y = np.zeros((2, N))            # analytical d2F/dy2
    forces_sim_avg = np.zeros((2, N))
    
    for i in range(N):
        # Current tool center
        r_xy_current = xy_tool_center_0 + np.array([feed_speed * t[i], 0.0])
        
        b = - r_xy_current[1] + milling.radius_tool + xy_workpiece[1,-1]  # radial depth of cut 
        v = -(r_xy_current[0] - xy_workpiece[0,-1]) 
        
        phi_st, phi_ex = compute_entry_exit_angles_downmilling( r_xy_current=r_xy_current,
                                                                xy_workpiece=xy_workpiece,
                                                                radius_tool=milling.radius_tool,
                                                                diameter_end_mill=milling.diameter_end_mill)
            
        # ---------------------------
        # Milling simulation step
        # ---------------------------
        orientation = orientation_0 + t[i] * omega # current tool orientation [rad]
        milling.erase_step(r_xy_current, orientation, -1, t[i])
        
        if i >= 2:
            forces_sim[:, i] = milling.total_milling_force
            
            force_analytic = analytical_force(
                (rot_angle_per_time_step * i + np.pi/2 )%(2*m.pi),
                phi_st,
                phi_ex,
                milling.axial_cutting_depth,
                feed_per_tooth,
                Kt,
                Kr
            )
            forces_analytical[:, i] = force_analytic
        
        dF = zero_order_dF(
            feed_per_tooth,
            axial_cutting_depth,
            b,
            v,
            milling.diameter_end_mill,
            N_tool_teeth,
            phi_st,
            phi_ex,
            Kt,
            Kr
        )
        dF_dx_analytical[:, i] = dF[:,0]
        dF_dy_analytical[:, i] = dF[:,1]
        
        d2F_dy2 = dF2_dy2(
            feed_per_tooth,
            b,
            milling.diameter_end_mill,
            Kt,
            Kr,
            axial_cutting_depth,
            db_dy = -1.0
        )
        Hessian_vector_y[:, i] = d2F_dy2
        
        # print(f"dF {dF}] N/mm")
        
        if i % samples_per_rev == 0 and i >= samples_per_rev:
            res = zero_order_force_analytical(
                feed_per_tooth,
                axial_cutting_depth,
                N_tool_teeth,
                phi_st,
                phi_ex,
                Kt,
                Kr
            )
            forces_zoe[:, i - samples_per_rev:i] = res[:, np.newaxis]
            
            average = forces_sim[0:2, i - samples_per_rev:i].mean(axis=1)
            forces_sim_avg[:, i - samples_per_rev:i] = average[:, np.newaxis]


    return {
        "milling": milling,
        "t": t,
        "forces_sim": forces_sim,
        "forces_analytical": forces_analytical,
        "forces_analytical_avg": forces_analytical_avg,
        "dF_dx_analytical": dF_dx_analytical,
        "dF_dy_analytical": dF_dy_analytical,
        "forces_zoe": forces_zoe,
        "Hessian_vector_y": Hessian_vector_y,
        "dt": dt,
        "sample_freq": sample_freq,
        "feed_speed": feed_speed,
        "forces_sim_average": forces_sim_avg,
    }


if __name__ == "__main__":
    # ---------------------------
    # Workpiece geometry
    
    # 4 ------ 3
    # |        |
    # |        |
    # 1 ------ 2
    
    # ---------------------------
    
    wp_p1 = np.array([0,0])
    wp_p2 = np.array([1000,0])
    wp_p3 = np.array([1000,76])
    wp_p4 = np.array([0,76])
    
    xy_workpiece = np.array([
        wp_p1,
        wp_p2,
        wp_p3,
        wp_p4
    ]).T
    
    # ---------------------------
    # Tooling parameters
    # ---------------------------
    
    tool_diameter = 20.0  # [mm]
    tool_radius = tool_diameter / 2.0
    radial_engagement = 8.0# [mm]
    axial_cutting_depth = 1.0  # [mm]
    N_tool_teeth = 2
    
    
    offset_tool_y = tool_radius - radial_engagement
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    else:
        offset_tool_x = -tool_radius
    
    xy_tool_center_0 = np.array([wp_p4[0] + offset_tool_x,
                                 wp_p4[1] + offset_tool_y])  # Start above and left of workpiece corner

    #   ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000
    feed_per_tooth = 0.15 # 0.15 [mm/tooth]
    delta_deg = 5 # [deg]
    rot_angle_per_time_step = delta_deg * (2*m.pi / 360)
    T_end = 1.0 # [s]
    
    # ---------------------------
    # Run simulation
    # ---------------------------
    
    out = simulate(
        xy_workpiece,
        xy_tool_center_0,
        tool_diameter,
        N_tool_teeth,
        axial_cutting_depth,
        rev_per_min,
        feed_per_tooth,
        rot_angle_per_time_step,
        T_end
    )
        
    t0 = out["t"]
    dt = out["dt"]
    Fx0 = out["forces_sim"][0,:]
    Fy0 = out["forces_sim"][1,:]
    Fx0_analytical = out["forces_analytical"][0,:]
    Fy0_analytical = out["forces_analytical"][1,:]

    Fx0_zoe = out["forces_zoe"][0,:]
    Fy0_zoe = out["forces_zoe"][1,:]
    Fx0_avg = out["forces_sim_average"][0,:]
    Fy0_avg = out["forces_sim_average"][1,:]
    
    dF_dx_analytical = out["dF_dx_analytical"]
    dF_dy_analytical = out["dF_dy_analytical"]
    
    hessian_vector_y = out["Hessian_vector_y"]
    
    # ----------------------------
    # Taylor linear approximation 
    # ---------------------------
    dx = 0.0  
    dy = 0.01 #.0 #  positive y moves tool away from workpiece in this setup

    Fx_lin = Fx0_zoe + dF_dx_analytical[0, :] * dx + dF_dy_analytical[0, :] * dy
    Fy_lin = Fy0_zoe + dF_dx_analytical[1, :] * dx + dF_dy_analytical[1, :] * dy
    F_lin = np.vstack((Fx_lin, Fy_lin))
    
    out1 = simulate( # Base truth for Taylor approximation comparison
        xy_workpiece,
        xy_tool_center_0 + np.array([dx, dy]),
        tool_diameter,
        N_tool_teeth,
        axial_cutting_depth,
        rev_per_min,
        feed_per_tooth,
        rot_angle_per_time_step,
        T_end
    )
    
    t1 = out1["t"]
    Fx1 = out1["forces_sim"][0,:]
    Fy1 = out1["forces_sim"][1,:]
    Fx1_zoe = out1["forces_zoe"][0,:]
    Fy1_zoe = out1["forces_zoe"][1,:]
    Fx1_avg = out1["forces_sim_average"][0,:]
    Fy1_avg = out1["forces_sim_average"][1,:]
    
    # ---------------------------
    # Plot results
    # ---------------------------
    
    out["milling"].show_plot()
    
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    # Fx
    axs[0].plot(t0, Fx0, alpha = 0.5, label=f"Sim 0 Fx ")
    axs[0].scatter(t0, Fx0, s=1, color='blue', marker='.')
    axs[0].plot(t0, Fx0_analytical, alpha = 0.5, label=f"Analytical Fx ")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Milling Forces from Simulation")
    axs[0].legend()
    # Fy
    axs[1].plot(t0, Fy0, alpha = 0.5, label=f"Sim 0 Fy ")
    axs[1].scatter(t0, Fy0, s=1, color='blue', marker='.')
    axs[1].plot(t0, Fy0_analytical, alpha = 0.5, label=f"Analytical Fy ")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    
    
    fig1 , axs1 = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    # Fx
    axs1[0].plot(t0, Fx0, alpha = 0.5, label=f"Sim 0 Fx ")
    axs1[0].scatter(t0, Fx0_zoe, s=0.2, label=f"Zero-order Fx ")
    axs1[0].scatter(t0, Fx0_avg, s=0.1, label=f"Average over Rev Fx ")
    axs1[0].set_ylabel("Force [N]")
    axs1[0].set_title("Averaged Milling Forces (Zero-order) Fx")
    axs1[0].legend()
    # Fy
    axs1[1].plot(t0, Fy0, alpha = 0.5, label=f"Sim 0 Fy ")
    axs1[1].scatter(t0, Fy0_zoe, s=0.2, label=f"Zero-order Fy ")
    axs1[1].scatter(t0, Fy0_avg, s=0.1, label=f"Average over Rev Fy ")
    axs1[1].set_xlabel("Time [s]")
    axs1[1].set_ylabel("Force [N]")
    axs1[1].legend()


    fig2 , axs2 = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    # dF/dx
    axs2[0].plot(t0, dF_dx_analytical[0,:], alpha = 0.5, label=f"dFx_dx Fx ")
    axs2[0].plot(t0, dF_dx_analytical[1,:], alpha = 0.5, label=f"dFy_dx Fy ")
    axs2[0].set_ylabel("dFx/dx [N/mm]")
    axs2[0].set_title("Analytical Force Gradients dF/dx")
    axs2[0].legend()
    # dF/dy
    axs2[1].plot(t0, dF_dy_analytical[1,:], alpha = 0.5, label=f"dFy_dy Fy ")
    axs2[1].plot(t0, dF_dy_analytical[0,:], alpha = 0.5, label=f"dFx_dy Fx ")
    axs2[1].set_xlabel("Time [s]")
    axs2[1].set_ylabel("dFy/dy [N/mm]")
    axs2[1].legend()
    

    fig3 , axs3 = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    # Fx
    axs3[0].plot(t0, Fx0_avg, alpha = 0.5, label=f"AVG 0 Fx ")
    axs3[0].plot(t0, Fx_lin, alpha = 0.5, label=f"Linearized Fx ")
    axs3[0].scatter(t1, Fx1_avg, s=1, color='red', label=f" Sim 1 Fx Avg")
    axs3[0].set_ylabel("Force [N]")
    axs3[0].set_title("Linearized Milling Forces Fx")
    axs3[0].legend()

    # Fy
    axs3[1].plot(t0, Fy0_avg, alpha = 0.5, label=f"AVG  0 Fy ")
    axs3[1].plot(t0, Fy_lin, alpha = 0.5 , label=f"Linearized Fy ")
    axs3[1].scatter(t1, Fy1_avg, s=1, color='red', label=f" Sim 1 Fy Avg")
    axs3[1].set_xlabel("Time [s]")
    axs3[1].set_ylabel("Force [N]")
    axs3[1].legend()
    
    # plt.figure( figsize=(10, 4))
    # plt.plot(t0, 0.5* np.linalg.norm(hessian_vector_y * dy*dy , axis=0) , label="||d2F_dy2||")
    # plt.xlabel("Time [s]")  
    # plt.ylabel("N")
    # plt.title("Error bound from 2nd order term in Taylor expansion")
    # plt.legend()
    
    
    plt.tight_layout()
    
    eps = 1e-12

    # Your dt must be defined. If not, derive it from t0:
    # dt = t0[1] - t0[0]

    sample_freq = 1.0 / dt
    rev_period = 1.0 / (rev_per_min / 60.0)
    samples_per_rev = int(sample_freq * rev_period)
    shift = samples_per_rev  # compare Fx0[i] to Fx1[i+shift]

    if shift <= 0 or shift >= len(t0):
        raise ValueError(f"Invalid shift={shift} for signal length {len(t0)}. Check dt / RPM.")

    # Use averaged forces for the check (aligned)
    Fx1_avg_shifted = Fx1_avg[shift:]
    Fy1_avg_shifted = Fy1_avg[shift:]

    Fx0_avg_trunc = Fx0_avg[:-shift]
    Fy0_avg_trunc = Fy0_avg[:-shift]

    t_grad = t0[:-shift]

    # Truncate analytical gradients to the same length for apples-to-apples
    dFx_dx_a_trunc = dF_dx_analytical[0, :-shift]
    dFy_dx_a_trunc = dF_dx_analytical[1, :-shift]
    dFx_dy_a_trunc = dF_dy_analytical[0, :-shift]
    dFy_dy_a_trunc = dF_dy_analytical[1, :-shift]

    # Numerical finite differences (forward difference, but aligned by 1 rev)
    dFx_dy_n = None
    dFy_dy_n = None
    dFx_dx_n = None
    dFy_dx_n = None

    if abs(dy) > eps:
        dFx_dy_n = (Fx1_avg_shifted - Fx0_avg_trunc) / dy
        dFy_dy_n = (Fy1_avg_shifted - Fy0_avg_trunc) / dy

    if abs(dx) > eps:
        dFx_dx_n = (Fx1_avg_shifted - Fx0_avg_trunc) / dx
        dFy_dx_n = (Fy1_avg_shifted - Fy0_avg_trunc) / dx

    # ---------------------------
    # Plot comparisons (numerical vs analytical)
    # ---------------------------

    fig_fd, ax_fd = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # dF/dy
    ax_fd[0].plot(t_grad, dFx_dy_a_trunc, alpha=0.8, label="Analytical dFx/dy")
    if dFx_dy_n is not None:
        ax_fd[0].plot(t_grad, dFx_dy_n, "--", alpha=0.8, label="Numerical dFx/dy (FD, 1-rev shift)")
    ax_fd[0].set_ylabel("dFx/dy [N/mm]")
    ax_fd[0].set_title("Gradient check using shifted averaged force (1 revolution)")
    ax_fd[0].legend()

    ax_fd[1].plot(t_grad, dFy_dy_a_trunc, alpha=0.8, label="Analytical dFy/dy")
    if dFy_dy_n is not None:
        ax_fd[1].plot(t_grad, dFy_dy_n, "--", alpha=0.8, label="Numerical dFy/dy (FD, 1-rev shift)")
    ax_fd[1].set_xlabel("Time [s]")
    ax_fd[1].set_ylabel("dFy/dy [N/mm]")
    ax_fd[1].legend()




    
    if dFx_dy_n is not None:
        err = dFx_dy_n - dFx_dy_a_trunc
        print(f"RMS error dFx/dy: {np.sqrt(np.mean(err**2)):.6g} N/mm, max abs: {np.max(np.abs(err)):.6g} N/mm")

    if dFy_dy_n is not None:
        err = dFy_dy_n - dFy_dy_a_trunc
        print(f"RMS error dFy/dy: {np.sqrt(np.mean(err**2)):.6g} N/mm, max abs: {np.max(np.abs(err)):.6g} N/mm")

    plt.tight_layout()
    plt.show()