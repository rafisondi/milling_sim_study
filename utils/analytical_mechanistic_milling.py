import math as m
import numpy as np
def zero_order_dF(
                    c: float,  
                    a_axial: float, 
                    b_radial: float,
                    v: float,
                    D: float,
                    N_teeth: float, 
                    phi_st: float, 
                    phi_ex: float, 
                    Kt: float, 
                    Kr: float):
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
        Kt, Kr : specific force coefficients Kt = Ktc and Kr = Krc / Ktc
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
        K = np.array([Kt, Kr])
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
        
    return phi_st , phi_ex


def zero_order_force_with_displacement(
    Dx: np.ndarray,
    c: float,
    a: float,
    N: int,
    phi_st: float,
    phi_ex: float,
    Ktc: float,
    Krc: float
) -> np.ndarray:
    """
    Zero-order (averaged) milling force including linear displacement effect.

    Returns:
        F_avg = [Fx, Fy]

    Theory:
        F = A0 * c + Ay * dy + Ax * dx
    """

    dx, dy = Dx

    C = (N * a) / (8.0 * m.pi)

    # helper terms
    dphi   = phi_ex - phi_st
    dsin2p = m.sin(2*phi_ex) - m.sin(2*phi_st)
    dcos2p = m.cos(2*phi_ex) - m.cos(2*phi_st)

    # zero-order directional coefficients
    A0x = C * ( Ktc * dcos2p - Krc * (2*dphi - dsin2p) )
    A0y = C * ( Ktc * (2*dphi - dsin2p) + Krc * dcos2p )

    # first-order directional coefficients (linearized displacement)
    Axx = C * ( Ktc * (2*dphi - dsin2p) + Krc * dcos2p )
    Axy = C * ( Ktc * dcos2p - Krc * (2*dphi - dsin2p) )

    Ayx = -Axy
    Ayy =  Axx

    # single-line force expression
    return np.array([
        A0x * c + Axx * dx + Axy * dy,
        A0y * c + Ayx * dx + Ayy * dy
    ], dtype=float)
