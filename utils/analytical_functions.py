import numpy as np
from scipy import integrate

def dF_dv(u, v, R, b, fz, Kt, Kr):
    u = float(np.clip(u, -R + 1e-12, R - 1e-12))
    v = float(np.min((v, 1e-12)))  # only consider v <= 0.0
    N = 4
    K = np.array([Kt, Kr], dtype=float)  # (2,)
    R_matr = np.array([[ np.sqrt(1 - v**2 / R**2),  -v/R],
                          [ v/R, np.sqrt(1 - v**2 / R**2)]], dtype=float)
    
    DF_n = -b * fz * (R_matr @ K) * v /(R**2 - v**2) # returns shape (2,)
    return (N / (2*np.pi)) * DF_n

def dF_du(dist, v, R, b, fz, Kt, Kr):
    d = float(np.clip(dist, -R + 1e-12, R - 1e-12)) # Edge distance
    v = float(np.min((v, 1e-12)))  # only consider v <= 0.0
    K = np.array([Kt, Kr], dtype=float)  # (2,)
    
    


    u = float(np.clip(u, -R + 1e-12, R - 1e-12))
    v = float(np.max((np.abs(v), 0.0)))  # only consider v <= 0.0
    K = np.array([Kt, Kr], dtype=float)  # (2,)
    R_matr = np.array([[ -u / R, -np.sqrt(1 - u**2 / R**2)],
                          [ np.sqrt(1 - u**2 / R**2), -u / R]], dtype=float)

    DF_n = b * fz * (R_matr @ K) * 1 / np.sqrt( R**2 - u**2 )  # returns shape (2,)
    return DF_n

    # R_matr = np.array([[ -u / R, -np.sqrt(1 - u**2 / R**2)],
    #                       [ np.sqrt(1 - u**2 / R**2), -u / R]], dtype=float)
    
    
    # m = np.sqrt(R**2 - d**2)
    # R_matr = np.array([[ -d , -m], 
    #                       [ m , -d ]], dtype=float)
    # h = b*fz / R**2*(R_matr @ K)
    # # DF_n = b * fz * (R_matr @ K) * 1 / np.sqrt( R**2 - u**2 )  # returns shape (2,)


    # N = 4
    # return (N / (2*np.pi))* h #* DF_n

def F(edge_distances, R, b, fz, Kt, Kr):
    u = float(np.clip(edge_distances[0], -R + 1e-12, R - 1e-12)) 
    v = float(np.min((np.clip(edge_distances[1], -R + 1e-12, R - 1e-12), 0.0)))  # only consider v <= 0.0
    
    
    phi_st = np.arccos(-np.abs(u) / R)
    phi_ex = np.pi/2 + np.arccos(np.abs(v) / R)
    
    # print("phi_st:", phi_st, "phi_ex:", phi_ex)

    if phi_ex - phi_st < 1e-12:
        return np.zeros(2)

    # K = np.array([Kt, Kr], dtype=float)  # (2,)
    
    Kte = 0.0
    Kre = 0.0
    N = 1 
    
    def integrand(phi):
        if phi < phi_st or phi > phi_ex:
            return np.array([0.0, 0.0], dtype=float)
        
        h = fz * np.sin(phi)

        # Ft, Fr
        Ft = b * (Kt * h + Kte)
        Fr = b * (Kr * h + Kre)

        # Tx^t from Eq. 
        T = np.array([[-np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), -np.cos(phi)]], dtype=float)

        return T @ np.array([Ft, Fr], dtype=float)

    I, _ = integrate.quad_vec(integrand, 0.0, 2 * np.pi)

    # average over tooth-passing period: Number of teeth/(2 pi(full rotation)) 
    F0 =  I
    return F0
