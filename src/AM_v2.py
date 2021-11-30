import numpy as np
from src.misc import D


def whichPart(part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0
    return factor_el, factor_mag


def s_canonical(r, theta, phi, fE, fH, part="both"):
    """
        Electric and magnetic field functions are expected to be
            fE = fE(r, theta, phi)
            fH = fH(r, theta, phi)
        In the spherical coord basis (e_r, e_theta, e_phi)
    """
    factor_el, factor_mag = whichPart(part)

    E, H = fE(r, theta, phi), fH(r, theta, phi)
    ExE = np.cross(np.conj(E), E, axis=0)
    HxH = np.cross(np.conj(H), H, axis=0)

    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2

    return (factor_el*ExE + factor_mag*HxH)/w


def w_canonical(x, y, z, fE, fH, part="both"):
    factor_el, factor_mag = whichPart(part)

    E, H = fE(x, y, z), fH(x, y, z)
    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2
    return w

def j2_canonical(x, y, z, fE, fH, part="both", dh=1e-5):
    """
        Electric and magnetic field functions are expected to be
            fE = fE(x, y, z)
            fH = fH(x, y, z)
        In cartesian coord basis (e_x, e_y, e_z)
    """
    factor_el, factor_mag = whichPart(part)

    E, H = fE(x, y, z), fH(x, y, z)
    dx_E, dx_H = D(fE, x, y, z, 'x'), D(fH, x, y, z, 'x')
    dy_E, dy_H = D(fE, x, y, z, 'y'), D(fH, x, y, z, 'y')
    dz_E, dz_H = D(fE, x, y, z, 'z'), D(fH, x, y, z, 'z')

    ddxx_E, ddxx_H = D(fE, x, y, z, 'xx'), D(fH, x, y, z, 'xx')
    ddyy_E, ddyy_H = D(fE, x, y, z, 'yy'), D(fH, x, y, z, 'yy')
    ddzz_E, ddzz_H = D(fE, x, y, z, 'zz'), D(fH, x, y, z, 'zz')

    ddxy_E, ddxy_H = D(fE, x, y, z, 'xy'), D(fH, x, y, z, 'xy')
    ddxz_E, ddxz_H = D(fE, x, y, z, 'xz'), D(fH, x, y, z, 'xz')
    ddzy_E, ddzy_H = D(fE, x, y, z, 'zy'), D(fH, x, y, z, 'zy')

    # J^2 = L^2 + S^2 + 2(LS)

    # E S^2 E
    ELECTRIC_PART = 2 * np.dot(np.conjugate(E), E)
    # E L^2 E
    ELECTRIC_PART += np.dot(
        np.conjugate(E),
        2 * (x * dx_E + y * dy_E + z * dz_E) +
        2*x*y*ddxy_E + 2*y*z*ddzy_E + 2*z*x*ddxz_E - (y*y+z*z)*ddxx_E - (x*x+z*z)*ddyy_E - (x*x+y*y)*ddzz_E
    )
    # 2 E (LS) E
    ELECTRIC_PART += 2 * np.dot(
        np.array([x, y, z]),
        np.conjugate(E[0])*dx_E + np.conjugate(E[1])*dy_E + np.conjugate(E[2])*dz_E
    )

    # H S^2 H
    MAGNETIC_PART = 2 * np.dot(np.conjugate(H), H)
    # H L^2 H
    MAGNETIC_PART += np.dot(
        np.conjugate(H),
        2 * (x * dx_H + y * dy_H + z * dz_H) +
        2*x*y*ddxy_H + 2*y*z*ddzy_H + 2*z*x*ddxz_H - (y*y+z*z)*ddxx_H - (x*x+z*z)*ddyy_H - (x*x+y*y)*ddzz_H
    )
    # 2 H (LS) H
    MAGNETIC_PART += 2 * np.dot(
        np.array([x, y, z]),
        np.conjugate(H[0])*dx_H + np.conjugate(H[1])*dy_H + np.conjugate(H[2])*dz_H
    )

    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2
    # w = 1

    return (factor_el * ELECTRIC_PART + factor_mag * MAGNETIC_PART) / w


def j2_canonical_short(x, y, z, fE, fH, part="both", dh=1e-5):
    """
        Electric and magnetic field functions are expected to be
            fE = fE(x, y, z)
            fH = fH(x, y, z)
        In cartesian coord basis (e_x, e_y, e_z)

        DOES NOT WORK!!!
    """
    factor_el, factor_mag = whichPart(part)

    r2 = x**2 + y**2 + z**2

    E, H = fE(x, y, z), fH(x, y, z)

    def r_dot_E(x, y, z):
        E = fE(x, y, z)
        return x*E[0] + y*E[1] + z*E[2]

    def r_dot_H(x, y, z):
        E = fH(x, y, z)
        return x*E[0] + y*E[1] + z*E[2]

    Enabla_rE = (
        np.conj(E[0]) * D(r_dot_E, x, y, z, 'x')
        + np.conj(E[1]) * D(r_dot_E, x, y, z, 'y')
        + np.conj(E[2]) * D(r_dot_E, x, y, z, 'z')
    )
    Hnabla_rH = (
        np.conj(H[0]) * D(r_dot_H, x, y, z, 'x')
        + np.conj(H[1]) * D(r_dot_H, x, y, z, 'y')
        + np.conj(H[2]) * D(r_dot_H, x, y, z, 'z')
    )

    E_rxnabla2_E = 0
    H_rxnabla2_H = 0
    for jj in range(3):
        E_rxnabla2_E += r2 * np.conj(E[jj]) * (D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'xx') +
                                               D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'yy') +
                                               D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'zz')) - (
            np.conj(E[jj]) * (   x**2 * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'xx')
                              +  y**2 * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'yy')
                              +  z**2 * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'zz')
                              + 2*x*y * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'xy')
                              + 2*x*z * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'xy')
                              + 2*y*z * D(lambda x,y,z: fE(x, y, z)[jj], x, y, z, 'yz'))
        )
        H_rxnabla2_H += r2 * np.conj(H[jj]) * (D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'xx') +
                                               D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'yy') +
                                               D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'zz')) - (
            np.conj(E[jj]) * (   x**2 * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'xx')
                              +  y**2 * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'yy')
                              +  z**2 * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'zz')
                              + 2*x*y * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'xy')
                              + 2*x*z * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'xy')
                              + 2*y*z * D(lambda x,y,z: fH(x, y, z)[jj], x, y, z, 'yz'))
        )

    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2

    return (factor_el * (Enabla_rE - E_rxnabla2_E) + factor_mag * (Hnabla_rH - H_rxnabla2_H)) / w
