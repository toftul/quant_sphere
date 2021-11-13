import numpy as np
from src.misc import D


def s_canonical(r, theta, phi, fE, fH, part="both"):
    """
        Electric and magnetic field functions are expected to be
            fE = fE(r, theta, phi)
            fH = fH(r, theta, phi)
        In the spherical coord basis (e_r, e_theta, e_phi)
    """
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    E, H = fE(r, theta, phi), fH(r, theta, phi)
    ExE = np.cross(np.conj(E), E, axis=0)
    HxH = np.cross(np.conj(H), H, axis=0)

    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2

    return (factor_el*ExE + factor_mag*HxH)/w


def j2_canonical(x, y, z, fE, fH, part="both", dh=1e-5):
    """
        Electric and magnetic field functions are expected to be
            fE = fE(x, y, z)
            fH = fH(x, y, z)
        In cartesian coord basis (e_x, e_y, e_z)
    """
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

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

    ELECTRIC_PART = (
        2 * np.dot(np.conjugate(E), E)
        - np.dot(
            np.conjugate(E),
            2*x*y*ddxy_E + 2*y*z*ddzy_E + 2*z*x*ddxz_E - (y*y+z*z)*ddxx_E - (x*x+z*z)*ddyy_E - (x*x+y*y)*ddzz_E
        )
        + 2j * np.dot(
            np.array([x, y, z]),
            np.conjugate(E[0])*dx_E + np.conjugate(E[1])*dy_E + np.conjugate(E[2])*dz_E
        )
    )

    MAGNETIC_PART = (
        2 * np.dot(np.conjugate(H), H)
        - np.dot(
            np.conjugate(H),
            2*x*y*ddxy_H + 2*y*z*ddzy_H + 2*z*x*ddxz_H - (y*y+z*z)*ddxx_H - (x*x+z*z)*ddyy_H - (x*x+y*y)*ddzz_H
        )
        + 2j * np.dot(
            np.array([x, y, z]),
            np.conjugate(H[0])*dx_H + np.conjugate(H[1])*dy_H + np.conjugate(H[2])*dz_H
        )
    )

    w = factor_el*np.linalg.norm(E)**2 + factor_mag*np.linalg.norm(H)**2

    return (factor_el * ELECTRIC_PART + factor_mag * MAGNETIC_PART) / w
