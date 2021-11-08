import numpy as np
from src.dispersion import *
from src.fields import *


def S_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    E = E_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    ExE = np.cross(np.conj(E), E, axis=0)
    HxH = np.cross(np.conj(H), H, axis=0)

    return 1/(4*np.real(omega)) * np.imag(factor_el * eps_in_out * const.epsilon_0 * ExE + factor_mag * mu_in_out * const.mu_0 * HxH)


def L_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    E = E_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    dr = 1e-4 * a
    dtheta = 1e-4
    # there were some problems with derivative over φ. Code below assumes that E,H ~ exp[imφ]
    # dphi = 1e-5
    dE_dr = (
        E_(m, n, r+dr, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - E_(m, n, r-dr, theta, phi, mode_type, a,
             omega, particle_type, eps_out, mu_ou, eps_dielectric, mu_dielectric)
    ) / (2*dr)
    dE_dtheta = (
        E_(m, n, r, theta+dtheta, phi, mode_type,
           a, omega, particle_type, eps_out, mu_out)
        - E_(m, n, r, theta-dtheta, phi, mode_type,
             a, omega, particle_type, eps_out, mu_out)
    ) / (2*dtheta)
    #dE_dphi = (
    #    E_(m, n, r, theta, phi+dphi, mode_type, a, omega, particle_type, eps_out, mu_out) -
    #    E_(m, n, r, theta, phi-dphi, mode_type, a, omega, particle_type, eps_out, mu_out)
    #) / (2*dphi)
    dE_dphi = 1j*m * E_(m, n, r, theta, phi, mode_type, a,
                        omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    dH_dr = (
        H_(m, n, r+dr, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - H_(m, n, r-dr, theta, phi, mode_type, a,
             omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dr)
    dH_dtheta = (
        H_(m, n, r, theta+dtheta, phi, mode_type,
           a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - H_(m, n, r, theta-dtheta, phi, mode_type,
             a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dtheta)
    #dH_dphi = (
    #    H_(m, n, r, theta, phi+dphi, mode_type, a, omega, particle_type, eps_out, mu_out) -
    #    H_(m, n, r, theta, phi-dphi, mode_type, a, omega, particle_type, eps_out, mu_out)
    #) / (2*dphi)
    dH_dphi = 1j*m * H_(m, n, r, theta, phi, mode_type, a,
                        omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    EdotdE_r = np.conj(E[0]) * dE_dr[0] + np.conj(E[1]) * \
        dE_dr[1] + np.conj(E[2]) * dE_dr[2]
    EdotdE_t = np.conj(E[0]) * dE_dtheta[0] + np.conj(E[1]) * \
        dE_dtheta[1] + np.conj(E[2]) * dE_dtheta[2]
    EdotdE_p = np.conj(E[0]) * dE_dphi[0] + np.conj(E[1]) * \
        dE_dphi[1] + np.conj(E[2]) * dE_dphi[2]

    HdotdH_r = np.conj(H[0]) * dH_dr[0] + np.conj(H[1]) * \
        dH_dr[1] + np.conj(H[2]) * dH_dr[2]
    HdotdH_t = np.conj(H[0]) * dH_dtheta[0] + np.conj(H[1]) * \
        dH_dtheta[1] + np.conj(H[2]) * dH_dtheta[2]
    HdotdH_p = np.conj(H[0]) * dH_dphi[0] + np.conj(H[1]) * \
        dH_dphi[1] + np.conj(H[2]) * dH_dphi[2]

    EnablaE = np.array([
        EdotdE_r,
        EdotdE_t/r + 2j/r * np.imag(np.conj(E[1]) * E[0]),
        EdotdE_p/(r * np.sin(theta)) + 2j/(r*np.sin(theta)) * np.imag(
            np.conj(E[2]) * E[0] * np.sin(theta)
            + np.conj(E[2]) * E[1] * np.cos(theta)
        )
    ])
    HnablaH = np.array([
        HdotdH_r,
        HdotdH_t/r + 2j/r * np.imag(np.conj(H[1]) * H[0]),
        HdotdH_p/(r * np.sin(theta)) + 2j/(r*np.sin(theta)) * np.imag(
            np.conj(H[2]) * H[0] * np.sin(theta)
            + np.conj(H[2]) * H[1] * np.cos(theta)
        )
    ])

    linear_momentum = 1/(4*np.real(omega)) * np.imag(
        factor_el * eps_in_out * const.epsilon_0 * EnablaE
        + factor_mag * mu_in_out * const.mu_0 * HnablaH
    )
    R = np.array([
        r, 0*r, 0*r
    ])

    return np.cross(R, linear_momentum, axis=0)


def J_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric,  part="both"):
    S = S_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, part)
    L = L_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, part)

    return L + S


def J2_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    S = S_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, part)
    L = L_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, part)
    J = L + S

    return np.real(np.conj(J[0])*J[0] + np.conj(J[1])*J[1] + np.conj(J[2])*J[2])


def W_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    E = E_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    EE = np.conj(E[0]) * E[0] + np.conj(E[1]) * E[1] + np.conj(E[2]) * E[2]
    HH = np.conj(H[0]) * H[0] + np.conj(H[1]) * H[1] + np.conj(H[2]) * H[2]

    return 0.25 * np.real(factor_el * eps_in_out * const.epsilon_0 * EE + factor_mag * mu_in_out * const.mu_0 * HH)


def j_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    J = J_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*J/W


def s_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    S = S_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*S/W


def l_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    L = L_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*L/W


def Jz_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    J = J_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    return J[0] * np.cos(theta) - J[1] * np.sin(theta)


def jz_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    J = J_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*(J[0] * np.cos(theta) - J[1] * np.sin(theta))/W


def j2_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    J = J_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    j1 = np.real(omega)*J[0]/W
    j2 = np.real(omega)*J[1]/W
    j3 = np.real(omega)*J[2]/W

    return np.real(np.conj(j1)*j1 + np.conj(j2)*j2 + np.conj(j3)*j3)


def sz_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    S = S_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*(S[0] * np.cos(theta) - S[1] * np.sin(theta))/W


def lz_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    L = L_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*(L[0] * np.cos(theta) - L[1] * np.sin(theta))/W


def PyontingVector_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    E = E_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    # check why it is 1/c^2!
    # There is probably a typo in eq.(2) of Picardi et al 2018 Optica
    return 1/(2*const.speed_of_light**2) * np.real(np.cross(np.conj(E), H, axis=0))


def J_kinetic_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    P = PyontingVector_(m, n, r, theta, phi, mode_type, a,
                        omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    r_sph = np.array([
        r,
        np.zeros(np.shape(r)),
        np.zeros(np.shape(r))
    ])

    return np.cross(r_sph, P, axis=0)


def J2_kinetic_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    P = PyontingVector_(m, n, r, theta, phi, mode_type, a,
                        omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    r_sph = np.array([
        r,
        np.zeros(np.shape(r)),
        np.zeros(np.shape(r))
    ])

    J = np.cross(r_sph, P, axis=0)

    return np.real(np.conj(J[0])*J[0] + np.conj(J[1])*J[1] + np.conj(J[2])*J[2])


def Jz_kinetic_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    J = J_kinetic_(m, n, r, theta, phi, mode_type, a,
                   omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    return J[0] * np.cos(theta) - J[1] * np.sin(theta)


def jz_kinetic_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    J = J_kinetic_(m, n, r, theta, phi, mode_type, a,
                   omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    W = W_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    return np.real(omega)*(J[0] * np.cos(theta) - J[1] * np.sin(theta))/W


def j2_kinetic_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    J = J_kinetic_(m, n, r, theta, phi, mode_type, a,
                   omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    W = W_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    j1 = np.real(omega)*J[0]/W
    j2 = np.real(omega)*J[1]/W
    j3 = np.real(omega)*J[2]/W

    return np.real(np.conj(j1)*j1 + np.conj(j2)*j2 + np.conj(j3)*j3)


def S_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type) + omega * (
        eps_in_func(omega+domega, particle_type)
        - eps_in_func(omega-domega, particle_type)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type) + omega * (
        mu_in_func(omega+domega, particle_type)
        - mu_in_func(omega-domega, particle_type)
    ) / (2*domega)

    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    E = E_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    ExE = np.cross(np.conj(E), E, axis=0)
    HxH = np.cross(np.conj(H), H, axis=0)

    return 1/(4*np.real(omega)) * np.imag(factor_el * eps_in_out * const.epsilon_0 * ExE + factor_mag * mu_in_out * const.mu_0 * HxH)


def L_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    E = E_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    dx = 1e-3*a
    dy = 1e-3*a
    dz = 1e-3*a

    # electric
    dE_dx = (
        E_cart_(m, n, x+dx, y, z, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - E_cart_(m, n, x-dx, y, z, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dx)

    dE_dy = (
        E_cart_(m, n, x, y+dy, z, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - E_cart_(m, n, x, y-dy, z, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dy)

    dE_dz = (
        E_cart_(m, n, x, y, z+dz, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - E_cart_(m, n, x, y, z-dz, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dz)

    # magnetic
    dH_dx = (
        H_cart_(m, n, x+dx, y, z, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - H_cart_(m, n, x-dx, y, z, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dx)

    dH_dy = (
        H_cart_(m, n, x, y+dy, z, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - H_cart_(m, n, x, y-dy, z, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dy)

    dH_dz = (
        H_cart_(m, n, x, y, z+dz, mode_type, a,
                omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
        - H_cart_(m, n, x, y, z-dz, mode_type, a,
                  omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    ) / (2*dz)

    EnablaE = np.array([
        np.conj(E[0]) * dE_dx[0] + np.conj(E[1])
        * dE_dx[1] + np.conj(E[2]) * dE_dx[2],
        np.conj(E[0]) * dE_dy[0] + np.conj(E[1])
        * dE_dy[1] + np.conj(E[2]) * dE_dy[2],
        np.conj(E[0]) * dE_dz[0] + np.conj(E[1])
        * dE_dz[1] + np.conj(E[2]) * dE_dz[2]
    ])
    HnablaH = np.array([
        np.conj(H[0]) * dH_dx[0] + np.conj(H[1])
        * dH_dx[1] + np.conj(H[2]) * dH_dx[2],
        np.conj(H[0]) * dH_dy[0] + np.conj(H[1])
        * dH_dy[1] + np.conj(H[2]) * dH_dy[2],
        np.conj(H[0]) * dH_dz[0] + np.conj(H[1])
        * dH_dz[1] + np.conj(H[2]) * dH_dz[2]
    ])

    linear_momentum = 1/(4*np.real(omega)) * np.imag(
        factor_el * eps_in_out * const.epsilon_0 * EnablaE
        + factor_mag * mu_in_out * const.mu_0 * HnablaH
    )
    R = np.array([
        x, y, z
    ])

    return np.cross(R, linear_momentum, axis=0)


def J_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    S = S_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part=part)
    L = L_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part=part)

    return S + L


def s_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    S = S_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*S/W


def l_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    L = L_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    W = W_(m, n, r, theta, phi, mode_type, a, omega,
           particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)

    return np.real(omega)*L/W


def j_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both"):
    s = s_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    l = l_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part)
    return s + l


def PyontingVector_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    E = E_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    # check why it is 1/c^2!
    # There is probably a typo in eq.(2) of Picardi et al 2018 Optica
    return 1/(2*const.speed_of_light**2) * np.real(np.cross(np.conj(E), H, axis=0))


def J_kinetic_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out):
    P = PyontingVector_cart_(m, n, x, y, z, mode_type,
                             a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    r_car = np.array([
        x,
        y,
        z
    ])

    return np.cross(r_car, P, axis=0)


def J2_canonical_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both", epsh=1e-2):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    r = np.sqrt(x*x + y*y + z*z)
    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    dx = epsh*a
    dy = epsh*a
    dz = epsh*a

    r2 = x**2 + y**2 + z**2

    def EE(x, y, z):
        return E_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    E = E_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    def HH(x, y, z):
        return H_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)
    H = H_cart_(m, n, x, y, z, mode_type, a, omega,
                particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    def Dx(foo, x, y, z):
        return (foo(x+dx, y, z) - foo(x-dx, y, z))/(2*dx)

    def Dy(foo, x, y, z):
        return (foo(x, y+dy, z) - foo(x, y-dy, z))/(2*dy)

    def Dz(foo, x, y, z):
        return (foo(x, y, z+dz) - foo(x, y, z-dz))/(2*dz)

    def Dxx(foo, x, y, z):
        return Dx(lambda x, y, z: Dx(foo, x, y, z), x, y, z)

    def Dyy(foo, x, y, z):
        return Dy(lambda x, y, z: Dy(foo, x, y, z), x, y, z)

    def Dzz(foo, x, y, z):
        return Dz(lambda x, y, z: Dz(foo, x, y, z), x, y, z)

    def Dxy(foo, x, y, z):
        return Dx(lambda x, y, z: Dy(foo, x, y, z), x, y, z)

    def Dxz(foo, x, y, z):
        return Dx(lambda x, y, z: Dz(foo, x, y, z), x, y, z)

    def Dyz(foo, x, y, z):
        return Dy(lambda x, y, z: Dz(foo, x, y, z), x, y, z)

    def r_dot_E(x, y, z):
        E = EE(x, y, z)
        return x*E[0] + y*E[1] + z*E[2]

    Enabla_rE = np.conj(E[0]) * Dx(r_dot_E, x, y, z) + np.conj(E[1]) * \
        Dy(r_dot_E, x, y, z) + np.conj(E[2]) * Dz(r_dot_E, x, y, z)

    E_rxnabla2_E = 0
    for jj in range(3):
        E_rxnabla2_E += r2 * np.conj(E[jj]) * (Dxx(lambda x,y,z: EE(x, y, z)[jj], x, y, z) +
                                               Dyy(lambda x,y,z: EE(x, y, z)[jj], x, y, z) +
                                               Dzz(lambda x,y,z: EE(x, y, z)[jj], x, y, z)) - (
            np.conj(E[jj]) * (   x**2 * Dxx(lambda x,y,z: EE(x, y, z)[jj], x, y, z)
                              +  y**2 * Dyy(lambda x,y,z: EE(x, y, z)[jj], x, y, z)
                              +  z**2 * Dzz(lambda x,y,z: EE(x, y, z)[jj], x, y, z)
                              + 2*x*y * Dxy(lambda x,y,z: EE(x, y, z)[jj], x, y, z)
                              + 2*x*z * Dxz(lambda x,y,z: EE(x, y, z)[jj], x, y, z)
                              + 2*y*z * Dyz(lambda x,y,z: EE(x, y, z)[jj], x, y, z))
        )

    def r_dot_H(x, y, z):
        H = HH(x, y, z)
        return x*H[0] + y*H[1] + z*H[2]

    Hnabla_rH = np.conj(H[0]) * Dx(r_dot_H, x, y, z) + np.conj(H[1]) * \
        Dy(r_dot_H, x, y, z) + np.conj(H[2]) * Dz(r_dot_H, x, y, z)

    H_rxnabla2_H = 0
    for jj in range(3):
        H_rxnabla2_H += r2 * np.conj(H[jj]) * (Dxx(lambda x, y, z: HH(x, y, z)[jj], x, y, z) +
                                               Dyy(lambda x, y, z: HH(
                                                   x, y, z)[jj], x, y, z)
                                               + Dzz(lambda x, y, z: HH(x, y, z)[jj], x, y, z)) - (
            np.conj(H[jj]) * (x**2 * Dxx(lambda x, y, z: HH(x, y, z)[jj], x, y, z) +
                              y**2 * Dyy(lambda x, y, z: HH(x, y, z)
                                         [jj], x, y, z)
                              + z**2 * Dzz(lambda x, y,
                                           z: HH(x, y, z)[jj], x, y, z)
                              + 2*x*y * Dxy(lambda x, y,
                                            z: HH(x, y, z)[jj], x, y, z)
                              + 2*x*z * Dxz(lambda x, y,
                                            z: HH(x, y, z)[jj], x, y, z)
                              + 2*y*z * Dyz(lambda x, y,
                                            z: HH(x, y, z)[jj], x, y, z)
                              )
        )

    return 1/(4*omega) * (const.epsilon_0*eps_in_out * factor_el * (Enabla_rE - E_rxnabla2_E) + const.mu_0*mu_in_out * factor_mag * (Hnabla_rH - H_rxnabla2_H))


def J2_canonical2_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric, part="both", epsh=1e-2):
    factor_el = 1
    factor_mag = 1
    if part == "electric":
        factor_mag = 0
    if part == "magnetic":
        factor_el = 0

    domega = np.real(omega) * 1e-6
    eps_in_tilda = eps_in_func(omega, particle_type, eps_dielectric) + omega * (
        eps_in_func(omega+domega, particle_type, eps_dielectric)
        - eps_in_func(omega-domega, particle_type, eps_dielectric)
    ) / (2*domega)
    mu_in_tilda = mu_in_func(omega, particle_type, mu_dielectric) + omega * (
        mu_in_func(omega+domega, particle_type, mu_dielectric)
        - mu_in_func(omega-domega, particle_type, mu_dielectric)
    ) / (2*domega)

    r = np.sqrt(x*x + y*y + z*z)
    eps_in_out = np.where(
        r <= a,
        eps_in_tilda,  # inside
        eps_out
    )
    mu_in_out = np.where(
        r <= a,
        mu_in_tilda,  # inside
        mu_out
    )

    dx = epsh*a
    dy = epsh*a
    dz = epsh*a

    r2 = x**2 + y**2 + z**2

    def EE(x, y, z):
        return E_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    def HH(x, y, z):
        return H_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    E = EE(x, y, z)
    H = HH(x, y, z)

    # ---- ELECTRIC -----
    dx_E = (
        EE(x+dx, y, z) - EE(x-dx, y, z)
    )/(2*dx)
    dy_E = (
        EE(x, y+dy, z) - EE(x, y-dy, z)
    )/(2*dy)
    dz_E = (
        EE(x, y, z+dz) - EE(x, y, z-dz)
    )/(2*dz)

    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    ddxx_E = 1/(dx*dx) * (
        EE(x+dx, y, z) - 2*EE(x, y, z) + EE(x-dx, y, z)
    )
    ddyy_E = 1/(dy*dy) * (
        EE(x, y+dy, z) - 2*EE(x, y, z) + EE(x, y-dy, z)
    )
    ddzz_E = 1/(dz*dz) * (
        EE(x, y, z+dz) - 2*EE(x, y, z) + EE(x, y, z-dz)
    )

    ddxy_E = 1/(4*dx*dy) * (
        EE(x+dx, y+dy, z) - EE(x-dx, y+dy, z)
        - EE(x+dx, y-dy, z) + EE(x-dx, y-dy, z)
    )
    ddxz_E = 1/(4*dx*dz) * (
        EE(x+dx, y, z+dz) - EE(x-dx, y, z+dz)
        - EE(x+dx, y, z-dz) + EE(x-dx, y, z-dz)
    )
    ddzy_E = 1/(4*dz*dy) * (
        EE(x, y+dy, z+dz) - EE(x, y+dy, z-dz)
        - EE(x, y-dy, z+dz) + EE(x, y-dy, z-dz)
    )

    # ---- MAGNETIC -----
    dx_H = (
        HH(x+dx, y, z) - HH(x-dx, y, z)
    )/(2*dx)
    dy_H = (
        HH(x, y+dy, z) - HH(x, y-dy, z)
    )/(2*dy)
    dz_H = (
        HH(x, y, z+dz) - HH(x, y, z-dz)
    )/(2*dz)

    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    ddxx_H = 1/(dx*dx) * (
        HH(x+dx, y, z) - 2*HH(x, y, z) + HH(x-dx, y, z)
    )
    ddyy_H = 1/(dy*dy) * (
        HH(x, y+dy, z) - 2*HH(x, y, z) + HH(x, y-dy, z)
    )
    ddzz_H = 1/(dz*dz) * (
        HH(x, y, z+dz) - 2*HH(x, y, z) + HH(x, y, z-dz)
    )

    ddxy_H = 1/(4*dx*dy) * (
        HH(x+dx, y+dy, z) - HH(x-dx, y+dy, z)
        - HH(x+dx, y-dy, z) + HH(x-dx, y-dy, z)
    )
    ddxz_H = 1/(4*dx*dz) * (
        HH(x+dx, y, z+dz) - HH(x-dx, y, z+dz)
        - HH(x+dx, y, z-dz) + HH(x-dx, y, z-dz)
    )
    ddzy_H = 1/(4*dz*dy) * (
        HH(x, y+dy, z+dz) - HH(x, y+dy, z-dz)
        - HH(x, y-dy, z+dz) + HH(x, y-dy, z-dz)
    )

    ELECTRIC_PART = 2 * np.dot(np.conjugate(E), E)
    ELECTRIC_PART -= np.dot(
        np.conjugate(E),
        2*x*y*ddxy_E + 2*y*z*ddzy_E + 2*z*x*ddxz_E
        - (y*y+z*z)*ddxx_E - (x*x+z*z)*ddyy_E - (x*x+y*y)*ddzz_E
    )
    ELECTRIC_PART += 2j * np.dot(
        np.array([x, y, z]),
        np.conjugate(E[0])*dx_E + np.conjugate(E[1])
        * dy_E + np.conjugate(E[2])*dz_E
    )

    MAGNETIC_PART = 2 * np.dot(np.conjugate(H), H)
    MAGNETIC_PART -= np.dot(
        np.conjugate(H),
        2*x*y*ddxy_H + 2*y*z*ddzy_H + 2*z*x*ddxz_H
        - (y*y+z*z)*ddxx_H - (x*x+z*z)*ddyy_H - (x*x+y*y)*ddzz_H
    )
    MAGNETIC_PART += 2j * np.dot(
        np.array([x, y, z]),
        np.conjugate(H[0])*dx_H + np.conjugate(H[1])
        * dy_H + np.conjugate(H[2])*dz_H
    )

    return 1/(4*omega) * (const.epsilon_0*eps_in_out * factor_el * ELECTRIC_PART + const.mu_0*mu_in_out * factor_mag * MAGNETIC_PART)
