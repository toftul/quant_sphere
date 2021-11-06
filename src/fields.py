import numpy as np
import scipy.constants as const
import scipy.special as sp
from src.dispersion import (
    eps_in_func,
    mu_in_func,
)
from src.VSH import (
    VSHcomplex_Mmn,
    VSHcomplex_Nmn,
    Nmn,
    Mmn
)
from src import extra_special


def E_VSH(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    superscript = np.where(r <= a, 1, 3)

    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)
    k0 = omega / const.speed_of_light

    rho = np.where(
        r <= a,
        n_in * k0 * r,  # inside
        n_out * k0 * r
    )

    A_TE = 1
    B_TE = sp.spherical_jn(n, n_in*k0*a) / \
        extra_special.spherical_h1(n, n_out*k0*a)
    A_TM = 1
    B_TM = np.sqrt(eps_in*mu_out / (eps_out * mu_in)) * \
        sp.spherical_jn(n, n_in*k0*a) / \
        extra_special.spherical_h1(n, n_out*k0*a)

    # we assume that A^TE = A^TM = 1
    if mode_type == "TM":
        normalization_const = np.where(
            r <= a,
            A_TM,  # inside = A_TM
            B_TM   # outside = B_TM
        )
        return Nmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    elif mode_type == "TE":
        normalization_const = np.where(
            r <= a,
            A_TE,  # inside = A_TE
            B_TE   # outside = B_TE
        )
        return Mmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    else:
        print("ERROR")
        return(0)


def H_VSH(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    superscript = np.where(r <= a, 1, 3)

    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)
    k0 = omega / const.speed_of_light

    rho = np.where(
        r <= a,
        n_in * k0 * r,  # inside
        n_out * k0 * r
    )

    A_TE = 1
    B_TE = sp.spherical_jn(n, n_in*k0*a) / \
        extra_special.spherical_h1(n, n_out*k0*a)
    A_TM = 1
    B_TM = np.sqrt(eps_in*mu_out / (eps_out * mu_in)) * \
        sp.spherical_jn(n, n_in*k0*a) / \
        extra_special.spherical_h1(n, n_out*k0*a)

    # we assume that A^TE = A^TM = 1
    if mode_type == "TM":
        normalization_const = np.where(
            r <= a,
            -1j * np.sqrt(eps_in * const.epsilon_0
                          / (mu_in * const.mu_0)) * A_TM,  # inside = A_TM
            -1j * np.sqrt(eps_out * const.epsilon_0 / \
                          (mu_out * const.mu_0)) * B_TM   # outside = B_TM
        )
        return Mmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    elif mode_type == "TE":
        normalization_const = np.where(
            r <= a,
            -1j * np.sqrt(eps_in * const.epsilon_0
                          / (mu_in * const.mu_0)) * A_TE,  # inside = A_TE
            -1j * np.sqrt(eps_out * const.epsilon_0 / \
                          (mu_out * const.mu_0)) * B_TE   # outside = B_TE
        )
        return Nmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    else:
        print("ERROR")
        return(0)


def E_VSH_complex(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    superscript = np.where(r <= a, 1, 3)

    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)
    k0 = omega / const.speed_of_light

    rho = np.where(
        r <= a,
        n_in * k0 * r,  # inside
        n_out * k0 * r
    )

    # we assume that A^TE = A^TM = 1
    if mode_type == "TM":
        normalization_const = np.where(
            r <= a,
            1,  # inside = A_TM
            np.sqrt(eps_in*mu_out / (eps_out * mu_in)) * sp.spherical_jn(n,
                                                                         n_in*k0*a) / extra_special.spherical_h1(n, n_out*k0*a)   # outside = B_TM
        )
        return VSHcomplex_Nmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    elif mode_type == "TE":
        normalization_const = np.where(
            r <= a,
            1,  # inside = A_TE
            sp.spherical_jn(n, n_in*k0*a) / extra_special.spherical_h1(n,
                                                                       n_out*k0*a)   # outside = B_TE
        )
        return VSHcomplex_Mmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    else:
        print("ERROR")
        return(0)


def H_VSH_complex(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    superscript = np.where(r <= a, 1, 3)

    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)
    k0 = omega / const.speed_of_light

    rho = np.where(
        r <= a,
        n_in * k0 * r,  # inside
        n_out * k0 * r
    )

    # we assume that A^TE = A^TM = 1
    if mode_type == "TM":
        normalization_const = np.where(
            r <= a,
            -1j * np.sqrt(eps_in * const.epsilon_0
                          / (mu_in * const.mu_0)),  # inside = A_TM
            -1j * np.sqrt(eps_out * const.epsilon_0 / (mu_out * const.mu_0)) * np.sqrt(eps_in*mu_out / (
                eps_out * mu_in)) * sp.spherical_jn(n, n_in*k0*a) / extra_special.spherical_h1(n, n_out*k0*a)   # outside = B_TM
        )
        return VSHcomplex_Mmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    elif mode_type == "TE":
        normalization_const = np.where(
            r <= a,
            -1j * np.sqrt(eps_in * const.epsilon_0
                          / (mu_in * const.mu_0)),  # inside = A_TE
            -1j * np.sqrt(eps_out * const.epsilon_0 / (mu_out * const.mu_0)) * \
            sp.spherical_jn(n, n_in*k0*a) / extra_special.spherical_h1(n,
                                                                       n_out*k0*a)   # outside = B_TE
        )
        return VSHcomplex_Nmn(m, n, rho, theta, phi, superscript=superscript) * normalization_const
    else:
        print("ERROR")
        return(0)


def E_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    return E_VSH(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)


def H_(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    return H_VSH(m, n, r, theta, phi, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)


def E_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    #R = np.array([
    #    [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
    #    [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
    #    [            np.cos(theta),            -np.sin(theta),            0]
    #])
    E = E_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    E_cart = np.array([
        np.sin(theta)*np.cos(phi)*E[0] + np.cos(theta)
        * np.cos(phi)*E[1] - np.sin(phi)*E[2],
        np.sin(theta)*np.sin(phi)*E[0] + np.cos(theta)
        * np.sin(phi)*E[1] + np.cos(phi)*E[2],
        np.cos(theta)*E[0] - np.sin(theta)*E[1]
    ])

    return E_cart  # R @ E


def H_cart_(m, n, x, y, z, mode_type, a, omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    #R = np.array([
    #    [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
    #    [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
    #    [            np.cos(theta),            -np.sin(theta),            0]
    #])
    H = H_(m, n, r, theta, phi, mode_type, a,
           omega, particle_type, eps_out, mu_out, eps_dielectric, mu_dielectric)

    H_cart = np.array([
        np.sin(theta)*np.cos(phi)*H[0] + np.cos(theta)
        * np.cos(phi)*H[1] - np.sin(phi)*H[2],
        np.sin(theta)*np.sin(phi)*H[0] + np.cos(theta)
        * np.sin(phi)*H[1] + np.cos(phi)*H[2],
        np.cos(theta)*H[0] - np.sin(theta)*H[1]
    ])

    return H_cart  # R @ H
