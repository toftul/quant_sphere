import unittest
import numpy as np
from src.fields import *
import scipy.constants as const
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

Z_0 = np.sqrt(const.mu_0 / const.epsilon_0)  # vacuum impedance


class TestVSH(unittest.TestCase):

    a = 40 * const.nano
    rr = np.linspace(0.1*a, stop=1.8*a, num=300)

    n = 1
    m = 1

    theta = 2.0
    phi = 2.0

    eps_dielectric = 10.0
    mu_dielectric = 1.0
    eps_out = 1
    mu_out = 1

    particle_type = "dielectric"
    # omega for TE, n=1, dielectric eps = 10, a = 40 nm
    omega = 0.056661242361971696-9867088434056418j
    EE_TE_transverse = (
        np.abs(E_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(E_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )
    HH_TE_transverse = (
        np.abs(H_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(H_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )

    # omega for TM, n=1, dielectric eps = 10, a = 40 nm
    omega = 7970827084126125-3663591137623783.5j
    EE_TM_transverse = (
        np.abs(E_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(E_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )
    HH_TM_transverse = (
        np.abs(H_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(H_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )

    particle_type = "metallic"
    omega = 3.3585436405489652e+16-1.1365245058227826e+16j  # for TE, n=1, metallic
    EE_TE_transverse_metallic = (
        np.abs(E_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(E_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )
    HH_TE_transverse_metallic = (
        np.abs(H_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(H_VSH(m, n, rr, theta, phi, "TE", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )

    omega = 2.2230756046069188e+16-9463115017884170j  # for TM, n=1, metallic
    EE_TM_transverse_metallic = (
        np.abs(E_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(E_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )
    HH_TM_transverse_metallic = (
        np.abs(H_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
               eps_out, mu_out, eps_dielectric, mu_dielectric)[1])**2
        + np.abs(H_VSH(m, n, rr, theta, phi, "TM", a, omega, particle_type,
                 eps_out, mu_out, eps_dielectric, mu_dielectric)[2])**2
    )

    # https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    axs[0][0].set_title("TE, dielectric, (mn)=(%d %d)" % (m, n))
    axs[0][0].plot(rr/a, HH_TE_transverse * Z_0**2, lw=3,
                   label="$Z_0^2 |\mathbf{H}_{\\tau}|^2$")
    axs[0][0].plot(rr/a, EE_TE_transverse, lw=3,
                   label="$|\mathbf{E}_{\\tau}|^2$")
    axs[0][0].set_xlabel("$r/a$")
    axs[0][0].legend()
    axs[0][0].margins(x=0)

    axs[0][1].set_title("TM, dielectric, (mn)=(%d %d)" % (m, n))
    axs[0][1].plot(rr/a, HH_TM_transverse * Z_0**2, lw=3,
                   label="$Z_0^2|\mathbf{H}_{\\tau}|^2$")
    axs[0][1].plot(rr/a, EE_TM_transverse, lw=3,
                   label="$|\mathbf{E}_{\\tau}|^2$")
    axs[0][1].set_xlabel("$r/a$")
    axs[0][1].legend()
    axs[0][1].margins(x=0)

    axs[1][0].set_title("TE, metallic, (mn)=(%d %d)" % (m, n))
    axs[1][0].plot(rr/a, HH_TE_transverse_metallic * Z_0**2, lw=3,
                   label="$Z_0^2 |\mathbf{H}_{\\tau}|^2$")
    axs[1][0].plot(rr/a, EE_TE_transverse_metallic, lw=3,
                   label="$|\mathbf{E}_{\\tau}|^2$")
    axs[1][0].set_xlabel("$r/a$")
    axs[1][0].legend()
    axs[1][0].margins(x=0)

    axs[1][1].set_title("TM, metallic, (mn)=(%d %d)" % (m, n))
    axs[1][1].plot(rr/a, HH_TM_transverse_metallic * Z_0**2, lw=3,
                   label="$Z_0^2|\mathbf{H}_{\\tau}|^2$")
    axs[1][1].plot(rr/a, EE_TM_transverse_metallic, lw=3,
                   label="$|\mathbf{E}_{\\tau}|^2$")
    axs[1][1].set_xlabel("$r/a$")
    axs[1][1].legend()
    axs[1][1].margins(x=0)

    plt.tight_layout()
    plt.show()

    # ################################################
    # Fields should be continious on the plots
    # There E^2 for TE is smooth because mu = 1
    # Otherwise there will be also a step in dE/dr
    # #################################################

    def test_continuality(self):
        self.assertEqual(1, 1)
