import unittest
from src.dispersion import *
import numpy as np


class TestVSH(unittest.TestCase):

    def test_derivative(self):
        n = 3
        k0a = 1.2
        a = 1.3
        eps_out = 1.1
        mu_out = 2.0
        particle_type = "dielectric"
        eps_dielectric = 10.0 + 0.1j
        mu_dielectric = 2.0 + 0.0j

        dx = 1e-8 * k0a
        first = (fTE(n, k0a+dx, a, eps_out, mu_out, particle_type,
                     eps_dielectric, mu_dielectric)
                 - fTE(n, k0a-dx, a, eps_out, mu_out, particle_type,
                       eps_dielectric, mu_dielectric)) / (2*dx)
        second = fTEp(n, k0a, a, eps_out, mu_out, particle_type,
                      eps_dielectric, mu_dielectric)

        self.assertAlmostEqual(
            np.real(first),
            np.real(second)
        )
        self.assertAlmostEqual(
            np.imag(first),
            np.imag(second)
        )

        dx = 1e-8 * k0a
        first = (fTM(n, k0a+dx, a, eps_out, mu_out, particle_type,
                     eps_dielectric, mu_dielectric)
                 - fTM(n, k0a-dx, a, eps_out, mu_out, particle_type,
                       eps_dielectric, mu_dielectric)) / (2*dx)
        second = fTMp(n, k0a, a, eps_out, mu_out, particle_type,
                      eps_dielectric, mu_dielectric)

        self.assertAlmostEqual(
            np.real(first),
            np.real(second),
            places=6
        )
        self.assertAlmostEqual(
            np.imag(first),
            np.imag(second),
            places=6
        )
