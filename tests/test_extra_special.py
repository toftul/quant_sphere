import unittest
import numpy as np
import scipy.special as sp
from scipy import integrate
from src import extra_special


class TestVSH(unittest.TestCase):

    def test_hankel_h1(self):
        # from Mathematica
        # In[4]:= N[SphericalHankelH1[4, 1 + I], 16]
        # Out[4]= 11.08530583035648 + 14.79616161710650 I
        self.assertAlmostEqual(
            np.real(extra_special.spherical_h1(4, 1+1j)),
            11.08530583035648
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_h1(4, 1+1j)),
            14.79616161710650
        )

        self.assertAlmostEqual(
            np.real(extra_special.spherical_h1p(4, 1+1j)),
            -65.42953344821775
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_h1p(4, 1+1j)),
            -5.64146759596510
        )

        self.assertAlmostEqual(
            np.real(extra_special.spherical_h1pp(4, 1+1j)),
            207.9473113848914
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_h1pp(4, 1+1j)),
            -185.4372857729240
        )

    def test_hankel_h2(self):

        self.assertAlmostEqual(
            np.real(extra_special.spherical_h2(4, 1+1j)),
            -11.09374184846489
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_h2(4, 1+1j)),
            -14.79539267401352
        )

        self.assertAlmostEqual(
            np.real(extra_special.spherical_h2p(4, 1+1j)),
            65.41502608750146
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_h2p(4, 1+1j)),
            5.66058598494007
        )

    def test_bessel(self):
        self.assertAlmostEqual(
            np.real(extra_special.spherical_jnpp(4, 1+1j)),
            0.00575721038978136
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_jnpp(4, 1+1j)),
            0.02498274414994558
        )

        self.assertAlmostEqual(
            np.real(extra_special.spherical_ynpp(4, 1+1j)),
            -185.4622685170739
        )
        self.assertAlmostEqual(
            np.imag(extra_special.spherical_ynpp(4, 1+1j)),
            -207.9415541745016
        )
