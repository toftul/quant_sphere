import unittest
from src.VSH import *
import numpy as np
import scipy.special as sp
from scipy import integrate


class TestVSH(unittest.TestCase):

    def test_orthogonality(self):
        """
            Explicit equation is from here:
            https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D1%8B%D0%B5_%D1%81%D1%84%D0%B5%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%B3%D0%B0%D1%80%D0%BC%D0%BE%D0%BD%D0%B8%D0%BA%D0%B8#%D0%9E%D1%80%D1%82%D0%BE%D0%B3%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D1%81%D1%82%D1%8C
            $$
                \int_{0}^{2 \pi} \int_{0}^{\pi} \mathbf{M}_{^e_omn} \cdot \mathbf{M}_{^e_omn} \sin \vartheta d \vartheta d \varphi =(1+\delta_{m,0}) \frac{2 \pi}{2 n+1} \frac{(n+m) !}{(n-m) !} n(n+1)\left[z_{n}(k r)\right]^2
            $$
        """
        def calculate_integral(n1, m1, n2, m2):
            #n1, m1 = 4, 2
            #n2, m2 = 4, 2
            r, theta, phi = 1.2, 2.1, 0.2

            def integrand(theta, phi):
                return np.sin(theta) * np.dot(
                            VSH_Memn(m1, n1, r, theta, phi, 1),
                            VSH_Memn(m2, n2, r, theta, phi, 1)
                )

            def integrand_re(theta, phi):
                return np.real(integrand(theta, phi))

            def integrand_im(theta, phi):
                return np.imag(integrand(theta, phi))

            ranges = [
                [0, np.pi],
                [0, 2*np.pi]
            ]

            numeric_ans_re = integrate.nquad(
                integrand_re,
                ranges,
                opts={'epsrel': 1e-4}
            )[0]

            numeric_ans_im = integrate.nquad(
                integrand_im,
                ranges,
                opts={'epsrel': 1e-4}
            )[0]

            numeric_ans = numeric_ans_re + 1j * numeric_ans_im

            cron_delta = 0
            if m1 == 0:
                cron_delta = 1
            analytic_ans = (1 + cron_delta) * 2*np.pi/(2*n1 + 1) * np.math.factorial(n1+m1) / \
                np.math.factorial(n1-m1) * n1 * (n1 + 1) * \
                sp.spherical_jn(n1, r)**2
            if (n1 != n2) or (m1 != m2):
                analytic_ans = 0

            return numeric_ans, analytic_ans

        numeric_ans, analytic_ans = calculate_integral(4, 2, 4, 2)
        self.assertAlmostEqual(np.abs(numeric_ans), np.abs(analytic_ans))
        numeric_ans, analytic_ans = calculate_integral(4, 3, 4, 3)
        self.assertAlmostEqual(np.abs(numeric_ans), np.abs(analytic_ans))
        numeric_ans, analytic_ans = calculate_integral(4, 2, 3, 3)
        self.assertAlmostEqual(np.abs(numeric_ans), 0)
        self.assertAlmostEqual(np.abs(analytic_ans), 0)

    def test_jackson(self):
        """
            Connection with Jackson
            $$
            \mathbf{M}_{mn} (\rho, \vartheta, \varphi) = -i \sqrt{n (n+1)} \cdot z_n(\rho) \mathbf{X}_{l=n,m}(\vartheta, \varphi)
            $$
        """

        def foo(m, n):
            r = 3
            theta = 1
            phi = 3

            myMmn = Mmn(m, n, r, theta, phi, 1)

            JacksonX = Xmn_Jackson(m, n, r, theta, phi)

            first = myMmn
            second = -1j*np.sqrt(n*(n+1))*sp.spherical_jn(n, r)*JacksonX
            return first, second

        first, second = foo(3, 4)
        self.assertAlmostEqual(np.linalg.norm(first - second), 0)
        first, second = foo(4, 4)
        self.assertAlmostEqual(np.linalg.norm(first - second), 0)
        first, second = foo(-2, 2)
        self.assertAlmostEqual(np.linalg.norm(first - second), 0)

    def test_bohren(self):
        """
            Connection with Bohren
            \begin{eqnarray}
            	\mathbf{M}_{mn} &=& \sqrt{\frac{2n+1}{4 \pi} \frac{(n-m)!}{(n+m)!}} \cdot \left( \mathbf{M}^{\text{B\&H}}_{emn} + i \mathbf{M}^{\text{B\&H}}_{omn} \right) , \\
            	%
            	\text{for } m>0: \qquad \qquad  \mathbf{N}_{mn} &=& \sqrt{\frac{2n+1}{4 \pi} \frac{(n-m)!}{(n+m)!}} \cdot \left( \mathbf{N}^{\text{B\&H}}_{emn} + i \mathbf{N}^{\text{B\&H}}_{omn} \right) , \\
            	%
            	\mathbf{L}_{mn} &=& \sqrt{\frac{2n+1}{4 \pi} \frac{(n-m)!}{(n+m)!}} \cdot k \left( \mathbf{L}^{\text{B\&H}}_{emn} + i \mathbf{L}^{\text{B\&H}}_{omn} \right) .
            \end{eqnarray}
        """
        def foo(m, n):
            r = 1.2
            theta = 2.1
            phi = 1.3

            index = 3

            M_my = Mmn(m, n, r, theta, phi, index)
            M_BH = np.sqrt((2*n + 1)/(4*np.pi) * np.math.factorial(n-m)
                           / np.math.factorial(n+m)) * VSHcomplex_Mmn(m, n, r, theta, phi, index)

            N_my = Nmn(m, n, r, theta, phi, index)
            N_BH = np.sqrt((2*n + 1)/(4*np.pi) * np.math.factorial(n-m)
                           / np.math.factorial(n+m)) * VSHcomplex_Nmn(m, n, r, theta, phi, index)

            delta_N = np.linalg.norm(N_my - N_BH)
            delta_M = np.linalg.norm(M_my - M_BH)

            return delta_N, delta_M

        delta_N, delta_M = foo(4, 4)
        self.assertAlmostEqual(delta_N, 0)
        self.assertAlmostEqual(delta_M, 0)
        delta_N, delta_M = foo(-4, 4)
        self.assertAlmostEqual(delta_N, 0)
        self.assertAlmostEqual(delta_M, 0)
        delta_N, delta_M = foo(3, 4)
        self.assertAlmostEqual(delta_N, 0)
        self.assertAlmostEqual(delta_M, 0)
