import numpy as np
import scipy.special as sp

from src import extra_special


def VSH_Memn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.17), (4.18)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    Mt = 0.0
    Mp = 0.0

    zn = np.asarray(0.0)
    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extra_special.spherical_h1(n, rho)
    )
    """
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yv(n, rho)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

    Mt = -m / np.sin(theta) * np.sin(m * phi) * sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Mp = -np.cos(m * phi) * dPnm

    Mr = np.zeros(np.shape(Mp*zn))

    return np.array([Mr, Mt*zn, Mp*zn])


def VSH_Momn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.17), (4.18)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15  # for vectorization

    Mt = 0.0
    Mp = 0.0

    zn = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extra_special.spherical_h1(n, rho))
    """
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

    Mt = m / np.sin(theta) * np.cos(m * phi) * sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Mp = -np.sin(m * phi) * dPnm

    Mr = np.zeros(np.shape(Mp*zn))

    return np.array([Mr, Mt*zn, Mp*zn])


def VSH_Nemn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.19), (4.20)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    Nr = 0.0
    Nt = 0.0
    Np = 0.0

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extra_special.spherical_h1(n, rho))
    znp = np.where(superscript == 1, sp.spherical_jn(
        n, rho, 1), extra_special.spherical_h1p(n, rho))

    """
    zn = 0
    znp = 0
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
        znp = spherical_jn(n, rho, 1)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
        znp = spherical_yn(n, rho, 1)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
        znp = spherical_h1p(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
        znp = spherical_h2p(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

    Pnm = sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Nr = zn/rho * np.cos(m*phi) * n * (n+1) * Pnm
    Nt = np.cos(m * phi) * dPnm * (zn/rho + znp)
    Np = -m * np.sin(m * phi) * Pnm / np.sin(theta) * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def VSH_Nomn(m, n, rho, theta, phi, superscript=1):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.19), (4.20)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    Nr = 0.0
    Nt = 0.0
    Np = 0.0

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extra_special.spherical_h1(n, rho))
    znp = np.where(superscript == 1, sp.spherical_jn(
        n, rho, 1), extra_special.spherical_h1p(n, rho))

    """
    zn = 0
    znp = 0
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
        znp = spherical_jn(n, rho, 1)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
        znp = spherical_yn(n, rho, 1)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
        znp = spherical_h1p(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
        znp = spherical_h2p(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

    Pnm = sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Nr = zn/rho * np.sin(m * phi) * n * (n+1) * Pnm
    Nt = np.sin(m * phi) * dPnm * (zn/rho + znp)
    Np = m * np.cos(m * phi) * Pnm / np.sin(theta) * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def VSHcomplex_Mmn(m, n, rho, theta, phi, superscript=1):
    '''
        complex vector Harmonics in spherical coordinates
        (r, θ, φ)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''

    if m == 0:
        return VSH_Memn(m, n, rho, theta, phi, superscript)
    elif m > 0:
        Memn = VSH_Memn(m, n, rho, theta, phi, superscript)
        Momn = VSH_Momn(m, n, rho, theta, phi, superscript)

        return Memn + 1j * Momn
    else:
        # for m < 0
        m = np.abs(m)
        Memn = VSH_Memn(m, n, rho, theta, phi, superscript)
        Momn = VSH_Momn(m, n, rho, theta, phi, superscript)

        return (Memn - 1j * Momn) * (-1)**m * sp.factorial(n - m) / sp.factorial(n + m)


def VSHcomplex_Nmn(m, n, rho, theta, phi, superscript=1):
    '''
        complex vector Harmonics in spherical coordinates
        (r, θ, φ)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = -n, ..., n
            superscript = 1, 2, 3, 4
    '''

    if m == 0:
        return VSH_Nemn(m, n, rho, theta, phi, superscript)
    elif m > 0:
        Nemn = VSH_Nemn(m, n, rho, theta, phi, superscript)
        Nomn = VSH_Nomn(m, n, rho, theta, phi, superscript)

        return Nemn + 1j * Nomn
    else:
        # for m < 0
        m = np.abs(m)
        Nemn = VSH_Nemn(m, n, rho, theta, phi, superscript)
        Nomn = VSH_Nomn(m, n, rho, theta, phi, superscript)

        return (Nemn - 1j * Nomn) * (-1)**m * sp.factorial(n - m) / sp.factorial(n + m)


def Mmn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = -n, .., 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    zn = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extra_special.spherical_h1(n, rho)
    )

    Mt = 1j * m / np.sin(theta) * zn * sp.sph_harm(m, n, phi, theta)

    dYmn = m/np.tan(theta) * sp.sph_harm(m, n, phi, theta)
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm(m+1, n, phi, theta)

    Mp = -zn * dYmn

    Mr = np.zeros(np.shape(Mp))

    return np.array([Mr, Mt, Mp])


def Nmn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = -n, .., 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extra_special.spherical_h1(n, rho)
    )
    znp = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho, 1),
        extra_special.spherical_h1p(n, rho)
    )

    Ymn = sp.sph_harm(m, n, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm(m+1, n, phi, theta)

    Nr = n*(n+1) * zn/rho * Ymn
    Nt = 1/rho * (zn + rho * znp) * dYmn
    Np = 1j*m/np.sin(theta) * 1/rho * (zn + rho * znp) * Ymn

    return np.array([Nr, Nt, Np])


def Lmn(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)

        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = -n, .., 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extra_special.spherical_h1(n, rho)
    )
    znp = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho, 1),
        extra_special.spherical_h1p(n, rho)
    )

    Ymn = sp.sph_harm(m, n, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm(m+1, n, phi, theta)

    Lr = znp * Ymn
    Lt = zn/rho * dYmn
    Lp = 1j*m/np.sin(theta) * zn/rho * Ymn

    return np.array([Lr, Lt, Lp])


def Xmn_Jackson(m, n, rho, theta, phi):
    '''
     from Jackson book paragraph 9.7
    '''
    Ymn = sp.sph_harm(m, n, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm(m+1, n, phi, theta)

    return (-1)/np.sqrt(n*(n+1)) * np.array([
        np.zeros(np.shape(theta * rho * phi)),
        m/np.sin(theta) * Ymn,
        1j * dYmn
    ])
