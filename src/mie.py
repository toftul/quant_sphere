import scipy.special as sp
from src.extra_special import (
    spherical_h1,
    spherical_h1p
)


def Mie_an(n, m, x, mu=1):
    """
    Electric Mie coefficent. For detatails see Bohren p. 100

    Arguments:
        - `m = N1/N = n_p / n_m` : relative refractive index;
        - `x = n_m k0 a` : size parameter;
        - `n` : 2^n multipole order;
        - `mu = mu_p / mu_m` : relative magnetic permittivity
    """
    mx = m * x
    jnmx = sp.spherical_jn(n, mx)
    jnx = sp.spherical_jn(n, x)
    h1nx = spherical_h1(n, x)
    xjnx_p = jnx + x * sp.spherical_jn(n, x, 1)
    mxjnmx_p = jnmx + mx * sp.spherical_jn(n, mx, 1)
    xh1nx_p = h1nx + x * spherical_h1p(n, x)

    return (m**2 * jnmx * xjnx_p - mu * jnx * mxjnmx_p) / (m**2 * jnmx * xh1nx_p - mu * h1nx * mxjnmx_p)


def Mie_bn(n, m, x, mu=1):
    """
    Magnetic Mie coefficent. For detatails see Bohren p. 100

    Arguments:
        - `m = N1 / N = n_p / n_m` : relative refractive index;
        - `x = n_m k0 a` : size parameter;
        - `n` : 2^n multipole order;
        - `mu = mu1 / mu =  mu_p / mu_m` : relative magnetic permittivity
    """
    mx = m * x
    jnmx = sp.spherical_jn(n, mx)
    jnx = sp.spherical_jn(n, x)
    h1nx = spherical_h1(n, x)
    xjnx_p = jnx + x * sp.spherical_jn(n, x, 1)
    mxjnmx_p = jnmx + mx * sp.spherical_jn(n, mx, 1)
    xh1nx_p = h1nx + x * spherical_h1p(n, x)

    return (mu * jnmx * xjnx_p - jnx * mxjnmx_p) / (mu * jnmx * xh1nx_p - h1nx * mxjnmx_p)
