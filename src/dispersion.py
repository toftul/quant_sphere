import scipy.constants as const
import scipy.special as sp
import numpy as np

from src import extra_special

"""
In compact form we can write them as
\begin{equation}
    \text{dispersion eq. for TE:} \qquad \frac{\mu_\text{in}}{\mu_\text{out}} \left( 1 + n_{\text{out}} z \frac{h_n^{(1)\prime} (n_{\text{out}} z)}{h_n^{(1)} (n_{\text{out}} z)}  \right) = 1 + n_\text{in} z \frac{j_n^\prime(n_\text{in} z)}{j_n (n_\text{in} z)}
\end{equation}
\begin{equation}
    \text{dispersion eq. for TM:} \qquad \frac{\varepsilon_\text{in}}{\varepsilon_\text{out}} \left( 1 + n_{\text{out}} z \frac{h_n^{(1)\prime} (n_{\text{out}} z)}{h_n^{(1)} (n_{\text{out}} z)}  \right) = 1 + n_\text{in} z \frac{j_n^\prime(n_\text{in} z)}{j_n (n_\text{in} z)}
\end{equation}
Here $z=k_0 a = \frac{\omega}{c} a \in \mathbb{Z}$ is the dimensionless **frequency**, and $n_{\text{in}} = \sqrt{\varepsilon_{\text{in}} \mu_{\text{in}}}$, $n_{\text{out}} = \sqrt{\varepsilon_{\text{out}} \mu_{\text{out}}}$. Prime shows the derivative with respect to the argument.

Which can be rewritten in a more sutable form for the numeric computation for TE modes
\begin{equation}
    f_{\text{TE}}(z) =  z \left( \frac{\mu_{\text{in}} }{\mu_{\text{out}}} n_{\text{out}} j_n h_n^\prime - n_{\text{in}} j_n^\prime h_n  \right)
    + h_n j_n \left( \frac{\mu_{\text{in}} }{\mu_{\text{out}}} - 1\right) = 0
\end{equation}
\begin{equation}
\frac{d}{dz}f_{\text{TE}}(z) =
j_n h_n^\prime n_{\text{out}} \left( 2 \frac{\mu_{\text{in}} }{\mu_{\text{out}}} - 1\right)
+ j_n^\prime h_n n_{\text{in}} \left( \frac{\mu_{\text{in}} }{\mu_{\text{out}}} - 2\right)
+ z \left\{ j_n^\prime h_n^\prime n_{\text{out}} n_{\text{in}} \left( \frac{\mu_{\text{in}} }{\mu_{\text{out}}} - 1\right)
+ \frac{\mu_{\text{in}} }{\mu_{\text{out}}} n_{\text{out}}^2 j_n h_n^{\prime \prime} - n_{\text{in}}^2 j_n^{\prime \prime} h_n \right\}
\end{equation}

and for TM
\begin{equation}
    f_{\text{TM}}(z) =  z \left( \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} n_{\text{out}} j_n h_n^\prime - n_{\text{in}} j_n^\prime h_n  \right)
    + h_n j_n \left( \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} - 1\right) = 0
\end{equation}
\begin{equation}
\frac{d}{dz}f_{\text{TM}}(z) =
j_n h_n^\prime n_{\text{out}} \left( 2 \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} - 1\right)
+ j_n^\prime h_n n_{\text{in}} \left( \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} - 2\right)
+ z \left\{ j_n^\prime h_n^\prime n_{\text{out}} n_{\text{in}} \left( \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} - 1\right)
+ \frac{\varepsilon_{\text{in}} }{\varepsilon_{\text{out}}} n_{\text{out}}^2 j_n h_n^{\prime \prime} - n_{\text{in}}^2 j_n^{\prime \prime} h_n \right\}
\end{equation}
"""


def eps_in_func(omega, particle_type, eps_dielectric):
    if particle_type == "metallic":
        eps_inf = 1
        # for Gold from Novotny p. 380
        omega_p = 13.8e15  # [1/s]
        Gamma = 0.0  # 1.075e14  # [1/s]
        return eps_inf - omega_p**2 / (omega**2 + 1j * Gamma*omega)
    elif particle_type == "dielectric":
        return eps_dielectric
    else:
        return 0


def mu_in_func(omega, particle_type, mu_dielectric=1):

    return mu_dielectric


def fTE(n, k0a, a, eps_out, mu_out, particle_type, eps_dielectric, mu_dielectric, weightOrder=0):
    z = k0a
    omega = k0a/a * const.speed_of_light
    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)

    jn = sp.spherical_jn(n, n_in * z)
    jnp = sp.spherical_jn(n, n_in * z, 1)
    hn = extra_special.spherical_h1(n, n_out * z)
    hnp = extra_special.spherical_h1p(n, n_out * z)

    kappa = mu_in/mu_out

    f = z * (kappa * n_out * jn * hnp - n_in*jnp*hn) + hn*jn*(kappa - 1)
    weight = z**weightOrder  # to aviod singularity at z = 0

    return weight * f


def fTEp(n, k0a, a, eps_out, mu_out, particle_type, eps_dielectric, mu_dielectric, weightOrder=0):
    """
        WORKS ONLY FOR NON-DISPERSIVE CASE
    """
    z = k0a
    omega = k0a/a * const.speed_of_light
    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)

    jn = sp.spherical_jn(n, n_in * z)
    jnp = sp.spherical_jn(n, n_in * z, 1)
    jnpp = extra_special.spherical_jnpp(n, n_in * z)

    hn = extra_special.spherical_h1(n, n_out * z)
    hnp = extra_special.spherical_h1p(n, n_out * z)
    hnpp = extra_special.spherical_h1pp(n, n_out * z)

    kappa = mu_in/mu_out

    # for Gold from Novotny p. 380
    # omega_p = 13.8e15  # [1/s]
    # dn_in = np.sqrt(mu_in / eps_in) * (a/const.speed_of_light)**2  * omega_p**2 / z**3

    term1 = jn*hnp*n_out * (2*kappa - 1)
    term2 = jnp*hn*n_in * (kappa - 2)
    term3 = z*(jnp*hnp*n_out*n_in*(kappa - 1) + kappa
               * n_out**2 * jn*hnpp - n_in**2 * jnpp*hn)

    f = z * (kappa * n_out * jn * hnp - n_in*jnp*hn) + hn*jn*(kappa - 1)
    df = term1 + term2 + term3
    weight = z**weightOrder  # to aviod singularity at z = 0
    dweight = 0
    if weightOrder != 0:
        dweight = weightOrder * z**(weightOrder-1)

    return weight * df + dweight * f


def fTM(n, k0a, a, eps_out, mu_out, particle_type, eps_dielectric, mu_dielectric, weightOrder=0):
    z = k0a
    omega = k0a/a * const.speed_of_light
    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)

    jn = sp.spherical_jn(n, n_in * z)
    jnp = sp.spherical_jn(n, n_in * z, 1)
    hn = extra_special.spherical_h1(n, n_out * z)
    hnp = extra_special.spherical_h1p(n, n_out * z)

    kappa = eps_in/eps_out

    weight = z**weightOrder  # to aviod singularity at z = 0
    f = z * (kappa * n_out * jn * hnp - n_in*jnp*hn) + hn*jn*(kappa - 1)

    return weight * f


def fTMp(n, k0a, a, eps_out, mu_out, particle_type, eps_dielectric, mu_dielectric, weightOrder=0):
    """
        WORKS ONLY FOR NON-DISPERSIVE CASE
    """
    z = k0a
    omega = k0a/a * const.speed_of_light
    eps_in = eps_in_func(omega, particle_type, eps_dielectric)
    mu_in = mu_in_func(omega, particle_type, mu_dielectric)
    n_in = np.sqrt(eps_in * mu_in)
    n_out = np.sqrt(eps_out * mu_out)

    jn = sp.spherical_jn(n, n_in * z)
    jnp = sp.spherical_jn(n, n_in * z, 1)
    jnpp = extra_special.spherical_jnpp(n, n_in * z)

    hn = extra_special.spherical_h1(n, n_out * z)
    hnp = extra_special.spherical_h1p(n, n_out * z)
    hnpp = extra_special.spherical_h1pp(n, n_out * z)

    kappa = eps_in/eps_out

    term1 = jn*hnp*n_out * (2*kappa - 1)
    term2 = jnp*hn*n_in * (kappa - 2)
    term3 = z*(jnp*hnp*n_out*n_in*(kappa - 1) + kappa
               * n_out**2 * jn*hnpp - n_in**2 * jnpp*hn)

    f = z * (kappa * n_out * jn * hnp - n_in*jnp*hn) + hn*jn*(kappa - 1)
    df = term1 + term2 + term3

    weight = z**weightOrder  # to aviod singularity at z = 0
    dweight = 0
    if weightOrder != 0:
        dweight = weightOrder * z**(weightOrder-1)

    return weight * df + dweight * f
