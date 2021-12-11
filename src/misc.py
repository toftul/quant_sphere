import numpy as np
from scipy.misc import derivative


def delta(i, j):
    return np.equal(i, j).astype(int)


def cart2sph(x, y, z):
    """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # [0, π]
    phi = np.arctan2(y, x)  # [0, 2π]
    return r, theta, phi


def sph2cart(r, theta, phi):
    """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def rotationMatrixCyl3D(phi, axis):
    """
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        A^cart = R(phi) A^sph
    """
    if axis == 'x':
        return np.array([
            [1,           0,            0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ])
    elif axis == 'y':
        return np.array([
            [ np.cos(phi), 0, np.sin(phi)],
            [           0, 1,           0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi),  np.cos(phi), 0],
            [          0,            0, 1]
        ])
    else:
        raise NameError('Invalid axis!')


def rotationMatrixSph(theta, phi):
    """
        A^cart = R(theta, phi) A^sph
    """
    return np.array([
        [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
        [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
        [            np.cos(theta),            -np.sin(theta),            0]
    ])


def fieldTrasformSph2cart(Asph):
    """
        Asph(r, θ, φ) --- vector field in spherical coord basis

        returns Acart(x, y, z) --- vector field function in cartesian coord basis
    """
    def Acart(x, y, z):
        r, theta, phi = cart2sph(x, y, z)
        R = rotationMatrixSph(theta, phi)  # A^cart = R(theta, phi) A^sph
        return R @ Asph(r, theta, phi)

    return Acart


def D(foo, x, y, z, variables='nan', dh=1e-5, order=3):
    """
        Example of usage:
        def f(x, y, z):
            return x**2 + x*y*z

        x, y, z = 1, 1, 1

        D(f, x, y, z, 'xy')  # gives 1.000000082740371
    """
    def d3(foo, x, y, z, var='nan', n=1):
        dd = np.nan
        if var == 'x' or var == 0:
            dd = derivative(lambda x: foo(x, y, z), x, dx=dh, order=order, n=n)
        elif var == 'y' or var == 1:
            dd = derivative(lambda y: foo(x, y, z), y, dx=dh, order=order, n=n)
        elif var == 'z' or var == 2:
            dd = derivative(lambda z: foo(x, y, z), z, dx=dh, order=order, n=n)
        return dd

    if len(variables) == 1:
        return d3(foo, x, y, z, var=variables)
    elif len(variables) == 2:
        if variables[0] == variables[1]:
            """
                For the case of xx, yy, zz
                use second order derivative from scipy (parameter n=2)
                https://github.com/scipy/scipy/blob/v1.7.1/scipy/misc/common.py#L75-L145
            """
            return d3(foo, x, y, z, var=variables[0], n=2)
        else:
            return d3(
                lambda x, y, z: d3(foo, x, y, z, var=variables[0]),
                x, y, z,
                var=variables[1]
            )
