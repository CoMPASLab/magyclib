"""
MAGYC - Benchmark Methods - Sphere Fit

This module contains sphere fit appraches for magnetometer calibration.

Functions:
    sphere_fit: Standard sphere fit method.

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import numpy as np
from typing import Union, Tuple


def sphere_fit(magnetic_field: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The sphere fit method fits a sphere to a collection of data using a closed
    form for the solution. With this purpose, propose an optimization problem that
    seeks to minimize the sum:

    $$\\sum_i ((x_i-x_c)^2+(y_i-y_c)^2+(z_i-z_c)^2-r^2)^2$$

    Where x, y, and z is the data; $x_c$, $y_c$, and $z_c$ are the sphere center;
    and r is the radius.

    The method assumes that points are not in a singular configuration and are
    real numbers to solve this problem. If you have coplanar data, use a circle
    fit with svd for determining the plane, recommended [Circle Fit (Pratt method),
    by Nikolai Chernov](http://www.mathworks.com/matlabcentral/fileexchange/22643)

    Inspired by Alan Jennings, University of Dayton, implementation ([source](
    https://www.mathworks.com/matlabcentral/fileexchange/34129-sphere-fit-least-squared))

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements

    Raises:
        TypeError: If the input is not a numpy array or a list.
        ValueError: If the input is not a 3xN or Nx3 numpy array.
    """
    # Check if the input is a list and convert it to a numpy array
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)

    # Check if the input is a numpy array
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The input must be a numpy array or a list.")

    # Check if the input is a 3xN or Nx3 numpy array
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The input must be a 3xN or Nx3 numpy array.")

    # Force the array to be a Nx3 numpy array
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T

    # Compute magnetic field calibration
    mf = magnetic_field
    a_matrix = np.array(
        [
            [
                np.mean(mf[:, 0] * (mf[:, 0] - np.mean(mf[:, 0]))),
                2 * np.mean(mf[:, 0] * (mf[:, 1] - np.mean(mf[:, 1]))),
                2 * np.mean(mf[:, 0] * (mf[:, 2] - np.mean(mf[:, 2]))),
            ],
            [
                0,
                np.mean(mf[:, 1] * (mf[:, 1] - np.mean(mf[:, 1]))),
                2 * np.mean(mf[:, 1] * (mf[:, 2] - np.mean(mf[:, 2]))),
            ],
            [0, 0, np.mean(mf[:, 2] * (mf[:, 2] - np.mean(mf[:, 2])))],
        ]
    )

    a_matrix = a_matrix + a_matrix.T
    b_matrix = np.array(
        [
            [np.mean((mf[:, 0] ** 2 + mf[:, 1] ** 2 + mf[:, 2] ** 2) * (mf[:, 0] - np.mean(mf[:, 0])))],
            [np.mean((mf[:, 0] ** 2 + mf[:, 1] ** 2 + mf[:, 2] ** 2) * (mf[:, 1] - np.mean(mf[:, 1])))],
            [np.mean((mf[:, 0] ** 2 + mf[:, 1] ** 2 + mf[:, 2] ** 2) * (mf[:, 2] - np.mean(mf[:, 2])))],
        ]
    )

    hard_iron = np.array(np.linalg.lstsq(a_matrix, b_matrix, rcond=None)[0])

    # Calibrate magnetic field
    calibrated_magnetic_field = magnetic_field - hard_iron.flatten()

    return hard_iron.flatten(), calibrated_magnetic_field
