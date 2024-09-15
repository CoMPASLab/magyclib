"""
MAGYC - Benchmark Methods - Ellipsoid Fit

This module contains ellipsoid fit appraches for magnetometer calibration.

Functions:
    ellipsoid_fit: Standard ellipsoid fit method.
    ellipsoid_fit_fang: Ellipsoid fit method by Fang et al.

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import warnings
from typing import Tuple, Union

import numpy as np


def ellipsoid_fit(magnetic_field: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The ellipsoid fit method is based on the fact that the error model of a magnetic
    compass is an ellipsoid, and a constraint least-squares method is adopted to
    estimate the parameters of an ellipsoid by rotating the magnetic compass in
    various random orientations.

    For further details about the implementation, refer to Aleksandr Bazhin [Github
    repository](https://github.com/aleksandrbazhin/ellipsoid_fit_python), where he
    ports to python [matlab's ellipsoid fit].(http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit)

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.

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
    x, y, z = magnetic_field[:, 0], magnetic_field[:, 1], magnetic_field[:, 2]

    d_matrix = np.array(
        [
            x * x + y * y - 2 * z * z,
            x * x + z * z - 2 * y * y,
            2 * x * y,
            2 * x * z,
            2 * y * z,
            2 * x,
            2 * y,
            2 * z,
            1 - 0 * x,
        ]
    )
    d2 = np.array(x * x + y * y + z * z).T
    u = np.linalg.solve(d_matrix.dot(d_matrix.T), d_matrix.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    a_matrix = np.array(
        [[v[0], v[3], v[4], v[6]], [v[3], v[1], v[5], v[7]], [v[4], v[5], v[2], v[8]], [v[6], v[7], v[8], v[9]]]
    )

    center = np.linalg.solve(-a_matrix[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    r_matrix = translation_matrix.dot(a_matrix).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(r_matrix[:3, :3] / -r_matrix[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))
    radii *= np.sign(evals)

    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    transformation = evecs.dot(D).dot(evecs.T)

    hard_iron = center.reshape(3, 1)
    soft_iron = transformation.reshape(3, 3)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    return hard_iron.flatten(), soft_iron, calibrated_magnetic_field


def ellipsoid_fit_fang(magnetic_field: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The ellipsoid fit method is based on the fact that the error model of a magnetic
    compass is an ellipsoid, and a constraint least-squares method is adopted to
    estimate the parameters of an ellipsoid by rotating the magnetic compass in
    various random orientations.

    For further details about the implementation, refer to section (III) in J. Fang,
    H. Sun, J. Cao, X. Zhang, and Y. Tao, “A novel calibration method of magnetic
    compass based on ellipsoid fitting,” IEEE Transactions on Instrumentation
    and Measurement, vol. 60, no. 6, pp. 2053--2061, 2011.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.

    Raises:
        TypeError: If the input is not a numpy array or a list.
        ValueError: If the input is not a 3xN or Nx3 numpy array.
        RuntimeWarning: If no positive eigenvalues are found.
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
    # Design matrix (S)
    s = np.concatenate(
        [
            np.square(magnetic_field[:, [0]]),
            magnetic_field[:, [0]] * magnetic_field[:, [1]],
            np.square(magnetic_field[:, [1]]),
            magnetic_field[:, [0]] * magnetic_field[:, [2]],
            magnetic_field[:, [1]] * magnetic_field[:, [2]],
            np.square(magnetic_field[:, [2]]),
            magnetic_field[:, :],
            np.ones((magnetic_field.shape[0], 1)),
        ],
        axis=1,
    )

    # Block Matrices: S_11, S_12, S_22
    sTs = s.T @ s
    s_11, s_12, s_22 = sTs[:3, :3], sTs[:3, 3:], sTs[3:, 3:]

    # Constrain matrix C_11
    c_11 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])

    # Ellipsoid Parameters Estimation
    eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(c_11) @ (s_11 - s_12 @ np.linalg.inv(s_22) @ s_12.T))

    if np.max(eigenvals) < 0:
        warnings.warn("No positive eigenvalues: max eigenvalue = {:.6f}".format(np.max(eigenvals)), RuntimeWarning)

    a_1 = -eigenvecs[:, [np.argmax(eigenvals)]]
    a_2 = -np.linalg.inv(s_22) @ s_12.T @ a_1
    a = np.concatenate([a_1, a_2], axis=0).flatten()

    # Determine A and b
    a_matrix = np.array([[a[0], a[1]/2, a[3]/2], [a[1]/2, a[2], a[4]/2], [a[3]/2, a[4]/2, a[5]]])
    hard_iron = np.linalg.inv(-2*a_matrix) @ np.vstack(a[6:9])

    # Determine G and M
    u, s, vh = np.linalg.svd(a_matrix)
    g = u @ np.sqrt(np.diag(s)) @ vh
    soft_iron = np.linalg.inv(g)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    return hard_iron.flatten(), soft_iron, calibrated_magnetic_field
