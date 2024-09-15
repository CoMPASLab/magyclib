"""
MAGYC - Benchmark Methods - TWOSTEP

This module contains TWOSTEP appraches for magnetometer calibration.

Functions:
    twostep_hi: TWOSTEP method for hard-iron estimation.
    twostep_hsi: TWOSTEP method for hard-iron and soft-iron estimation.

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import warnings
from typing import Tuple, Union

import numpy as np


def twostep_hi(magnetic_field: Union[np.ndarray, list], reference_magnetic_field: Union[np.ndarray, list],
               max_iterations: int = 2000, measurement_noise_std: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    The TWOSTEP method proposes a fast, robust algorithm for estimating magnetometer
    biases when the attitude is unknown. This algorithm combines the convergence
    in a single step of a heuristic algorithm currently in use with the correct
    treatment of the statistics of the measurements and does without discarding
    data.

    This algorithm was the in a first publication developed for the estimation of
    the hard-iron (Alonso, R. Shuster, M.D. (2002a). TWOSTEP: A fast, robust
    algorithm for attitude-independent magnetometer-bias determination. Journal
    of the Astronautical Sciences, 50(4):433-452.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        reference_magnetic_field (numpy.ndarray or list): Reference magnetic field
            measurements in a 3, or 1x3, or 3x1 numpy array or list.
        max_iterations (int): Maximum number of iterations for the second step.
        measurement_noise_std (float): Standard deviation that characterizes the
            measurements' noise, by default 0.001 G.

    Returns:
        hard_iron (numpy.ndarray): Estimated hard-iron bias as a 3x1 numpy array.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field as a
            3xN numpy array.

    Raises:
        TypeError: If the magnetic field or reference magnetic field inputs are not
            numpy arrays or lists.
        ValueError: If the magnetic field input is not a 3xN or Nx3 numpy array, if
            the reference magnetic field input is not a 3, or 1x3, or 3x1 numpy array,
            if the maximum number of iterations is not a positive integer, or if the
            measurement noise standard deviation is not a positive float.
    """
    # Check if the magnetic field input is a list and convert it to a numpy array
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)

    # Check if the reference magnetic field input is a list and convert it to a numpy array
    if isinstance(reference_magnetic_field, list):
        reference_magnetic_field = np.array(reference_magnetic_field).flatten()

    # Check if the magnetic field input is a numpy array
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field input must be a numpy array or a list.")

    # Check if the reference magnetic field input is a numpy array
    if not isinstance(reference_magnetic_field, np.ndarray):
        raise TypeError("The reference magnetic field input must be a numpy array or a list.")

    # Check if the magnetic field input is a 3xN or Nx3 numpy array
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field input must be a 3xN or Nx3 numpy array.")

    # Check if the reference magnetic field input is a 3, numpy array
    if reference_magnetic_field.ndim != 1 and reference_magnetic_field.size != 3:
        raise ValueError("The reference magnetic field input must be a 3, or 1x3, or 3x1 numpy array.")

    # Force the magnetic field array to be a Nx3 numpy array
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T

    # Check that the maximum number of iterations is a positive integer
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("The maximum number of iterations must be a positive integer.")

    # Check that the measurement noise standard deviation is a positive float
    if not isinstance(measurement_noise_std, float) or measurement_noise_std <= 0:
        raise ValueError("The measurement noise standard deviation must be a positive float.")

    # Compute magnetic field calibration
    mf = magnetic_field

    # First step
    b0 = np.zeros((3, 1))

    # Effective measurement from paper equation (3a)
    b_matrix = np.ones((mf.shape)) * reference_magnetic_field
    z_k = (np.square(np.linalg.norm(mf, axis=1)) - np.square(np.linalg.norm(b_matrix, axis=1))).reshape(-1, 1)

    # Sensor measurements noise modeled as white gaussian with standard deviation epsilon_k
    epsilon_sq_k = np.ones(mf.shape) * (measurement_noise_std**2)

    # Sensor error scalar measurement noise characterization as gaussian.
    # Gaussian distribution mean, equation (7a)
    mu_k = -np.sum(epsilon_sq_k, axis=1, keepdims=True)

    # Gaussian distribution variance, equation (5.15)
    sigma_sq_k = (
        4
        * (
            (mf.reshape(-1, 1, 3) - b0.reshape(1, 3))
            @ np.apply_along_axis(np.diag, 1, epsilon_sq_k)
            @ (mf.reshape(-1, 3, 1) - b0)
        )
        + 2 * np.apply_along_axis(lambda x: np.square(np.trace(np.diag(x))), 1, epsilon_sq_k).reshape(-1, 1, 1)
    ).reshape(-1, 1)

    # Calculate  centered sigma squared, equation (14)
    sigma_sq_bar = 1 / np.sum(1 / sigma_sq_k)

    # Center  the  data
    mu_bar, mu_k_tilde = _center_data(mu_k, sigma_sq_k, sigma_sq_bar)
    z_bar, z_k_tilde = _center_data(z_k, sigma_sq_k, sigma_sq_bar)
    b_bar, b_k_tilde = _center_data(b_matrix, sigma_sq_k, sigma_sq_bar)

    # Offset and error covariance matrix calculation from paper equations (33) and (34)
    F_bb_tilde = np.einsum(
        "ijk->jk",
        (4 / sigma_sq_k.reshape(-1, 1, 1)) * (b_k_tilde.reshape(-1, 3, 1) @ b_k_tilde.reshape(-1, 1, 3)),
    )
    F_zb = np.einsum(
        "ijk->jk",
        ((z_k_tilde - mu_k_tilde) * (2 / sigma_sq_k)).reshape(-1, 1, 1) * b_k_tilde.reshape(-1, 3, 1),
    )
    b = np.linalg.inv(F_bb_tilde) @ F_zb

    # Second Step: Iterative
    F_bb_bar = (4 / sigma_sq_bar) * (b_bar.reshape(-1, 1) - b) @ (b_bar.reshape(-1, 1) - b).T
    b_asterisk = np.copy(b)

    if np.max(np.diag(F_bb_bar) / np.diag(F_bb_tilde)) > 0.001:
        F_bb = F_bb_tilde + F_bb_bar
        gg = (F_bb_tilde @ (b - b_asterisk)) - (1 / sigma_sq_bar) * (
            z_bar - 2 * (b_bar @ b) + np.linalg.norm(b) ** 2 - mu_bar
        ) * 2 * (b_bar.reshape(-1, 1) - b)
        bn = b - np.linalg.inv(F_bb) @ gg

        iter = 1
        while ((bn - b).T @ F_bb @ (bn - b)) > 0.001:
            b = np.copy(bn)
            gg = (F_bb_tilde @ (b - b_asterisk)) - (1 / sigma_sq_bar) * (
                z_bar - 2 * (b_bar @ b) + np.linalg.norm(b) ** 2 - mu_bar
            ) * 2 * (b_bar.reshape(-1, 1) - b)
            F_bb_bar = (4 / sigma_sq_bar) * (b_bar.reshape(-1, 1) - b) @ (b_bar.reshape(-1, 1) - b).T
            F_bb = F_bb_tilde + F_bb_bar
            bn = b - np.linalg.inv(F_bb) @ gg

            iter += 1
            if iter > max_iterations:
                warnings.warn("Second step: Maximum number of iterations reached.", RuntimeWarning)
                break

    hard_iron = bn.reshape(3, 1)

    # Calibrate magnetic field
    calibrated_magnetic_field = magnetic_field - hard_iron.flatten()

    return hard_iron.flatten(), calibrated_magnetic_field


def twostep_hsi(magnetic_field: Union[np.ndarray, list], reference_magnetic_field: Union[np.ndarray, list],
                max_iterations: int = 2000, measurement_noise_std: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    The TWOSTEP method proposes a fast, robust algorithm for estimating magnetometer
    biases when the attitude is unknown. This algorithm combines the convergence
    in a single step of a heuristic algorithm currently in use with the correct
    treatment of the statistics of the measurements and does without discarding
    data.

    This algorithm was extended in a second iteration to compute also the soft-iron
    (Alonso, R. Shuster, M.D. (2002b). Complete linear attitude-independent
    magnetometer calibration. Journal of the Astronautical Science, 50(4):477-490).

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        reference_magnetic_field (numpy.ndarray or list): Reference magnetic field
            measurements in a 3, or 1x3, or 3x1 numpy array or list.
        max_iterations (int): Maximum number of iterations for the second step.
        measurement_noise_std (float): Standard deviation that characterizes the
            measurements' noise, by default 0.001 G.

    Returns:
        hard_iron (numpy.ndarray): Estimated hard-iron bias as a 3x1 numpy array.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field as a
            3xN numpy array.

    Raises:
        TypeError: If the magnetic field or reference magnetic field inputs are not
            numpy arrays or lists.
        ValueError: If the magnetic field input is not a 3xN or Nx3 numpy array, if
            the reference magnetic field input is not a 3, or 1x3, or 3x1 numpy array,
            if the maximum number of iterations is not a positive integer, or if the
            measurement noise standard deviation is not a positive float.
    """
    # Check if the magnetic field input is a list and convert it to a numpy array
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)

    # Check if the reference magnetic field input is a list and convert it to a numpy array
    if isinstance(reference_magnetic_field, list):
        reference_magnetic_field = np.array(reference_magnetic_field).flatten()

    # Check if the magnetic field input is a numpy array
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field input must be a numpy array or a list.")

    # Check if the reference magnetic field input is a numpy array
    if not isinstance(reference_magnetic_field, np.ndarray):
        raise TypeError("The reference magnetic field input must be a numpy array or a list.")

    # Check if the magnetic field input is a 3xN or Nx3 numpy array
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field input must be a 3xN or Nx3 numpy array.")

    # Check if the reference magnetic field input is a 3, numpy array
    if reference_magnetic_field.ndim != 1 and reference_magnetic_field.size != 3:
        raise ValueError("The reference magnetic field input must be a 3, or 1x3, or 3x1 numpy array.")

    # Force the magnetic field array to be a Nx3 numpy array
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T

    # Check that the maximum number of iterations is a positive integer
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("The maximum number of iterations must be a positive integer.")

    # Check that the measurement noise standard deviation is a positive float
    if not isinstance(measurement_noise_std, float) or measurement_noise_std <= 0:
        raise ValueError("The measurement noise standard deviation must be a positive float.")

    # Compute magnetic field calibration
    mf = magnetic_field

    stop_tol = 1e-24  # Stop Condition from Alonso paper
    I3 = np.eye(3, dtype=np.float64)

    # TWOSTEP Centered estimate
    # Set initial guess for b and D.
    b0 = np.zeros((3, 1))
    d0 = np.zeros((3, 3))

    # Form L matrix, equations (5.10b) and (5.12a)
    l1 = 2 * mf
    l2 = -np.square(mf)
    l3 = -2 * mf[:, [0]] * mf[:, [1]]
    l4 = -2 * mf[:, [0]] * mf[:, [2]]
    l5 = -2 * mf[:, [1]] * mf[:, [2]]
    L_k = np.concatenate([l1, l2, l3, l4, l5], axis=1)

    # Compute sensor error as scalar measurement, equation (5.7a)
    h_matrix = np.ones((mf.shape)) * reference_magnetic_field
    z_k = (np.square(np.linalg.norm(mf, axis=1)) - np.square(np.linalg.norm(h_matrix, axis=1))).reshape(-1, 1)

    # Sensor measurements noise modeled as white gaussian with standard deviation epsilon_k
    epsilon_sq_k = np.ones(mf.shape) * (measurement_noise_std**2)

    # Sensor error scalar measurement noise characterization as gaussian.
    # Gaussian distribution mean, equation (5.14)
    mu_k = -np.sum(epsilon_sq_k, axis=1, keepdims=True)

    # Gaussian distribution variance, equation (5.15)
    sigma_sq_k = (
        4
        * np.einsum(
            "ijk->ikj",
            np.tile(I3 + d0, (mf.shape[0], 1, 1)) @ mf.reshape(-1, 3, 1) - np.tile(b0, (mf.shape[0], 1, 1)),
        )
        @ np.apply_along_axis(np.diag, 1, epsilon_sq_k)
        @ (np.tile(I3 + d0, (mf.shape[0], 1, 1)) @ mf.reshape(-1, 3, 1) - np.tile(b0, (mf.shape[0], 1, 1)))
        + 2 * np.apply_along_axis(lambda x: np.square(np.trace(np.diag(x))), 1, epsilon_sq_k).reshape(-1, 1, 1)
    ).reshape(-1, 1)

    # Calculate centered sigma squared, equation (5.18)
    sigma_sq_bar = 1 / np.sum(1 / sigma_sq_k)

    # Center the data, equation (5.19)
    mu_bar, mu_k_tilde = _center_data(mu_k, sigma_sq_k, sigma_sq_bar)
    z_bar, z_k_tilde = _center_data(z_k, sigma_sq_k, sigma_sq_bar)
    L_bar, L_k_tilde = _center_data(L_k, sigma_sq_k, sigma_sq_bar)

    # Compute fisher information matrix
    I_fisher_tilde, I_fishinv_tilde = _TS_fisher_centered(sigma_sq_k, L_k_tilde)

    # Compute centered estimate, equation (5.24)
    f_matrix = np.einsum(
        "ijk->jk",
        (
            (1 / sigma_sq_k).reshape(-1, 1, 1)
            * ((z_k_tilde - mu_k_tilde).reshape(-1, 1, 1) * L_k_tilde.reshape(-1, 9, 1))
        ),
    )
    theta_0_tilde = I_fishinv_tilde @ f_matrix

    # TWOSTEP Center correction
    theta_n, theta_np1 = theta_0_tilde, theta_0_tilde  # Initiate theta for  first  iteration
    n = 0  # Initialise  iteration counter
    TS_err = np.Inf  # Initial  condition  for  error.

    # ABC is used to remove intensive calculations out of for loop
    abc = -np.einsum(
        "ijk->jk",
        (
            (1 / sigma_sq_k).reshape(-1, 1, 1)
            * ((z_k_tilde - mu_k_tilde).reshape(-1, 1, 1) * L_k_tilde.reshape(-1, 9, 1))
        ),
    )

    while TS_err > stop_tol and n < max_iterations:
        if n != 0:  # If  we are not  in the first	iteration
            theta_n = theta_np1

        # Extract  c  and  E  components
        c, e_matrix = _theta_to_c_E(theta_n)

        # Compute  second  derivative  of  b^2  wrt theta
        tmp = np.linalg.solve((np.eye(3) + e_matrix), c) @ np.linalg.solve((np.eye(3) + e_matrix), c).T
        dbsqdtheta_p = np.concatenate(
            [
                2 * np.linalg.solve((np.eye(3) + e_matrix), c),
                -np.diag(tmp).reshape(3, 1),
                np.vstack([-2 * tmp[0, 1], -2 * tmp[0, 2], -2 * tmp[1, 2]]),
            ]
        )
        # Compute gradient of J
        dJdThetap_tilde = abc + I_fisher_tilde @ theta_n
        dJdThetap_bar = (
            -(1 / sigma_sq_bar)
            * (L_bar.reshape(-1, 1) - dbsqdtheta_p)
            * (z_bar - (L_bar.reshape(1, -1) @ theta_n) + (c.T @ np.linalg.solve((np.eye(3) + e_matrix), c)) - mu_bar)
        )
        dJdTheta = dJdThetap_tilde + dJdThetap_bar

        # Calculate Fisher matrix
        I_fisher_bar = _TS_fisher_center(sigma_sq_bar, L_bar, dbsqdtheta_p)

        # Update theta
        theta_np1 = theta_n - np.linalg.solve((I_fisher_tilde + I_fisher_bar), dJdTheta)

        # Compute error
        TS_err = ((theta_np1 - theta_n).T @ (I_fisher_tilde + I_fisher_bar)) @ (theta_np1 - theta_n)
        n += 1

    b, d_matrix = _theta_to_b_D(theta_np1)

    # Extract covariance matrix
    m_cd = np.array(
        [
            [b[0, 0], 0, 0, b[1, 0], b[2, 0], 0],
            [0, b[1, 0], 0, b[0, 0], 0, b[2, 0]],
            [0, 0, b[2, 0], 0, b[0, 0], b[1, 0]],
        ]
    )
    m_ed = np.array(
        [
            [2 * d_matrix[0, 0], 0, 0, 2 * d_matrix[0, 1], 2 * d_matrix[0, 2], 0],
            [0, 2 * d_matrix[1, 1], 0, 2 * d_matrix[0, 1], 0, 2 * d_matrix[1, 2]],
            [0, 0, 2 * d_matrix[2, 2], 0, 2 * d_matrix[0, 2], 2 * d_matrix[1, 2]],
            [d_matrix[0, 1], d_matrix[0, 1], 0, d_matrix[0, 0] + d_matrix[1, 1], d_matrix[1, 2], d_matrix[0, 2]],
            [d_matrix[0, 2], 0, d_matrix[0, 2], d_matrix[1, 2], d_matrix[0, 0] + d_matrix[2, 2], d_matrix[0, 1]],
            [0, d_matrix[1, 2], d_matrix[1, 2], d_matrix[0, 2], d_matrix[0, 1], d_matrix[1, 1] + d_matrix[2, 2]],
        ]
    )
    dbD_dcE = np.eye(9)
    dbD_dcE[:3, :3], dbD_dcE[:3, 3:] = np.eye(3) + d_matrix, m_cd
    dbD_dcE[3:, :3], dbD_dcE[3:, 3:] = np.zeros((6, 3)), 2 * np.eye(6) @ m_ed
    dbD_dcE = np.linalg.inv(dbD_dcE)
    # Cov_est = dbD_dcE @ np.linalg.solve((I_fisher_tilde + I_fisher_bar), dbD_dcE.T)

    # END   TWOSTEP
    hard_iron = (np.linalg.inv(np.eye(3) + d_matrix)) @ b
    soft_iron = np.linalg.inv(np.eye(3) + d_matrix)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    return hard_iron.flatten(), soft_iron, calibrated_magnetic_field


def _center_data(x: np.ndarray, sigma_sq_k: np.ndarray, sigma_sq_bar: float) -> Tuple[float, np.ndarray]:
    """
    Calculates  the  centered  and  center  components  of  a vector X.
    Based on the equations (13a), (13b) and (14) in (Alonso, R. Shuster, M.D.
    (2002a). TWOSTEP: A fast robust algorithm for attitude-independent
    magnetometer-bias determination. Journal of the Astronautical Sciences,
    50(4):433-452).

    Args:
        x (np.ndarray): Column vector of data as a (n, 1) numpy array.
        sigma_sq_k (np.ndarray): Column vector with the variance of each sample as
            a (n, 1) numpy array.
        sigma_sq_bar (float): Inverse of the sum of the reciprocal of the variances.

    Returns:
        x_bar (float): The centered data, i.e., the sum of the samples weighted by
            the reciprocal of the variance, multiplied by the inverse of the sum of
            the reciprocal of the variances.
        x_tilde (np.ndarray): The samples with the X_bar subtracted.
    """
    # Center   component
    x_bar = sigma_sq_bar * np.sum(x * (1.0 / sigma_sq_k), axis=0)
    # Centered  component
    x_tilde = x - np.tile(x_bar, (x.shape[0], 1))

    return x_bar, x_tilde


def _TS_fisher_centered(sigma_sq: np.ndarray, L_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the fisher information matrix for the centered estimate, when
    given variance sigma_sq and centered vectors of L_tilde, based on (5.28)
    in Diane, J.P. (2013). Magnetic test facility - sensor and coil calibrations.
    Master's thesis, University of Adelaide, School of Electrical and Electronic
    Engineering.

    Args:
        sigma_sq (np.ndarray): Variance based of each sample defined in equation (5b)
            as a (-1, 1) numpy array.
        L_tilde (np.ndarray): L metric for each sample defined in equation (30) as a
            (n, 9) numpy array.

    Returns:
        I_fisher_tilde (np.ndarray): The fisher information matrix as a (9, 9) numpy array.
        Lfishinv_tilde (np.ndarray): The inverse of the fisher information matrix as a
            (9, 9) numpy array.
    """
    # Compute  fisher  information  matrix and the inverse
    I_fisher_tilde = L_tilde.T @ (L_tilde * (1 / sigma_sq))
    Lfishinv_tilde = np.linalg.inv(I_fisher_tilde)

    return I_fisher_tilde, Lfishinv_tilde


def _theta_to_c_E(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts c and E elements from theta as equation (30) in Alonso, R. Shuster,
    M.D. (2002b). Complete linear attitude-independent magnetometer calibration.
    Journal of the Astronautical Science, 50(4):477-490.

    Args:
        theta (np.ndarray): Theta vector defined in equation (30) as a (9, 1) numpy array.

    Returns:
        c (np.ndarray): The vector c as a (3, 1) numpy array.
        E (np.ndarray): The matrix E as a (3, 3) numpy array.
    """
    c = theta[:3, :]
    e_matrix = np.array(
        [
            [theta[3, 0], theta[6, 0], theta[7, 0]],
            [theta[6, 0], theta[4, 0], theta[8, 0]],
            [theta[7, 0], theta[8, 0], theta[5, 0]],
        ]
    )
    return c, e_matrix


def _TS_fisher_center(sigma_sq_bar: float, L_bar: np.ndarray, dbsqdtheta_p: np.ndarray) -> np.ndarray:
    """
    Computes center information matrix based in (5.29) in Diane, J.P. (2013).
    Magnetic test facility - sensor and coil calibrations. Master's thesis,
    University of Adelaide, School of Electrical and Electronic Engineering.

    Args:
        sigma_sq_bar (float): Inverse of the sum of the reciprocal of the variances.
        L_bar (np.ndarray): The centered L data, i.e., the sum of the samples
            weighted by the reciprocal of the variance, multiplied by the inverse
            of the sum of the reciprocal, as a (9, ) numpy array.
        dbsqdtheta_p (np.ndarray): The differentiation of the norm of the magnetic
            field measurements squared with respect to the theta prime value as
            described in equation (40) as a (9, 1) numpy array.

    Returns:
        I_fisher_bar (np.ndarray): The center of the Fisher matrix as a (9, 9) numpy array.
    """
    I_fisher_bar = (
        (L_bar.reshape(-1, 1) - dbsqdtheta_p) @ ((L_bar.reshape(-1, 1) - dbsqdtheta_p).T)
    ) / sigma_sq_bar
    return I_fisher_bar


def _theta_to_b_D(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a value of theta to usable physical values as described in
    equation (31) of Alonso, R. Shuster, M.D. (2002b). Complete linear
    attitude-independent magnetometer calibration. Journal of the Astronautical
    Science, 50(4):477-490.

    Args:
        theta (np.ndarray): Theta vector as a (9, 1) numpy array.

    Returns:
        b (np.ndarray): The vector b as a (3, 1) numpy array.
        d_matrix (np.ndarray): The matrix D as a (3, 3) numpy array.
    """
    c, e_matrix = _theta_to_c_E(theta)
    s_matrix, u_matrix = np.linalg.eig(e_matrix)
    w_matrix = -np.eye(3) + np.sqrt(np.eye(3) + np.diag(s_matrix))
    d_matrix = u_matrix @ w_matrix @ u_matrix.T
    # Calculate b  using  the  inverse of (I+D)
    b = np.linalg.solve(np.eye(3) + d_matrix, c)
    return b, d_matrix
