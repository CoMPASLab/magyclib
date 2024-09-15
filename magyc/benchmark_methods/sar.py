"""
MAGYC - Benchmark Methods - SAR

This module contains the benchmark methods for magnetometer hard-iron calibration.

Functions:
    sar_ls: The linear least squares for sensor bias calibration.
    sar_kf: The Kalman filter for sensor bias calibration.
    sar_aid: The adaptive identification for sensor bias calibration.

Authors: Giancarlo Troni and Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
from typing import Tuple, Union

import navlib.math as nm
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import expm


def sar_ls(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
           time: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The linear least squares for sensor bias calibration is seeks to minimize the
    sum of squared residuals

    $$\\sum_{i=1}^{n} \\frac{1}{\\sigma_i^2} ||\\dot{x}_i + \\omega_i \\times (x_i - b)||^2$$

    Where $x(t)$ is the measured magnetic field, $\\dot{x(t)}$ is the measured
    magnetic field differentiated with respect to time, $\\omega(t)$ is the
    measured angular-rate in instrument coordinates, $b$ is the hard-iron, and
    $\\times$ is the standard cross product operator.

    This optimization problem can be solved in an analytical way. For further
    information refer to section IV.A in Troni, G. and Whitcomb, L. L. (2019).
    Field sensor bias calibration with angular-rate sensors: Theory and experimental
    evaluation with application to magnetometer calibration. IEEE/ASME Transactions
    on Mechatronics, 24(4):1698--1710.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time measurements in a 1D numpy array or list.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements

    Raises:
        TypeError: If the magnetic field, angular rate, or time are not numpy arrays or lists.
        ValueError: If the magnetic field, angular rate, or time are not 3xN or Nx3 numpy arrays.
        ValueError: If the magnetic field, angular rate, and time do not have the same number of samples.
    """
    # Check if the magnetic_field, angular_rate, and time are lists and convert them to numpy arrays
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)
    if isinstance(angular_rate, list):
        angular_rate = np.array(angular_rate)
    if isinstance(time, list):
        time = np.array(time)

    # Check if the magnetic_field, angular_rate, and time are numpy arrays
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field must be a numpy array or a list.")
    if not isinstance(angular_rate, np.ndarray):
        raise TypeError("The angular rate must be a numpy array or a list.")
    if not isinstance(time, np.ndarray):
        raise TypeError("The time must be a numpy array or a list.")

    # Check if the magnetic_field and angular_rate are 3xN or Nx3 numpy arrays
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field must be a 3xN or Nx3 numpy array.")
    if angular_rate.ndim != 2 or (angular_rate.shape[0] != 3 and angular_rate.shape[1] != 3):
        raise ValueError("The angular rate must be a 3xN or Nx3 numpy array.")

    # Check if the time is a 1D numpy array
    time = time.flatten()
    if time.ndim != 1:
        raise ValueError("The time must be a (n, ), (n, 1) or (1, n) numpy array.")

    # Force the magnetic_field and angular_rate to be Nx3 numpy arrays
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T
    if angular_rate.shape[0] == 3:
        angular_rate = angular_rate.T

    # Check if the magnetic_field, angular_rate, and time have the same number of samples
    if magnetic_field.shape[0] != angular_rate.shape[0] or magnetic_field.shape[0] != time.shape[0]:
        raise ValueError("The magnetic field, angular rate, and time must have the same number of samples.")

    # Compute the magnetic calibration
    # Get the data variance
    magnetic_field_variance = _get_sigma_noise(magnetic_field)
    sigma_i = np.linalg.norm(magnetic_field_variance).reshape(-1, 1, 1)

    # Compute the skew-symmetric matrix of the angular rate.
    skew_symmetric_angular_rate = np.apply_along_axis(nm.vec_to_so3, 1, angular_rate)

    # Compute the magnetic field derivative
    magnetic_field_derivative = np.diff(magnetic_field, axis=0) / np.diff(time).reshape(-1, 1)
    magnetic_field_derivative = np.concatenate([np.zeros((1, 3)), magnetic_field_derivative], axis=0)

    # Estimate b
    b1_inv = np.linalg.inv(np.einsum("ijk->jk", (skew_symmetric_angular_rate ** 2) * (1 / sigma_i)))

    yi = np.einsum(
        "ijk->ikj",
        np.cross(angular_rate.reshape(-1, 1, 3), magnetic_field.reshape(-1, 1, 3))
        + magnetic_field_derivative.reshape(-1, 1, 3),
    )
    b2 = np.einsum("ijk->jk", (skew_symmetric_angular_rate @ yi) * (1 / sigma_i))

    hard_iron = b1_inv @ b2

    # Calibrate magnetic field
    calibrated_magnetic_field = magnetic_field - hard_iron.flatten()

    return hard_iron.flatten(), calibrated_magnetic_field


def sar_kf(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
           time: Union[np.ndarray, list], gains: Tuple[float, float] = (1.0, 1.0),
           f_normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The Kalman filter for sensor bias calibration uses the system model with  a
    discretization of the continuous-time system the sensor bias estimation can
    be solved with a standard discrete-time Kalman filter implementation that
    does not require differentiation.

    $$\\dot{x}_i = -\\omega_i \\times (x_i - b)$$

    Where $x(t)$ is the measured magnetic field, $\\dot{x(t)}$ is the measured
    magnetic field differentiated with respect to time, $\\omega(t)$ is the
    measured angular-rate in instrument coordinates, and $b$ is the hard-iron.

    For further information refer to section IV.B in Troni, G. and Whitcomb, L. L.
    (2019). Field sensor bias calibration with angular-rate sensors: Theory and
    experimental evaluation with application to magnetometer calibration. IEEE/ASME
    Transactions on Mechatronics, 24(4):1698--1710.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time measurements in a 1D numpy array or list.
        gains (tuple): Kalman filter gains.
        f_normalize (bool): Whether the k2 gain should be scaled by and adaptive
            constant computed as the reciprocal of the norm of the gyroscope measurement
            for that step, by default False.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements
        calibrated_filteres_magnetic_field (numpy.ndarray): Calibrated and filtered magnetic field measurements

    Raises:
        TypeError: If the magnetic field, angular rate, or time are not numpy arrays or lists.
        ValueError: If the magnetic field, angular rate, or time are not 3xN or Nx3 numpy arrays.
        ValueError: If the magnetic field, angular rate, and time do not have the same number of samples.
        TypeError: If the gains are not a tuple of floats.
        TypeError: If the f_normalize is not a boolean.
    """
    # Check if the magnetic_field, angular_rate, and time are lists and convert them to numpy arrays
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)
    if isinstance(angular_rate, list):
        angular_rate = np.array(angular_rate)
    if isinstance(time, list):
        time = np.array(time)

    # Check if the magnetic_field, angular_rate, and time are numpy arrays
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field must be a numpy array or a list.")
    if not isinstance(angular_rate, np.ndarray):
        raise TypeError("The angular rate must be a numpy array or a list.")
    if not isinstance(time, np.ndarray):
        raise TypeError("The time must be a numpy array or a list.")

    # Check if the magnetic_field and angular_rate are 3xN or Nx3 numpy arrays
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field must be a 3xN or Nx3 numpy array.")
    if angular_rate.ndim != 2 or (angular_rate.shape[0] != 3 and angular_rate.shape[1] != 3):
        raise ValueError("The angular rate must be a 3xN or Nx3 numpy array.")

    # Check if the time is a 1D numpy array
    time = time.flatten()
    if time.ndim != 1:
        raise ValueError("The time must be a (n, ), (n, 1) or (1, n) numpy array.")

    # Force the magnetic_field and angular_rate to be Nx3 numpy arrays
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T
    if angular_rate.shape[0] == 3:
        angular_rate = angular_rate.T

    # Check if the magnetic_field, angular_rate, and time have the same number of samples
    if magnetic_field.shape[0] != angular_rate.shape[0] or magnetic_field.shape[0] != time.shape[0]:
        raise ValueError("The magnetic field, angular rate, and time must have the same number of samples.")

    # Check that the gains are a tuple of floats
    if not isinstance(gains, tuple):
        raise TypeError("The gains must be a tuple of floats.")
    if not all(isinstance(gain, float) for gain in gains):
        raise TypeError("The gains must be a tuple of floats.")

    # Check that the f_normalize is a boolean
    if not isinstance(f_normalize, bool):
        raise TypeError("The f_normalize must be a boolean.")

    # Compute the magnetic calibration
    # Initial parameters
    b0 = np.zeros((3, ))
    k1a = gains[0]
    k1b = gains[1] if len(gains) >= 2 else gains[0]
    k2 = gains[2] if len(gains) >= 3 else gains[1]
    mf = magnetic_field.reshape(3, -1)
    w = angular_rate.reshape(3, -1)
    dt = np.diff(time)
    dt_vec = np.concatenate([np.array([dt[0]]), dt])

    # Kalman Model
    Bc = np.zeros([6, 1])
    # Measurement model
    H1 = np.hstack([np.eye(3), np.zeros([3, 3])])
    # Process noise covariance
    Qc = np.diag([k1a, k1a, k1a, k1b, k1b, k1b])
    # Variance in the measurements
    R = np.diag([k2, k2, k2])

    # KF
    F1 = _kf_transition_matrix([0, 0, 0])
    n = F1.shape[0]
    m = F1.shape[1]
    MM = np.zeros([n, mf.shape[1]])
    PP = np.zeros([n, m, mf.shape[1]])
    AA = np.zeros([n, m, mf.shape[1]])
    QQ = np.zeros([n, m, mf.shape[1]])
    KK = np.zeros([n, H1.shape[0], mf.shape[1]])

    # Initial guesses for the state mean and covariance.
    x = np.hstack([mf[:, 0], b0])
    p01 = 0.001  # P0 gyro
    p02 = 0.001  # P0 bias
    P0 = np.diag([p01, p01, p01, p02, p02, p02])
    P = P0

    # Filtering steps.
    for ix in range(mf.shape[1]):
        # Discretization of the continous-time system (dtk)
        dtk = dt_vec[ix]
        u = w[:, ix]

        [Ak, Bk, Qk] = _kf_lti_discretize(_kf_transition_matrix(u), Bc, Qc, dtk)

        AA[:, :, ix] = Ak
        QQ[:, :, ix] = Qk

        # Prediction
        [x, P] = _kf_predict(x, P, Ak, Qk)
        [x, P, K, dy, S] = _kf_update(x, P, mf[:, ix], H1, R)

        MM[:, ix] = x
        PP[:, :, ix] = P
        KK[:, :, ix] = K

    # Final Bias averaging last 20%
    hard_iron = np.mean(MM[3:, -int(np.round(mf.shape[1]*0.2)):], axis=1, keepdims=True)

    # Calibrate magnetic field
    calibrated_magnetic_field = magnetic_field - hard_iron.flatten()
    calibrated_filteres_magnetic_field = MM[:3, :].T

    return hard_iron.flatten(), calibrated_magnetic_field, calibrated_filteres_magnetic_field


def sar_aid(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
            time: Union[np.ndarray, list], gains: Tuple[float, float] = (1.0, 1.0),
            f_normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The adaptive identification for sensor bias calibration proposes that the
    unknown sensor bias, $b$, can be estimated on-line with a novel adaptive
    identification algorithm. The possible advantages of this adaptive approach
    are that (i) it does not require numerical differentiation of the sensor
    measurement $x(t)$, (ii) it is less computationally expensive than the SAR-KF,
    and (iii) it could be combined with other nonlinear observer methods.

    For further information refer to section IV.C in Troni, G. and Whitcomb, L. L.
    (2019). Field sensor bias calibration with angular-rate sensors: Theory and
    experimental evaluation with application to magnetometer calibration. IEEE/ASME
    Transactions on Mechatronics, 24(4):1698--1710.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time measurements in a 1D numpy array or list.
        gains (tuple): Gains defined in the set of equations (5) of the proposed method as
            a tuple of floats, by default (1.0, 1.0)
        f_normalize (bool): Whether the k2 gain should be scaled by and adaptive
            constant computed as the reciprocal of the norm of the gyroscope measurement
            for that step, by default False.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements
        calibrated_filtered_magnetic_field (numpy.ndarray): Calibrated and filtered magnetic field measurements

    Raises:
        TypeError: If the magnetic field, angular rate, or time are not numpy arrays or lists.
        ValueError: If the magnetic field, angular rate, or time are not 3xN or Nx3 numpy arrays.
        ValueError: If the magnetic field, angular rate, and time do not have the same number of samples.
        TypeError: If the gains are not a tuple of floats.
        TypeError: If the f_normalize is not a boolean.
    """
    # Check if the magnetic_field, angular_rate, and time are lists and convert them to numpy arrays
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)
    if isinstance(angular_rate, list):
        angular_rate = np.array(angular_rate)
    if isinstance(time, list):
        time = np.array(time)

    # Check if the magnetic_field, angular_rate, and time are numpy arrays
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field must be a numpy array or a list.")
    if not isinstance(angular_rate, np.ndarray):
        raise TypeError("The angular rate must be a numpy array or a list.")
    if not isinstance(time, np.ndarray):
        raise TypeError("The time must be a numpy array or a list.")

    # Check if the magnetic_field and angular_rate are 3xN or Nx3 numpy arrays
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field must be a 3xN or Nx3 numpy array.")
    if angular_rate.ndim != 2 or (angular_rate.shape[0] != 3 and angular_rate.shape[1] != 3):
        raise ValueError("The angular rate must be a 3xN or Nx3 numpy array.")

    # Check if the time is a 1D numpy array
    time = time.flatten()
    if time.ndim != 1:
        raise ValueError("The time must be a (n, ), (n, 1) or (1, n) numpy array.")

    # Force the magnetic_field and angular_rate to be Nx3 numpy arrays
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T
    if angular_rate.shape[0] == 3:
        angular_rate = angular_rate.T

    # Check if the magnetic_field, angular_rate, and time have the same number of samples
    if magnetic_field.shape[0] != angular_rate.shape[0] or magnetic_field.shape[0] != time.shape[0]:
        raise ValueError("The magnetic field, angular rate, and time must have the same number of samples.")

    # Check that the gains are a tuple of floats
    if not isinstance(gains, tuple):
        raise TypeError("The gains must be a tuple of floats.")
    if not all(isinstance(gain, float) for gain in gains):
        raise TypeError("The gains must be a tuple of floats.")

    # Check that the f_normalize is a boolean
    if not isinstance(f_normalize, bool):
        raise TypeError("The f_normalize must be a boolean.")

    # Compute the magnetic calibration
    # Initial parameters
    b0 = np.zeros((3, ))
    k1 = gains[0]
    k2 = gains[1]
    mf = magnetic_field.reshape(3, -1)
    w = angular_rate.reshape(3, -1)
    dt = np.diff(time)
    dt_vec = np.concatenate([np.array([dt[0]]), dt])

    # Compute the skew-symmetric matrix of the angular rate.
    skew_symmetric_angular_rate = np.apply_along_axis(nm.vec_to_so3, 1, angular_rate)

    # Adaptive ID system
    mh = np.zeros((3, mf.shape[1] + 1))
    mhd = np.zeros((3, mf.shape[1]))
    bh = np.zeros((3, mf.shape[1] + 1))
    bhd = np.zeros((3, mf.shape[1]))
    mh[:, 0] = magnetic_field[0, :]
    bh[:, 0] = b0

    for ix in range(mf.shape[1]):
        mhd[:, ix] = (
            - skew_symmetric_angular_rate[ix, :, :] @ mh[:, ix]
            + skew_symmetric_angular_rate[ix, :, :] @ bh[:, ix]
            - k1 * (mh[:, ix] - mf[:, ix])
        )

        if (np.linalg.norm(w[:, ix]) > 0.01) and f_normalize:
            k_adap = 1 / np.linalg.norm(w[:, ix])
            bhd[:, ix] = -k_adap * k2 * skew_symmetric_angular_rate[ix, :, :] @ (mh[:, ix] - mf[:, ix])
        else:
            bhd[:, ix] = -k2 * skew_symmetric_angular_rate[ix, :, :].T @ (mh[:, ix] - mf[:, ix])

        mh[:, ix + 1] = mh[:, ix] + dt_vec[ix] * mhd[:, ix]
        bh[:, ix + 1] = bh[:, ix] + dt_vec[ix] * bhd[:, ix]

    # Final Bias averaging last 20%
    hard_iron = np.mean(bh[:, -int(np.round(mf.shape[1] * 0.2)):], axis=1, keepdims=True)

    # Calibrate magnetic field
    calibrated_magnetic_field = magnetic_field - hard_iron.flatten()
    calibrated_filtered_magnetic_field = mh[:, :-1].T

    return hard_iron.flatten(), calibrated_magnetic_field, calibrated_filtered_magnetic_field


def _get_sigma_noise(mat: np.ndarray) -> np.ndarray:
    """
    Gets a nxm array and returns an array of the same size where
    each row is the variance of each axis. The assumption is that the variance is
    the same for all samples, then all the rows are equals.

    Args:
        mat (np.ndarray): Data matrix as a (n, m) numpy array.

    Returns:
        np.ndarray: Data matrix where each column is the axis variance as a numpy array.
    """
    # Sensor Measurement
    mat_copy = np.copy(mat)

    # Compute data trend
    mat_center = savgol_filter(mat_copy, 25, 2, axis=0)

    # Remove the data trend to have a zero-mean data
    mat_centered = mat - mat_center
    var = np.var(mat_centered, axis=0, keepdims=True)
    sigma = np.tile(var, (mat.shape[0], 1))
    return sigma


def _kf_lti_discretize(Ac: np.ndarray, Bc: np.ndarray = None, Qc: np.ndarray = None,
                       dt: float = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretize a Linear Time-Invariant (LTI) system using the matrix fraction decomposition
    for use in a discrete-time Kalman filter.

    Args:
        Ac (np.ndarray): Continuos state transition matrix.
        Bc (np.ndarray): Continuos input matrix, by default None.
        Qc (np.ndarray): Continuos covariance matrix, by default None.
        dt (float): Time step, by default 1.

    Returns:
        np.ndarray: Discrete state transition matrix.
        np.ndarray: Discrete input matrix.
        np.ndarray: Discrete covariance matrix.
    """
    # Check the number of states
    n = Ac.shape[0]

    # Default to zero non provided matrices
    if Bc is None:
        Bc = np.zeros([n, 1])

    if Qc is None:
        Qc = np.zeros([n, n])

    # Discretize state transition and input matrix (close form)
    # Ad = expm(Ac*dt)
    M = np.vstack([np.hstack([Ac, Bc]), np.zeros([1, n+1])])
    ME = expm(M*dt)

    # Discretize state transition and input matrix
    Ad = ME[:n, :n]
    Bd = ME[:n, n:]

    # Discretize Covariance: by (Van Loan, 1978)
    F = np.vstack([np.hstack([-Ac, Qc]), np.hstack([np.zeros([n, n]), Ac.T])])
    G = expm(F*dt)
    Qd = np.dot(G[n:, n:].T, G[:n, n:])

    # # Discretize Covariance: by matrix fraction decomposition
    # Phi = vstack([hstack([Ac,            Qc]),
    #               hstack([np.zeros([n,n]),-Ac.T])])
    # AB  = np.dot (scipy.linalg.expm(Phi*dt), vstack([np.zeros([n,n]),np.eye(n)]))
    # Qd  = np.linalg.solve(AB[:n,:].T, AB[n:2*n,:].T).T

    return Ad, Bd, Qd


def _kf_predict(x: np.ndarray, P: np.ndarray, A: np.ndarray = None, Q: np.ndarray = None,
                B: np.ndarray = None, u: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prediction step of the Kalman filter.

    Args:
        x (np.ndarray): State mean.
        P (np.ndarray): State covariance.
        A (np.ndarray): State transition matrix, by default None.
        Q (np.ndarray): Process noise covariance, by default None.
        B (np.ndarray): Input matrix, by default None.
        u (np.ndarray): Input vector, by default None.

    Returns:
        np.ndarray: Updated state mean.
        np.ndarray: Updated state covariance.
    """

    # Check Arguments
    n = A.shape[0]

    # Default state transition matrix to the identity matrix if not provided
    if A is None:
        A = np.eye(n)

    # Default process noise covariance to zero matrix if not provided
    if Q is None:
        Q = np.zeros([n, 1])

    # Default input matrix to the identity matrix if not provided
    if (B is None) and (u is not None):
        B = np.eye([n, u.shape(u)[0]])

    # Prediction step
    # State
    if u is None:
        x = np.dot(A, x)
    else:
        x = np.dot(A, x) + np.dot(B, u)

    # Covariance
    P = np.dot(np.dot(A, P), A.T) + Q

    return x, P


def _kf_update(x: np.ndarray, P: np.ndarray, y: np.ndarray, H: np.ndarray,
               R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Update step of the Kalman filter.

    Args:
        x (np.ndarray): State mean.
        P (np.ndarray): State covariance.
        y (np.ndarray): Measurement.
        H (np.ndarray): Measurement matrix.
        R (np.ndarray): Measurement noise covariance.
    """
    # Compute measurement residual
    dy = y - np.dot(H, x)
    # Compute covariance residual
    S = R + np.dot(np.dot(H, P), H.T)
    # Compute Kalman Gain
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

    # Update state estimate
    dy = dy.flatten()
    x = x + np.dot(K, dy)
    P = P - np.dot(np.dot(K, H), P)

    return x, P, K, dy, S


def _kf_transition_matrix(angular_rate: np.ndarray) -> np.ndarray:
    """
    Compute the transition matrix for the Kalman filter.
    """
    angular_rate = angular_rate.flatten() if isinstance(angular_rate, np.ndarray) else angular_rate
    skew_symmetric_angular_rate = nm.vec_to_so3(angular_rate)
    a_matrix = np.zeros((6, 6))
    a_matrix[:3, :] = np.hstack([-skew_symmetric_angular_rate, skew_symmetric_angular_rate])
    return a_matrix
