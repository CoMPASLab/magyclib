"""
MAGYC - Proposed Methdods

This module contains the proposed methods for magnetometer and gyroscope calibration.

Functions:
    magyc_ls: MAGYC-LS method.
    magyc_nls: MAGYC-NLS method.
    magyc_bfg: MAGYC-BFG method.
    magyc_ifg: MAGYC-IFG method.

Authors: Sebastián Rodríguez-Martínez and Giancarlo Troni
Contact: srodriguez@mbari.org
"""
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import gtsam
import navlib.math as nm
import numpy as np
import scipy
from gtsam.symbol_shorthand import B, S, W
from numpy import exp as exp


def magyc_ls(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
             time: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MAGYC-LS: Proposed method for the full calibration of a three-axis magnetometer
    using magnetic field and angular rate measurements. This particular approach
    is based on a least squares optimization and poses the probems as a linear
    least squares optimization problem.

    Even tough a closed solution can be computed, it is a ill-conditioned problem
    and the optimization is preferred.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time stamps of the measurements.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.

    Raises:
        TypeError: If the magnetic field, angular rate, and time are not numpy arrays or lists.
        ValueError: If the magnetic field and angular rate are not 3xN or Nx3 numpy
            arrays, or if the time is not a 1D numpy array.
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

    # Compute the skew symmetric matrix of the angular rate
    skew_symmetric_angular_rate = np.apply_along_axis(nm.vec_to_so3, 1, angular_rate)

    # Compute the magnetic field derivative
    magnetic_field_derivative = np.diff(magnetic_field, axis=0) / np.diff(time).reshape(-1, 1)
    magnetic_field_derivative = np.concatenate([np.zeros((1, 3)), magnetic_field_derivative], axis=0).reshape(-1, 3, 1)

    # Reshape magnetic field
    magnetic_field_3d = magnetic_field.reshape(-1, 3, 1)

    # Compute the magnetic calibration
    # Least Squares Initial Guess and Constraints
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Optimization
    res = scipy.optimize.least_squares(
        _magyc_ls_cost_function,
        x0,
        jac="3-point",
        method="trf",
        verbose=0,
        loss="linear",
        max_nfev=1000,
        ftol=1.00e-06,
        gtol=None,
        xtol=None,
        x_scale="jac",
        args=(magnetic_field_3d, magnetic_field_derivative, skew_symmetric_angular_rate)
    )

    # Compute SI and scale it so the deformed sphere keeps the volume.
    x = res["x"]
    soft_iron = np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]])
    eigvals = np.linalg.eigvals(soft_iron)
    scaled_SI = (np.prod(eigvals) ** (-1 / 3)) * soft_iron

    # Compute HI, Wb (optional)
    hard_iron = scaled_SI @ x[6:9].reshape(3, 1)
    soft_iron = scaled_SI

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    return hard_iron.flatten(), soft_iron, calibrated_magnetic_field


def magyc_nls(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
              time: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MAGYC-NLS: Proposed method for the full calibration of a three-axis magnetometer
    and a three-axis gyroscope using magnetic field and angular rate measurements.
    This particular approach is based on a least squares optimization and poses
    the probems as a non-linear least squares optimization problem.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time stamps of the measurements.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        gyro_bias (numpy.ndarray): Gyroscope bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.
        calibrated_angular_rate (numpy.ndarray): Calibrated angular rate measurements.

    Raises:
        TypeError: If the magnetic field, angular rate, and time are not numpy arrays or lists.
        ValueError: If the magnetic field and angular rate are not 3xN or Nx3 numpy
            arrays, or if the time is not a 1D numpy array.
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

    # Compute the skew symmetric matrix of the angular rate
    skew_symmetric_angular_rate = np.apply_along_axis(nm.vec_to_so3, 1, angular_rate)

    # Compute the magnetic field derivative
    magnetic_field_derivative = np.diff(magnetic_field, axis=0) / np.diff(time).reshape(-1, 1)
    magnetic_field_derivative = np.concatenate([np.zeros((1, 3)), magnetic_field_derivative], axis=0).reshape(-1, 3, 1)

    # Reshape magnetic field
    magnetic_field_3d = magnetic_field.reshape(-1, 3, 1)

    # Compute the magnetic calibration
    # Least Squares Initial Guess and Constraints
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Optimization
    res = scipy.optimize.least_squares(
        _magyc_nls_cost_function,
        x0,
        jac="3-point",
        method="trf",
        verbose=0,
        loss="linear",
        max_nfev=1000,
        ftol=1.00e-06,
        gtol=None,
        xtol=None,
        x_scale="jac",
        args=(magnetic_field_3d, magnetic_field_derivative, skew_symmetric_angular_rate)
    )

    # Compute SI and scale it so the deformed sphere keeps the volume.
    x = res["x"]
    soft_iron = np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]])
    eigvals = np.linalg.eigvals(soft_iron)
    scaled_SI = (np.prod(eigvals) ** (-1 / 3)) * soft_iron

    # Compute HI, Wb (optional)
    hard_iron = scaled_SI @ x[6:9].reshape(3, 1)
    soft_iron = scaled_SI
    gyro_bias = x[9:].reshape(3, 1)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    # Calibrated gyroscope measurements
    calibrated_angular_rate = angular_rate - gyro_bias.flatten()

    return hard_iron.flatten(), soft_iron, gyro_bias.flatten(), calibrated_magnetic_field, calibrated_angular_rate


def magyc_bfg(magnetic_field: Union[np.ndarray, list], angular_rate: Union[np.ndarray, list],
              time: Union[np.ndarray, list], measurements_window: int = 25, optimizer: str = "dogleg",
              relative_error_tol: float = 1.00e-07, absolute_error_tol: float = 1.00e-07,
              max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                             Dict[str, Union[List[float], int]]]:
    """
    MAGYC-BFG: Proposed method for the full calibration of a three-axis magnetometer
    using magnetic field and angular rate measurements. This particular approach
    is based on a factor graph processing all the data in a batch manner.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time stamps of the measurements.
        measurements_window (int): Window size for the measurements.
        optimizer (str): Optimization algorithm to use. Options are "dogleg" or "lm"
            for the Dogleg and Levenberg-Marquardt optimizers respectively.
        relative_error_tol (float): Relative error tolerance for the optimizer. Default is 1.00e-07
        absolute_error_tol (float): Absolute error tolerance for the optimizer. Default is 1.00e-07
        max_iter (int): Maximum number of iterations for the optimizer. Default is 1000

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        gyro_bias (numpy.ndarray): Gyroscope bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.
        calibrated_angular_rate (numpy.ndarray): Calibrated angular rate measurements.
        optimization_status (Dict[str, Union[List[float], int]]): Dictionary with
            the optimization status. The keys are "error" and "iterations".

    Raises:
        TypeError: If the magnetic field, angular rate, and time are not numpy arrays or lists.
        ValueError: If the magnetic field and angular rate are not 3xN or Nx3 numpy
            arrays, or if the time is not a 1D numpy array.
        ValueError: If the optimizer is not a string or not "dogleg" or "lm"
        TypeError: If the relative error tolerance is not a float
        TypeError: If the absolute error tolerance is not a float
        ValueError: If the maximum number of iterations is not a positive integer
        ValueError: If the measurements window is not a positive integer
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

    # Check that the optimizer is a string and is either "dogleg" or "lm"
    if not isinstance(optimizer, str) or optimizer not in ["dogleg", "lm"]:
        raise ValueError("The optimizer must be a string and either 'dogleg' or 'lm'.")

    # Check that the relative error tolerance is a float
    if not isinstance(relative_error_tol, float) or relative_error_tol <= 0:
        raise TypeError("The relative error tolerance must be a float.")

    # Check that the absolute error tolerance is a float
    if not isinstance(absolute_error_tol, float) or absolute_error_tol <= 0:
        raise TypeError("The absolute error tolerance must be a float.")

    # Check that the maximum number of iterations is a positive integer
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("The maximum number of iterations must be a positive integer.")

    # Check that the measurements window is a positive integer
    if not isinstance(measurements_window, int) or measurements_window <= 0:
        raise ValueError("The measurements window must be a positive integer.")

    # Compute the magnetic field derivative
    magnetic_field_derivative = np.diff(magnetic_field, axis=0) / np.diff(time).reshape(-1, 1)
    magnetic_field_derivative = np.concatenate([np.zeros((1, 3)), magnetic_field_derivative], axis=0)

    # Compute the magnetic calibration
    # Smoothing and Mapping Factor Graph
    # 1. Create the non-linear graph
    graph = gtsam.NonlinearFactorGraph()

    # 2. noise model for each factor.
    residual_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)
    volume_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)
    difference_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)

    # 3. Creates values structure with initial values
    initial = gtsam.Values()
    initial.insert(S(0), np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
    initial.insert(B(0), gtsam.Point3(0, 0, 0))
    initial.insert(W(0), gtsam.Point3(0, 0, 0))
    keys = [S(0), B(0), W(0)]

    # 4. Add factor for each measurement accumulates in the measurements window into a single node
    measurements_window = int(measurements_window)
    m_dot_window = np.empty((measurements_window, 3))
    m_window = np.empty((measurements_window, 3))
    w_window = np.empty((measurements_window, 3))

    # 5. Add factors to the graph
    for i in range(magnetic_field.shape[0]):
        # Get sensor measurements and estimated magnetic field derivative
        m_dot_window[i % measurements_window, :] = magnetic_field_derivative[i, :]
        m_window[i % measurements_window, :] = magnetic_field[i, :]
        w_window[i % measurements_window, :] = angular_rate[i, :]

        if (i % measurements_window == 0 and i != 0):
            # Average measurements by the measurements window size.
            m_dot_meadian = np.median(m_dot_window, axis=0).reshape(3, 1)
            m_median = np.median(m_window, axis=0).reshape(3, 1)
            w_median = np.median(w_window, axis=0).reshape(3, 1)

            # 5.1 Residual factor
            rf = gtsam.CustomFactor(residual_noise, keys, partial(_residual_factor, m_dot_meadian, m_median, w_median))
            graph.push_back(rf)

            # 5.2 Add constraint over the solution to avoid the trivial zero solution with a volume factor
            vf = gtsam.CustomFactor(volume_noise, [S(0)], _volume_factor)
            graph.push_back(vf)

            # 5.3 Avoid the diagonal terms of the soft-iron to diverge keeping a constrained difference between them
            df = gtsam.CustomFactor(difference_noise, [S(0)], _difference_factor)
            graph.push_back(df)

            # 5.3 Reset the measurements window
            m_dot_window = np.empty((measurements_window, 3))
            m_window = np.empty((measurements_window, 3))
            w_window = np.empty((measurements_window, 3))

    # 6. Optimize the graph
    # 6.1 Create optimizer parameters
    params = gtsam.DoglegParams() if optimizer == "dogleg" else gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(relative_error_tol)
    params.setAbsoluteErrorTol(absolute_error_tol)
    params.setMaxIterations(max_iter)
    params.setLinearSolverType("MULTIFRONTAL_CHOLESKY")

    # 6.2 Create optimizer
    if optimizer == "dogleg":
        optimizer = gtsam.DoglegOptimizer(graph, initial, params)
    else:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    # 6.3 Optimize
    result, optimization_status = _gtsam_optimize(optimizer, params)

    # 7. Process Results
    l_params = result.atVector(S(0))
    b = result.atVector(B(0))
    d = result.atVector(W(0))

    lower_triangular_matrix = np.array([[exp(l_params[0]), 0, 0],
                                        [l_params[1], exp(l_params[2]), 0],
                                        [l_params[3], l_params[4], exp(l_params[5])]])
    soft_iron = np.linalg.inv(lower_triangular_matrix @ lower_triangular_matrix.T)
    hard_iron = soft_iron @ np.vstack(b)
    gyro_bias = np.vstack(d)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    # Calibrated gyroscope measurements
    calibrated_angular_rate = angular_rate - gyro_bias.flatten()

    return (hard_iron.flatten(), soft_iron, gyro_bias.flatten(), calibrated_magnetic_field, calibrated_angular_rate,
            optimization_status)


def magyc_ifg(
        magnetic_field: Union[np.ndarray, list],
        angular_rate: Union[np.ndarray, list],
        time: Union[np.ndarray, list],
        measurements_window: int = 25
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    MAGYC-IFG: Proposed method for the full calibration of a three-axis magnetometer
    using magnetic field and angular rate measurements. This particular approach
    is based on a factor graph processing all the data in an incremental manner.

    Args:
        magnetic_field (numpy.ndarray or list): Magnetic field measurements in a
            3xN or Nx3 numpy array or list.
        angular_rate (numpy.ndarray or list): Angular rate measurements in a 3xN or
            Nx3 numpy array or list.
        time (numpy.ndarray or list): Time stamps of the measurements.
        measurements_window (int): Window size for the measurements.

    Returns:
        hard_iron (numpy.ndarray): Hard iron bias.
        soft_iron (numpy.ndarray): Soft iron matrix.
        gyro_bias (numpy.ndarray): Gyroscope bias.
        calibrated_magnetic_field (numpy.ndarray): Calibrated magnetic field measurements.
        calibrated_angular_rate (numpy.ndarray): Calibrated angular rate measurements.
        optimization_status (Dict[str, np.ndarray]): Dictionary with the SI, HI
            and Wb for each iteartions. The keys are: "soft_iron", "hard_iron",
            "gyro_bias".

    Raises:
        TypeError: If the magnetic field, angular rate, and time are not numpy arrays or lists.
        ValueError: If the magnetic field and angular rate are not 3xN or Nx3 numpy
            arrays, or if the time is not a 1D numpy array.
        ValueError: If the measurements window is not a positive integer
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

    # Check that the measurements window is a positive integer
    if not isinstance(measurements_window, int) or measurements_window <= 0:
        raise ValueError("The measurements window must be a positive integer.")

    # Compute the magnetic calibration
    # Smoothing and Mapping Factor Graph
    # 1. Create the non-linear graph
    graph = gtsam.NonlinearFactorGraph()

    # 2. Set iSAM2 parameters and create iSAM2 object
    isam_parameters = gtsam.ISAM2Params()
    isam_parameters.setOptimizationParams(gtsam.ISAM2DoglegParams())
    isam = gtsam.ISAM2(isam_parameters)

    # 3. noise model for each factor.
    residual_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)
    volume_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)
    difference_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)

    # 4. Creates values structure with initial values
    initial = gtsam.Values()
    initial.insert(S(0), np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
    initial.insert(B(0), gtsam.Point3(0, 0, 0))
    initial.insert(W(0), gtsam.Point3(0, 0, 0))
    keys = [S(0), B(0), W(0)]

    # Dictionary to save the progress of parameters during optimization
    optimization_status = {"S": [], "B": [], "W": []}

    # 5. Add factor for each measurement accumulates in the measurements window into a single node
    measurements_window = int(measurements_window)
    m_window = np.empty((measurements_window, 3))
    w_window = np.empty((measurements_window, 3))
    t_window = np.empty((measurements_window, ))

    # 6. Add factors to the graph
    for i in range(magnetic_field.shape[0]):
        # Get sensor measurements and estimated magnetic field derivative
        t_window[i % measurements_window] = time[i]
        m_window[i % measurements_window, :] = magnetic_field[i, :]
        w_window[i % measurements_window, :] = angular_rate[i, :]

        if (i % measurements_window == 0 and i != 0):
            # Compute the derivative of the magnetic field for the window
            m_dot_window = np.diff(m_window, axis=0) / np.diff(t_window).reshape(-1, 1)

            # Average measurements by the measurements window size.
            m_dot_meadian = np.median(m_dot_window, axis=0).reshape(3, 1)
            m_median = np.median(m_window, axis=0).reshape(3, 1)
            w_median = np.median(w_window, axis=0).reshape(3, 1)

            # 6.1 Residual factor
            rf = gtsam.CustomFactor(residual_noise, keys, partial(_residual_factor, m_dot_meadian, m_median, w_median))
            graph.push_back(rf)

            # 6.2 Add constraint over the solution to avoid the trivial zero solution
            nf = gtsam.CustomFactor(volume_noise, [S(0)], _volume_factor)
            graph.push_back(nf)

            # 6.3 Avoid the diagonal terms of the soft-iron to diverge keeping a constrained difference between them
            df = gtsam.CustomFactor(difference_noise, [S(0)], _difference_factor)
            graph.push_back(df)

            # 6.4 Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
            # Set iterations to start optimization, otherwise the optimizations starts as a ill-posed problem.
            if (i // measurements_window) % 10 == 0:
                isam.update(graph, initial)
                current = isam.calculateEstimate()

                # Save the current parameters
                for key, variable in zip([S(0), B(0), W(0)], "SBW"):
                    vector = current.atVector(key).reshape(1, -1)
                    optimization_status[variable].append(vector)

                graph = gtsam.NonlinearFactorGraph()
                initial = gtsam.Values()

            # 6.5 Reset the measurements window
            t_window = np.empty((measurements_window, ))
            m_window = np.empty((measurements_window, 3))
            w_window = np.empty((measurements_window, 3))

    # 7. Process Results
    l_params = current.atVector(S(0))
    b = current.atVector(B(0))
    d = current.atVector(W(0))

    lower_triangular_matrix = np.array([[exp(l_params[0]), 0, 0],
                                        [l_params[1], exp(l_params[2]), 0],
                                        [l_params[3], l_params[4], exp(l_params[5])]])
    soft_iron = np.linalg.inv(lower_triangular_matrix @ lower_triangular_matrix.T)
    hard_iron = soft_iron @ np.vstack(b)
    gyro_bias = np.vstack(d)

    # Calibrate magnetic field
    calibrated_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.reshape(3, -1) - hard_iron.reshape(3, 1))).T

    # Calibrated gyroscope measurements
    calibrated_angular_rate = angular_rate - gyro_bias.flatten()

    # Update optimization status to have the actual matrices instead of the keys
    optimization_steps = len(optimization_status["S"])
    optimization_status_final = {
        "soft_iron": np.empty((optimization_steps, 9)),
        "hard_iron": np.empty((optimization_steps, 3)),
        "gyro_bias": np.empty((optimization_steps, 3))
        }

    for i in range(optimization_steps):
        # Get parameters
        l_params = optimization_status["S"][i].flatten()
        b = optimization_status["B"][i]
        d = optimization_status["W"][i]

        # Compute soft-iron, hard-iron and gyroscope bias
        lower_triangular_matrix = np.array([[exp(l_params[0]), 0, 0],
                                            [l_params[1], exp(l_params[2]), 0],
                                            [l_params[3], l_params[4], exp(l_params[5])]])
        soft_iron_i = np.linalg.inv(lower_triangular_matrix @ lower_triangular_matrix.T)
        hard_iron_i = soft_iron_i @ b.reshape(3, 1)
        gyro_bias_i = d.reshape(3, 1)

        # Fill the new optimization status dictionary
        optimization_status_final["soft_iron"][i, :] = soft_iron_i.flatten()
        optimization_status_final["hard_iron"][i, :] = hard_iron_i.flatten()
        optimization_status_final["gyro_bias"][i, :] = gyro_bias_i.flatten()

    return (hard_iron.flatten(), soft_iron, gyro_bias.flatten(), calibrated_magnetic_field, calibrated_angular_rate,
            optimization_status_final)


def _magyc_ls_cost_function(x: np.ndarray, magnetic_field: np.ndarray, magnetic_field_derivative: np.ndarray,
                            skew_symmetric_angular_rate: np.ndarray) -> np.ndarray:
    """
    Function which computes the vector of residuals, and the minimization
    proceeds with respect to x for the MAGYC-LS method.

    Args:
        x (np.ndarray): Optimization variables as a (9, ) numpy array.
        magnetic_field (np.ndarray): Magnetic field as a (n, 3) numpy array.
        magnetic_field_derivative (np.ndarray): Magnetic field derivative as a
            (n, 3) numpy array.
        skew_symmetric_angular_rate (np.ndarray): Skew symmetric matrix of the
            angular rate as a (n, 3, 3) numpy array.

    Returns:
        ssr (np.ndarray): Vector of residuals for the optimization as a (9, ) numpy array.
    """
    # Compute SI and scale it so the deformed sphere keeps the volume.
    soft_iron = np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]])
    eigvals = np.linalg.eigvals(soft_iron)
    scaled_soft_iron = (np.prod(eigvals) ** (-1 / 3)) * soft_iron

    # Compute C, HI and Wb (optional)
    scaled_soft_iron_inv = np.linalg.inv(scaled_soft_iron)
    mb = x[6:9].reshape(3, 1)

    # Tile HI and SI and Wb to (n, i, j) arrays
    scaled_soft_iron_inv_i = np.tile(scaled_soft_iron_inv, (skew_symmetric_angular_rate.shape[0], 1, 1))
    mb_i = np.tile(mb, (skew_symmetric_angular_rate.shape[0], 1, 1))

    # Cost function
    sensor_model = (
        (scaled_soft_iron_inv_i @ magnetic_field_derivative)
        + (skew_symmetric_angular_rate @ scaled_soft_iron_inv_i @ magnetic_field)
        - (skew_symmetric_angular_rate @ mb_i)
    )

    # Error norm
    error = (np.linalg.norm(sensor_model.reshape(-1, 3), axis=1)).flatten()
    return error


def _magyc_nls_cost_function(x: np.ndarray, magnetic_field: np.ndarray, magnetic_field_derivative: np.ndarray,
                             skew_symmetric_angular_rate: np.ndarray) -> np.ndarray:
    """
    Function which computes the vector of residuals, and the minimization
    proceeds with respect to x for the MAGYC-NLS method.

    Args:
        x (np.ndarray): Optimization variables as a (9, ) numpy array.
        magnetic_field (np.ndarray): Magnetic field as a (n, 3) numpy array.
        magnetic_field_derivative (np.ndarray): Magnetic field derivative as a
            (n, 3) numpy array.
        skew_symmetric_angular_rate (np.ndarray): Skew symmetric matrix of the
            angular rate as a (n, 3, 3) numpy array.

    Returns:
        ssr (np.ndarray): Vector of residuals for the optimization as a (9, ) numpy array.
    """
    # Compute SI and scale it so the deformed sphere keeps the volume.
    soft_iron = np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]])
    eigvals = np.linalg.eigvals(soft_iron)
    scaled_soft_iron = (np.prod(eigvals) ** (-1 / 3)) * soft_iron

    # Compute C, HI and Wb (optional)
    scaled_soft_iron_inv = np.linalg.inv(scaled_soft_iron)
    mb = x[6:9].reshape(3, 1)
    wb = x[9:].reshape(3, 1)

    # Tile HI and SI and Wb to (n, i, j) arrays
    scaled_soft_iron_inv_i = np.tile(scaled_soft_iron_inv, (skew_symmetric_angular_rate.shape[0], 1, 1))
    mb_i = np.tile(mb, (skew_symmetric_angular_rate.shape[0], 1, 1))
    wb_i = np.tile(nm.vec_to_so3(wb), (skew_symmetric_angular_rate.shape[0], 1, 1))

    # Cost function
    sensor_model = (
        (scaled_soft_iron_inv_i @ magnetic_field_derivative)
        + (skew_symmetric_angular_rate @ scaled_soft_iron_inv_i @ magnetic_field)
        - (skew_symmetric_angular_rate @ mb_i)
        - (wb_i @ scaled_soft_iron_inv_i @ magnetic_field)
        + (wb_i @ mb_i)
    )

    # Error norm
    error = (np.linalg.norm(sensor_model.reshape(-1, 3), axis=1)).flatten()
    return error


def _residual_factor(m_dot: np.ndarray, m: np.ndarray, g: np.ndarray, this: gtsam.CustomFactor, v: gtsam.Values,
                     H: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Unary factor for the residual of the sensor model:

    $$R_i = (m_i(t)^T \\otimes [w_i(t)]) \\cdot vec(C) - (m_i(t)^T \\otimes [d]) \\cdot vec(C) +
    (\\dot{m}_i(t)^T \\otimes \\mathbb{I}_ 3) \\cdot vec(C) - (\\mathbb{I}_1 \\otimes [w_i(t)]) \\cdot b +
    (\\mathbb{I}_1 \\otimes [d]) \\cdot b $$

    Where, $m_i(t) \\; \\in \\; \\mathbb{R}^3$ is the magnetic field measurement,
    $\\dot{m}_i(t) \\; \\in \\; \\mathbb{R}^3$ is the differentiation with respect
    to the time of the magnetic field measurement, $[w_i(t)] \\; \\in \\; \\mathbb{R}^{3\\times 3}$
    is the skew-symmetric matrix of the gyroscope measurements, $vec(C) \\; \\in \\; \\mathbb{R}^{12}$
    is the stack of the inverse of the soft-iron, $[d] \\; \\in \\; \\mathbb{R}^{3\\times 3}$
    is the skew-symmetric matrix of the gyroscope bias, and $b \\; \\in \\; \\mathbb{R}^3$
    is the hard-iron.

    Args:
        m_dot (np.ndarray): Derivative of the magnetic field measurement in G/s as a (3, 1)
            numpy array.
        m (np.ndarray): Magnetic field measurements in G as a (3, 1) numpy array.
        g (np.ndarray): Gyroscope measurement in rad/s as a (3, 1) numpy array.
        this (gtsam.CustomFactor): Reference to the current CustomFactor being evaluated.
        v (gtsam.Values): A values structure that maps from keys to values.
        H (List[np.ndarray], Optional): List of references to the Jacobian arrays.

    Returns:
        error (np.ndarray): The non-linear residual error as a gtsam factor.
    """
    key1, key2, key3 = this.keys()[0], this.keys()[1], this.keys()[2]
    l, b, d = v.atVector(key1), v.atVector(key2), v.atVector(key3)

    # Convert state into single variables
    l_matrix = np.array([[np.exp(l[0]), 0.0, 0.0], [l[1], np.exp(l[2]), 0.0], [l[3], l[4], np.exp(l[5])]])

    # Get measurements
    m0, m1, m2, n0, n1, n2 = m[0, 0], m[1, 0], m[2, 0], m_dot[0, 0], m_dot[1, 0], m_dot[2, 0]
    w0, w1, w2 = g[0, 0], g[1, 0], g[2, 0]
    l0, l1, l2, l3, l4, l5 = l[0], l[1], l[2], l[3], l[4], l[5]
    b0, b1, b2, d0, d1, d2 = b[0], b[1], b[2], d[0], d[1], d[2]

    # Jacobian construction
    # Residual Jacibian with respect to the li components
    j1 = np.zeros((3, 6))
    j1[0, 0] = l1*m0*(d2 - w2)*exp(l0) + l1*n1*exp(l0) + l3*m0*(-d1 + w1)*exp(l0) + l3*n2*exp(l0) + 2*n0*exp(2*l0)
    j1[1, 0] = l1*n0*exp(l0) + l3*m0*(d0 - w0)*exp(l0) + (-d2 + w2)*(l1*m1*exp(l0) + l3*m2*exp(l0) + 2*m0*exp(2*l0))
    j1[2, 0] = l1*m0*(-d0 + w0)*exp(l0) + l3*n0*exp(l0) + (d1 - w1)*(l1*m1*exp(l0) + l3*m2*exp(l0) + 2*m0*exp(2*l0))
    j1[0, 1] = l3*m1*(-d1 + w1) + n1*exp(l0) + (d2 - w2)*(2*l1*m1 + l3*m2 + m0*exp(l0))
    j1[1, 1] = 2*l1*n1 + l3*m1*(d0 - w0) + l3*n2 + m1*(-d2 + w2)*exp(l0) + n0*exp(l0)
    j1[2, 1] = l3*n1 + m1*(d1 - w1)*exp(l0) + (-d0 + w0)*(2*l1*m1 + l3*m2 + m0*exp(l0))
    j1[0, 2] = l4*m1*(-d1 + w1)*exp(l2) + (d2 - w2)*(l4*m2*exp(l2) + 2*m1*exp(2*l2))
    j1[1, 2] = l4*m1*(d0 - w0)*exp(l2) + l4*n2*exp(l2) + 2*n1*exp(2*l2)
    j1[2, 2] = l4*n1*exp(l2) + (-d0 + w0)*(l4*m2*exp(l2) + 2*m1*exp(2*l2))
    j1[0, 3] = l1*m2*(d2 - w2) + n2*exp(l0) + (-d1 + w1)*(l1*m1 + 2*l3*m2 + m0*exp(l0))
    j1[1, 3] = l1*n2 + m2*(-d2 + w2)*exp(l0) + (d0 - w0)*(l1*m1 + 2*l3*m2 + m0*exp(l0))
    j1[2, 3] = l1*m2*(-d0 + w0) + l1*n1 + 2*l3*n2 + m2*(d1 - w1)*exp(l0) + n0*exp(l0)
    j1[0, 4] = m2*(d2 - w2)*exp(l2) + (-d1 + w1)*(2*l4*m2 + m1*exp(l2))
    j1[1, 4] = n2*exp(l2) + (d0 - w0)*(2*l4*m2 + m1*exp(l2))
    j1[2, 4] = 2*l4*n2 + m2*(-d0 + w0)*exp(l2) + n1*exp(l2)
    j1[0, 5] = 2*m2*(-d1 + w1)*exp(2*l5)
    j1[1, 5] = 2*m2*(d0 - w0)*exp(2*l5)
    j1[2, 5] = 2*n2*exp(2*l5)

    # Residual Jacibian with respect to the b components
    j2 = np.zeros((3, 3))
    j2[:, [0]] = np.array([[0], [d2 - w2], [-d1 + w1]])
    j2[:, [1]] = np.array([[-d2 + w2], [0], [d0 - w0]])
    j2[:, [2]] = np.array([[d1 - w1], [-d0 + w0], [0]])

    # Residual Jacibian with respect to the d components
    j3 = np.zeros((3, 3))
    j3[0, 0] = 0
    j3[1, 0] = -b2 + l3*m0*exp(l0) + m1*(l1*l3 + l4*exp(l2)) + m2*(l3**2 + l4**2 + exp(2*l5))
    j3[2, 0] = b1 - l1*m0*exp(l0) - m1*(l1**2 + exp(2*l2)) - m2*(l1*l3 + l4*exp(l2))
    j3[0, 1] = b2 - l3*m0*exp(l0) - m1*(l1*l3 + l4*exp(l2)) - m2*(l3**2 + l4**2 + exp(2*l5))
    j3[1, 1] = 0
    j3[2, 1] = -b0 + l1*m1*exp(l0) + l3*m2*exp(l0) + m0*exp(2*l0)
    j3[0, 2] = -b1 + l1*m0*exp(l0) + m1*(l1**2 + exp(2*l2)) + m2*(l1*l3 + l4*exp(l2))
    j3[1, 2] = b0 - l1*m1*exp(l0) - l3*m2*exp(l0) - m0*exp(2*l0)
    j3[2, 2] = 0

    if H is not None:
        H[0] = j1
        H[1] = j2
        H[2] = j3

    # Cost Function
    c_matrix = l_matrix @ l_matrix.T
    error = nm.vec_to_so3(g.flatten() - d) @ (c_matrix @ m - np.vstack(b)) + c_matrix @ m_dot
    return error


def _volume_factor(this: gtsam.CustomFactor, v: gtsam.Values, H: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Unary factor for the difference of the volume of the unitary sphere and the
    volume of the deformed sphere:

    $$ E_i = \\exp(2*l_0) \\cdot \\exp(2*l_2) \\cdot \\exp(2*l_5) - 1$$

    Where, $\\exp(2*l_i) \\; \\forall \\; i \\; \\in \\; \\{0, 2, 5\\}$ are
    the diagonals terms of the inverse of the soft-iron.

    Args:
        this (gtsam.CustomFactor): Reference to the current CustomFactor being evaluated.
        v (gtsam.Values): A values structure that maps from keys to values.
        H (List[np.ndarray], Optional): List of references to the Jacobian arrays.

    Returns:
        error (np.ndarray): The non-linear residual error of the factor.
    """
    key0 = this.keys()[0]
    l_params = v.atVector(key0)
    l0, l2, l5 = l_params[0], l_params[2], l_params[5]

    # The determinant of C is the multiplication of the diagonal elements, as
    # the matrix is upper triangular.
    det_c = exp(2*l0) * exp(2*l2) * exp(2*l5)

    # Jacobian construction
    j0 = np.zeros((1, 6))

    j0[0, 0] = 2*det_c
    j0[0, 1] = 0.0
    j0[0, 2] = 2*det_c
    j0[0, 3] = 0.0
    j0[0, 4] = 0.0
    j0[0, 5] = 2*det_c

    if H is not None:
        H[0] = j0

    # Cost Function
    error = np.array([det_c - 1])
    return error


def _difference_factor(this: gtsam.CustomFactor, v: gtsam.Values, H: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Unary factor for the difference between the diagonal terms of the matrix:

    $$ E_i = (\\exp(2*l_0) - \\exp(2*l_2))^2 + (\\exp(2*l_0) - \\exp(2*l_5))^2 + (\\exp(2*l_2) - \\exp(2*l_5))^2 +$$

    Where, $\\exp(2*l_i) \\; \\forall \\; i \\; \\in \\; \\{0, 2, 5\\}$ are
    the diagonals terms of the inverse of the soft-iron.

    Args:
        this (gtsam.CustomFactor): Reference to the current CustomFactor being evaluated.
        v (gtsam.Values): A values structure that maps from keys to values.
        H (List[np.ndarray], Optional): List of references to the Jacobian arrays.

    Returns:
        error (np.ndarray): The non-linear residual error of the factor.
    """
    key0 = this.keys()[0]
    l_params = v.atVector(key0)
    l0, l1, l2, l3, l4, l5 = l_params[0], l_params[1], l_params[2], l_params[3], l_params[4], l_params[5]
    l_matrix = np.array([[np.exp(l_params[0]), 0.0, 0.0],
                         [l_params[1], np.exp(l_params[2]), 0.0],
                         [l_params[3], l_params[4], np.exp(l_params[5])]])

    # Jacobian construction
    j0 = np.zeros((1, 6))

    j0[0, 0] = 4*(-l1**2 - l3**2 - l4**2 + 2*exp(2*l0) - exp(2*l2) - exp(2*l5))*exp(2*l0)
    j0[0, 1] = 4*l1*(2*l1**2 - l3**2 - l4**2 - exp(2*l0) + 2*exp(2*l2) - exp(2*l5))
    j0[0, 2] = 4*(2*l1**2 - l3**2 - l4**2 - exp(2*l0) + 2*exp(2*l2) - exp(2*l5))*exp(2*l2)
    j0[0, 3] = 4*l3*(-l1**2 + 2*l3**2 + 2*l4**2 - exp(2*l0) - exp(2*l2) + 2*exp(2*l5))
    j0[0, 4] = 4*l4*(-l1**2 + 2*l3**2 + 2*l4**2 - exp(2*l0) - exp(2*l2) + 2*exp(2*l5))
    j0[0, 5] = 4*(-l1**2 + 2*l3**2 + 2*l4**2 - exp(2*l0) - exp(2*l2) + 2*exp(2*l5))*exp(2*l5)

    if H is not None:
        H[0] = j0

    # Cost Function
    c_matrix = l_matrix @ l_matrix.T
    c00, c11, c22 = c_matrix[0, 0], c_matrix[1, 1], c_matrix[2, 2]
    error = np.array([(c00 - c11)**2 + (c00 - c22)**2 + (c11 - c22)**2])
    return error


def _gtsam_optimize(optimizer: Union[gtsam.LevenbergMarquardtOptimizer, gtsam.DoglegOptimizer],
                    optimizer_params: Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams],
                    ) -> Tuple[gtsam.Values, Dict[str, Union[List[float], int]]]:
    """
    Wrapper for the batch optimization of the non-linear graph with a callback to
    store the optimization error and check the termination conditions.

    Args:
        optimizer (Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams]): Optimizer parameters.
        optimizer_params (Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams]): Optimizer parameters.

    Returns:
        gtsam.Values: The state value in each node as a gtsam.Values structure.
        optimization_status (Dict[str, Union[List[float], int]]): Dictionary with
            the optimization status. The keys are "error" and "iterations".
    """
    error_before = optimizer.error()
    optimization_status = {"error": [error_before], "iterations": 0}

    while True:
        # Optimize
        optimizer.iterate()
        error_after = optimizer.error()

        # Store errors
        optimization_status["error"].append(error_after)

        # Check termination condition
        # Condition 1: Maximum number of iterations
        condition_1 = optimizer.iterations() >= optimizer_params.getMaxIterations()

        # Condition 2: Convergence
        condition_2 = gtsam.checkConvergence(optimizer_params.getRelativeErrorTol(),
                                             optimizer_params.getAbsoluteErrorTol(),
                                             optimizer_params.getErrorTol(),
                                             error_before, error_after)

        # Condition 3: Reach upper bound of lambda
        condition_3 = (isinstance(optimizer, gtsam.LevenbergMarquardtOptimizer) and
                       optimizer.lambda_() > optimizer_params.getlambdaUpperBound())

        if condition_1 or condition_2 or condition_3:
            optimization_status["iterations"] = optimizer.iterations()
            return optimizer.values(), optimization_status

        error_before = error_after
