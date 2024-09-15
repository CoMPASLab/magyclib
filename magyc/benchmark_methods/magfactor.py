"""
MAGYC - Benchmark Methods - magfactor3

This module contains the magfactor3 method. This method is a factor graph
implementation of the full-magnetometer calibration least-squares problems.

Functions:
    magfactor3: Factor graph based approach to full-magnetometer calibration.

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
from functools import partial
from typing import List, Tuple, Union, Optional

import gtsam
import navlib.math as nm
import numpy as np
from gtsam.symbol_shorthand import B, S


def magfactor3(magnetic_field: Union[np.ndarray, list], rph: Union[np.ndarray, list], magnetic_declination: float,
               reference_magnetic_field: Union[np.ndarray, list], optimizer: str = "dogleg",
               relative_error_tol: float = 1.00e-12, absolute_error_tol: float = 1.00e-12,
               max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    The full-magnetometer calibration least-squares problems can also be modeled
    as a factor graph. This can be implemented using the [GTSAM](https://github.com/borglab/gtsam)
    python wrapper with the magFactor3 factor. This approach allows the user to get the soft-iron
    (SI) as the identity scaled by a constant and the hard-iron (HI) from the
    magnetometer bias.

    This method assumes that the rotation from the body frame with respect to
    the world frame and the local magnetic field are known.

    Args:
        magnetic_field (Union[np.ndarray, list]): Magnetic field raw data
        rph (Union[np.ndarray, list]): Roll, pitch and heading data
        magnetic_declination (float): Magnetic declination in degrees
        reference_magnetic_field (Union[np.ndarray, list]): Reference magnetic field
        optimizer (str): Optimization algorithm to use. Options are "dogleg" or "lm"
            for the Dogleg and Levenberg-Marquardt optimizers respectively.
        relative_error_tol (float): Relative error tolerance for the optimizer. Default is 1.00e-12
        absolute_error_tol (float): Absolute error tolerance for the optimizer. Default is 1.00e-12
        max_iter (int): Maximum number of iterations for the optimizer. Default is 1000

    Returns:
        hard_iron (np.ndarray): Hard-iron offset in G
        soft_iron (np.ndarray): Soft-iron scaling matrix
        corrected_magnetic_field (np.ndarray): Corrected magnetic field data
        optimization_errors (list): List of optimization errors in each iteration

    Raises:
        TypeError: If the magnetic field input is not a numpy array or a list
        TypeError: If the reference magnetic field input is not a numpy array or a list
        TypeError: If the rph input is not a numpy array or a list
        ValueError: If the magnetic field input is not a 3xN or Nx3 numpy array
        ValueError: If the reference magnetic field input is not a 3, numpy array
        ValueError: If the rph input is not a 3xN or Nx3 numpy array
        TypeError: If the magnetic declination is not a float
        ValueError: If the optimizer is not a string or not "dogleg" or "lm"
        TypeError: If the relative error tolerance is not a float
        TypeError: If the absolute error tolerance is not a float
        ValueError: If the maximum number of iterations is not a positive integer
    """
    # Check if the magnetic field input is a list and convert it to a numpy array
    if isinstance(magnetic_field, list):
        magnetic_field = np.array(magnetic_field)

    # Check if the reference magnetic field input is a list and convert it to a numpy array
    if isinstance(reference_magnetic_field, list):
        reference_magnetic_field = np.array(reference_magnetic_field).flatten()

    # Check if the rph input is a list and convert it to a numpy array
    if isinstance(rph, list):
        rph = np.array(rph)

    # Check if the magnetic field input is a numpy array
    if not isinstance(magnetic_field, np.ndarray):
        raise TypeError("The magnetic field input must be a numpy array or a list.")

    # Check if the reference magnetic field input is a numpy array
    if not isinstance(reference_magnetic_field, np.ndarray):
        raise TypeError("The reference magnetic field input must be a numpy array or a list.")

    # Check if the rph input is a numpy array
    if not isinstance(rph, np.ndarray):
        raise TypeError("The rph input must be a numpy array or a list.")

    # Check if the magnetic field input is a 3xN or Nx3 numpy array
    if magnetic_field.ndim != 2 or (magnetic_field.shape[0] != 3 and magnetic_field.shape[1] != 3):
        raise ValueError("The magnetic field input must be a 3xN or Nx3 numpy array.")

    # Check if the reference magnetic field input is a 3, numpy array
    reference_magnetic_field = reference_magnetic_field.flatten()
    if reference_magnetic_field.shape[0] != 3:
        raise ValueError("The reference magnetic field input must be a 3, or 1x3, or 3x1 numpy array.")

    # Check if the rph input is is a 3xN or Nx3 numpy array
    if rph.ndim != 2 or (rph.shape[0] != 3 and rph.shape[1] != 3):
        raise ValueError("The rph input must be a 3xN or Nx3 numpy array.")

    # Force the magnetic field array to be a Nx3 numpy array
    if magnetic_field.shape[0] == 3:
        magnetic_field = magnetic_field.T

    # Force the rph array to be a Nx3 numpy array
    if rph.shape[0] == 3:
        rph = rph.T

    # Check that the magnetic declination is a float
    if not isinstance(magnetic_declination, float):
        raise TypeError("The magnetic declination must be a float.")

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

    # Compute attitude based on magnetic heading
    magnetic_hdg = _ahrs_raw_hdg(magnetic_field, rph) - np.deg2rad(magnetic_declination)
    magnetic_rph = np.concatenate([rph[:, :2], magnetic_hdg.reshape(-1, 1)], axis=1)

    # Compute calibration
    # Smoothing and Mapping Factor Graph
    # 1. Create the non-linear graph
    graph = gtsam.NonlinearFactorGraph()

    # 2. noise model for each factor.
    residual_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)

    # 3. Creates values structure with initial values: S -> Scale, D -> Direction, B -> Bias
    initial = gtsam.Values()
    initial.insert(S(0), 1.0)
    initial.insert(B(0), gtsam.Point3(0, 0, 0))
    keys = [S(0), B(0)]

    # 4. Add factor for each measurement into a single node
    h0 = gtsam.Point3(reference_magnetic_field.flatten())

    for i in range(magnetic_field.shape[0]):
        mi = gtsam.Point3(magnetic_field[i, :])
        bRw = gtsam.Rot3(nm.rph2rot(magnetic_rph[i, :]).T)

        # 5.1 magFactor3
        rf = gtsam.CustomFactor(residual_noise, keys, partial(_residual_factor, mi, h0, bRw))
        graph.add(rf)

    # 5. If not online optimize the full batch
    # 5.1 Create optimizer parameters
    params = gtsam.DoglegParams() if optimizer == "dogleg" else gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(relative_error_tol)
    params.setAbsoluteErrorTol(absolute_error_tol)
    params.setMaxIterations(max_iter)
    params.setLinearSolverType("MULTIFRONTAL_CHOLESKY")

    # 5.2 Create optimizer
    if optimizer == "dogleg":
        optimizer = gtsam.DoglegOptimizer(graph, initial, params)
    else:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    # 5.3 Optimize
    result, optimization_errors = _gtsam_optimize(optimizer, params)

    # 7. Process Results
    hard_iron = np.vstack(result.atPoint3(B(0)))
    soft_iron = result.atDouble(S(0)) * np.eye(3)

    # Correct the magnetic field
    corrected_magnetic_field = (np.linalg.inv(soft_iron) @ (magnetic_field.T - hard_iron.reshape(3, -1))).T

    return hard_iron.flatten(), soft_iron, corrected_magnetic_field, optimization_errors


def _ahrs_raw_hdg(magnetic_field: Union[np.ndarray, List[float]],
                  rph: Union[np.ndarray, List[float]] = None) -> np.ndarray:
    """
    raw_hdg computes the heading from magnetic field measurements and rph data.

    If rph is a parameter, the using roll and pitch the corresponding rotation
    matrices are computed and the magnetic field measuremnts are transformated
    to measurements in the xy plane. With the planar magnetic field measuremnts,
    the heading is computed as: heading = np.arcant2(-my, mx)

    Args:
        magnetic_field (Union[np.ndarray, List[float]]): Magnetic field raw data
        rph (Union[np.ndarray, List[float]], optional): Roll, pitch and heading data

    Returns:
        np.ndarray: Heading angle in radians

    Raises:
        ValueError: If mag_field is not a numpy array or list
        ValueError: If rph is not a numpy array or list
    """
    # Check inputs
    if not (isinstance(magnetic_field, np.ndarray) or isinstance(magnetic_field, list)):
        raise ValueError("mag_field must be a numpy array or list")

    if rph is not None:
        if not (isinstance(rph, np.ndarray) or isinstance(rph, list)):
            raise ValueError("rph must be a numpy array or list")

    # Flatten Magnetic Field if the RPH is provided
    if rph is not None:
        rph = rph.reshape(-1, 3)
        rot_mat_flat = np.apply_along_axis(nm.rph2rot, 1, np.concatenate([rph[:, [0, 1]], rph[:, [2]]*0], axis=1))
        mf = np.einsum('ijk->ikj', rot_mat_flat @ magnetic_field.reshape(-1, 3, 1))
    else:
        mf = magnetic_field

    # Calculate HDG
    heading = np.arctan2(-mf[:, :, [1]], mf[:, :, [0]]). reshape(-1, 1)

    return heading


def _residual_factor(magfield: gtsam.Point3, local_magfield: gtsam.Point3, bRw: gtsam.Rot3, this: gtsam.CustomFactor,
                     v: gtsam.Values, H: Optional[List[np.ndarray]]) -> np.ndarray:
    """
    Unary factor for the magnetometer model:

    $$m_m(t) = \\text{scale} \\cdot m_t(t) + \\text{bias}$$

    Where $m_m(t) \\; \\in \\; \\mathbb{R}^3$ is the measured magnetic field,
    $m_t(t) \\; \\in \\; \\mathbb{R}^3$ is the true magnetic field, $\\text{scale} \\; \\in \\; \\mathbb{R}$
    is the scale factor and $\\text{bias} \\; \\in \\; \\mathbb{R}^3$ is the
    magnetometer bias.

    Args:
        magfield (gtsam.Point3): Magnetic field measurements in G as a (3, 1) gtsam Point3 object.
        local_magfield (gtsam.Point3): Local magnetic field from model in G as a (3, 1) gtsam Point 3 object.
        bRw (gtsam.Rot3): Attitude of the world frame with respect to the body frame as gtsam Rot3 object.
        this (gtsam.CustomFactor): Reference to the current CustomFactor being evaluated.
        v (gtsam.Values): A values structure that maps from keys to values.
        H (List[np.ndarray], optional): List of references to the Jacobian arrays.

    Returns:
        error (np.ndarray): The non-linear norm error with respect to the unitary norm as a gtsam factor.
    """
    key0, key1 = this.keys()[0], this.keys()[1]
    scale, bias = v.atDouble(key0), v.atPoint3(key1)

    # Cost Function
    rotated = gtsam.Point3(bRw.rotate(local_magfield))
    hx = scale * rotated + bias
    error = hx - magfield

    if H is not None:
        H[0] = np.vstack(rotated)
        H[1] = np.eye(3)

    return error


def _gtsam_optimize(optimizer: Union[gtsam.LevenbergMarquardtOptimizer, gtsam.DoglegOptimizer],
                    optimizer_params: Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams],
                    ) -> Union[gtsam.Values, list]:
    """
    Wrapper for the batch optimization of the non-linear graph with a callback to
    store the optimization error and check the termination conditions.

    Args:
        optimizer (Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams]): Optimizer parameters.
        optimizer_params (Union[gtsam.LevenbergMarquardtParams, gtsam.DoglegParams]): Optimizer parameters.

    Returns:
        gtsam.Values: The state value in each node as a gtsam.Values structure.
        optimization_error (list): The optimization error in each iteration.
    """
    optimization_error = []
    error_before = optimizer.error()

    while True:
        # Optimize
        optimizer.iterate()
        error_after = optimizer.error()

        # Store errors
        optimization_error.append([error_before, error_after, error_before - error_after])

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
            return optimizer.values(), optimization_error

        error_before = error_after
