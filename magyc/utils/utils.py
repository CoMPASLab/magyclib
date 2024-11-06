import numpy as np
from warnings import warn


def hsi_calibration_validation(soft_iron: np.ndarray, hard_iron: np.ndarray) -> bool:
    """
    Check if the computed soft-iron and hard-iron matrices correspond to the
    parametrization of an ellipsoid in the real numbers domain and if meet
    the positive definite condition for the soft-iron.

    Args:
        soft_iron (np.ndarray): Soft-iron matrix as a (3, 3) numpy array.
        hard_iron (np.ndarray): Hard-iron matrix as a (3, 1) numpy array.

    Returns:
        bool: Whether the soft-iron and hard-iron parametrize a ellipsoid in the
        real numbers domain.
    """
    soft_iron, hard_iron = soft_iron.reshape(3, 3), hard_iron.reshape(-1, 1)
    soft_iron_inv = np.linalg.inv(soft_iron)
    S = soft_iron_inv.T @ soft_iron_inv
    P = -hard_iron.T @ soft_iron_inv.T @ soft_iron_inv
    d = -(hard_iron.T @ soft_iron_inv.T @ soft_iron_inv @ hard_iron + 1)

    # Create block matrix with S, P and d
    E = np.block([[S, P.T], [P, d]])

    # Conditions
    try:
        cond1 = np.linalg.matrix_rank(S) == 3
        cond2 = np.linalg.matrix_rank(E) == 4
        cond3 = np.linalg.det(E) < 0
        cond4 = all([i > 0 for i in np.linalg.eigvals(S)])
        cond5 = all([i > 0 for i in np.linalg.eigvals(soft_iron)])
    except Exception as e:
        warn(f"An error occurred while validating the calibration matrices: {e}")
        return False

    return all([cond1, cond2, cond3, cond4, cond5])
