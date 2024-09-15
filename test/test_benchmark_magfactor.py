"""
Test Module for the Benchmark Magfactor3 Module

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import unittest
from typing import Tuple

import navlib.math as nm
import numpy as np

from magyc.benchmark_methods import (
    magfactor3
)


def generateData() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a synthetic dataset using a constant magnetic vector that is
    randomly rotated in three different degrees of motion: low, mid, and high.
    """
    # Initial Conditions
    t = np.linspace(0, 4000, 100000)
    r_A, p_A, h_A = [np.pi, np.pi, np.pi]

    # Roll
    r_init = np.random.uniform(-1, 1, (1,)) * r_A
    r_f = np.random.uniform(0.005, 0.01)
    r = r_A * np.sin(2 * np.pi * r_f * t + r_init)

    # Pitch
    p_init = np.random.uniform(-1, 1, (1,)) * p_A
    p_f = np.random.uniform(0.005, 0.008)
    p = p_A * np.sin(2 * np.pi * p_f * t + p_init)

    # Heading
    h_init = np.random.uniform(-1, 1, (1,)) * h_A
    h_f = np.random.uniform(0.004, 0.008)
    h = h_A * np.sin(2 * np.pi * h_f * t + h_init)

    # RPH
    rph = np.concatenate([r.reshape(-1, 1), p.reshape(-1, 1), h.reshape(-1, 1)], axis=1)

    # Rotation Matrices
    R = np.apply_along_axis(nm.rph2rot, 1, rph)
    RT = np.einsum("ijk->ikj", R)

    # Magnetic field
    magnetic_vector = np.array([[227.207, 51.796, 411.731]]) / 1000
    m = (RT @ np.tile(magnetic_vector.reshape(3, 1), (R.shape[0], 1, 1))).reshape(-1, 3)

    # Gyroscope
    Rij = (RT[:-1, :, :] @ R[1:, :, :]).reshape(-1, 9)
    skew_w = np.apply_along_axis(lambda x: nm.matrix_log3(x.reshape(3, 3)), 1, Rij).reshape(-1, 9)
    w_prime = np.apply_along_axis(lambda x: nm.so3_to_vec(x.reshape(3, 3)).reshape(3, 1), 1, skew_w)
    w = ((1 / np.diff(t)).reshape(-1, 1, 1) * w_prime).reshape(-1, 3)
    w = np.concatenate([w[[0], :], w], axis=0)

    # Measurements Noise
    mNoise = np.random.randn(100000, 3) * 0.001

    # add HSI and Wb
    si = np.array([[1.1, 0.02, 0.01], [0.02, 1.2, 0.03], [0.01, 0.03, 1.3]])
    hi = np.array([0.1, 0.2, 0.3])
    mm = (si @ m.T + hi.reshape(-1, 1)).T

    # Add data to dictionary
    return mm + mNoise, rph


TEST_MAGNETIC_FIELD, TEST_RPH = generateData()


class TestBenchmarkMagfactor3(unittest.TestCase):
    def setUp(self):
        # Set up valid test data
        self.valid_magnetic_field = TEST_MAGNETIC_FIELD
        self.valid_rph = TEST_RPH
        self.valid_magnetic_declination = 10.0
        self.valid_reference_magnetic_field = np.array([25.0, 35.0, 45.0])
        self.valid_optimizer = "dogleg"
        self.valid_relative_error_tol = 1.00e-12
        self.valid_absolute_error_tol = 1.00e-12
        self.valid_max_iter = 1000

    def test_valid_input(self):
        """
        Test the benchmark method with valid input
        """
        magnetic_field = TEST_MAGNETIC_FIELD
        rph = TEST_RPH
        hard_iron, soft_iron, correctded_magnetic_field, error = magfactor3(magnetic_field, rph, 0.0,
                                                                            self.valid_reference_magnetic_field)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(correctded_magnetic_field.shape, magnetic_field.shape)

    def test_list_input(self):
        """
        Test the benchmark method with list input
        """
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        rph = TEST_RPH.tolist()
        hard_iron, soft_iron, correctded_magnetic_field, error = magfactor3(magnetic_field, rph, 0.0,
                                                                            self.valid_reference_magnetic_field)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(correctded_magnetic_field.shape, np.array(magnetic_field).shape)

    def test_invalid_magnetic_field_type(self):
        with self.assertRaises(TypeError):
            magfactor3("invalid_type", self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_reference_magnetic_field_type(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       "invalid_type", self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_rph_type(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, "invalid_type", self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_magnetic_field_shape(self):
        with self.assertRaises(ValueError):
            magfactor3(np.array([1, 2, 3]), self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_reference_magnetic_field_shape(self):
        with self.assertRaises(ValueError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       np.array([1, 2]), self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_rph_shape(self):
        with self.assertRaises(ValueError):
            magfactor3(self.valid_magnetic_field, np.array([1, 2, 3]), self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_magnetic_declination_type(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, "invalid_type",
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_optimizer_type(self):
        with self.assertRaises(ValueError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, "invalid_optimizer",
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_relative_error_tol_type(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       "invalid_type", self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_absolute_error_tol_type(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, "invalid_type",
                       self.valid_max_iter)

    def test_invalid_max_iter_type(self):
        with self.assertRaises(ValueError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       "invalid_type")

    def test_invalid_relative_error_tol_value(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       -1.0, self.valid_absolute_error_tol,
                       self.valid_max_iter)

    def test_invalid_absolute_error_tol_value(self):
        with self.assertRaises(TypeError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, -1.0,
                       self.valid_max_iter)

    def test_invalid_max_iter_value(self):
        with self.assertRaises(ValueError):
            magfactor3(self.valid_magnetic_field, self.valid_rph, self.valid_magnetic_declination,
                       self.valid_reference_magnetic_field, self.valid_optimizer,
                       self.valid_relative_error_tol, self.valid_absolute_error_tol,
                       -1)
