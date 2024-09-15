"""
Test Module for the Benchmark SAR Module

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import unittest
from typing import Tuple

import navlib.math as nm
import numpy as np

from magyc.methods import (
    magyc_ls,
    magyc_nls,
    magyc_bfg,
    magyc_ifg,
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
    wNoise = np.random.randn(100000, 3) * 0.005

    # add HSI and Wb
    si = np.array([[1.1, 0.02, 0.01], [0.02, 1.2, 0.03], [0.01, 0.03, 1.3]])
    hi = np.array([0.1, 0.2, 0.3])
    wb = np.zeros_like(hi)
    mm = (si @ m.T + hi.reshape(-1, 1)).T
    wm = w + wb

    # Add data to dictionary
    return mm + mNoise, wm + wNoise, t.reshape(-1, 1)


TEST_MAGNETIC_FIELD, TEST_ANGULAR_RATE, TEST_TIME = generateData()


class TestMagycLs(unittest.TestCase):

    def test_valid_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        hard_iron, soft_iron, calibrated_magnetic_field = magyc_ls(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)

    def test_list_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        angular_rate = TEST_ANGULAR_RATE.tolist()
        time = TEST_TIME.tolist()
        hard_iron, soft_iron, calibrated_magnetic_field = magyc_ls(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(calibrated_magnetic_field.shape, np.array(magnetic_field).shape)

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = "invalid"
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_invalid_time_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = "invalid"
        with self.assertRaises(TypeError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([0.1, 0.2, 0.3, 0.4])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_invalid_time_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            magyc_ls(magnetic_field, angular_rate, time)

    def test_mismatched_samples(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1])
        with self.assertRaises(ValueError):
            magyc_ls(magnetic_field, angular_rate, time)


class TestMagycNls(unittest.TestCase):

    def test_valid_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field, calibrated_angular_rates = magyc_nls(magnetic_field,
                                                                                                         angular_rate,
                                                                                                         time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)
        self.assertEqual(calibrated_angular_rates.shape, angular_rate.shape)

    def test_list_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        angular_rate = TEST_ANGULAR_RATE.tolist()
        time = TEST_TIME.tolist()
        hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field, calibrated_angular_rates = magyc_nls(magnetic_field,
                                                                                                         angular_rate,
                                                                                                         time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, TEST_MAGNETIC_FIELD.shape)
        self.assertEqual(calibrated_angular_rates.shape, TEST_ANGULAR_RATE.shape)

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = "invalid"
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_invalid_time_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = "invalid"
        with self.assertRaises(TypeError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([0.1, 0.2, 0.3, 0.4])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_invalid_time_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            magyc_nls(magnetic_field, angular_rate, time)

    def test_mismatched_samples(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1])
        with self.assertRaises(ValueError):
            magyc_nls(magnetic_field, angular_rate, time)


class TestMagycBfg(unittest.TestCase):
    def setUp(self):
        # Set up valid test data
        self.valid_magnetic_field = TEST_MAGNETIC_FIELD
        self.valid_angular_rate = TEST_ANGULAR_RATE
        self.valid_time = TEST_TIME
        self.valid_measurement_window = 25
        self.valid_optimizer = "dogleg"
        self.valid_relative_error_tol = 1.00e-12
        self.valid_absolute_error_tol = 1.00e-12
        self.valid_max_iter = 1000

    def test_valid_input(self):
        magnetic_field = self.valid_magnetic_field
        angular_rate = self.valid_angular_rate
        time = self.valid_time
        (hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field,
         calibrated_angular_rates, optimization_status) = magyc_bfg(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)
        self.assertEqual(calibrated_angular_rates.shape, angular_rate.shape)
        self.assertEqual(set(optimization_status.keys()), {"error", "iterations"})

    def test_list_input(self):
        magnetic_field = self.valid_magnetic_field.tolist()
        angular_rate = self.valid_angular_rate.tolist()
        time = self.valid_time.tolist()
        (hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field,
         calibrated_angular_rates, optimization_status) = magyc_bfg(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, TEST_MAGNETIC_FIELD.shape)
        self.assertEqual(calibrated_angular_rates.shape, TEST_ANGULAR_RATE.shape)
        self.assertEqual(set(optimization_status.keys()), {"error", "iterations"})

    def test_invalid_magnetic_field_type(self):
        with self.assertRaises(TypeError):
            magyc_bfg("invalid_type", self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_angular_rate_type(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, "invalid_type", self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_time_type(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, "invalid_type",
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_magnetic_field_shape(self):
        with self.assertRaises(ValueError):
            magyc_bfg(np.array([1, 2, 3, 4]), self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_angular_rate_shape(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, np.array([1, 2, 3, 4]), self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_time_shape(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, np.array([0, 1, 2]),
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_measurement_window_type(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      "invalid_type", self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_optimizer_type(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, "invalid_type",
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_relative_error_tol_type(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      "invalid_type", self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_absolute_error_tol_type(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, "invalid_type",
                      self.valid_max_iter)

    def test_invalid_max_iter_type(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      "invalid_type")

    def test_invalid_relative_error_tol_value(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      -1.0, self.valid_absolute_error_tol,
                      self.valid_max_iter)

    def test_invalid_absolute_error_tol_value(self):
        with self.assertRaises(TypeError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, -1.0,
                      self.valid_max_iter)

    def test_invalid_max_iter_value(self):
        with self.assertRaises(ValueError):
            magyc_bfg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window, self.valid_optimizer,
                      self.valid_relative_error_tol, self.valid_absolute_error_tol,
                      -1)


class TestMagycIfg(unittest.TestCase):
    def setUp(self):
        # Set up valid test data
        self.valid_magnetic_field = TEST_MAGNETIC_FIELD
        self.valid_angular_rate = TEST_ANGULAR_RATE
        self.valid_time = TEST_TIME
        self.valid_measurement_window = 25

    def test_valid_input(self):
        magnetic_field = self.valid_magnetic_field
        angular_rate = self.valid_angular_rate
        time = self.valid_time
        (hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field,
         calibrated_angular_rates, optimization_status) = magyc_ifg(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)
        self.assertEqual(calibrated_angular_rates.shape, angular_rate.shape)
        self.assertEqual(set(optimization_status.keys()), {"soft_iron", "hard_iron", "gyro_bias"})

    def test_list_input(self):
        magnetic_field = self.valid_magnetic_field.tolist()
        angular_rate = self.valid_angular_rate.tolist()
        time = self.valid_time.tolist()
        (hard_iron, soft_iron, gyro_bias, calibrated_magnetic_field,
         calibrated_angular_rates, optimization_status) = magyc_ifg(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(gyro_bias.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, TEST_MAGNETIC_FIELD.shape)
        self.assertEqual(calibrated_angular_rates.shape, TEST_ANGULAR_RATE.shape)
        self.assertEqual(set(optimization_status.keys()), {"soft_iron", "hard_iron", "gyro_bias"})

    def test_invalid_magnetic_field_type(self):
        with self.assertRaises(TypeError):
            magyc_ifg("invalid_type", self.valid_angular_rate, self.valid_time, self.valid_measurement_window)

    def test_invalid_angular_rate_type(self):
        with self.assertRaises(TypeError):
            magyc_ifg(self.valid_magnetic_field, "invalid_type", self.valid_time, self.valid_measurement_window)

    def test_invalid_time_type(self):
        with self.assertRaises(TypeError):
            magyc_ifg(self.valid_magnetic_field, self.valid_angular_rate, "invalid_type", self.valid_measurement_window)

    def test_invalid_magnetic_field_shape(self):
        with self.assertRaises(ValueError):
            magyc_ifg(np.array([1, 2, 3, 4]), self.valid_angular_rate, self.valid_time,
                      self.valid_measurement_window)

    def test_invalid_angular_rate_shape(self):
        with self.assertRaises(ValueError):
            magyc_ifg(self.valid_magnetic_field, np.array([1, 2, 3, 4]), self.valid_time,
                      self.valid_measurement_window)

    def test_invalid_time_shape(self):
        with self.assertRaises(ValueError):
            magyc_ifg(self.valid_magnetic_field, self.valid_angular_rate, np.array([0, 1, 2]),
                      self.valid_measurement_window)

    def test_invalid_measurement_window_type(self):
        with self.assertRaises(ValueError):
            magyc_ifg(self.valid_magnetic_field, self.valid_angular_rate, self.valid_time, "invalid_type")
