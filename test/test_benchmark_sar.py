"""
Test Module for the Benchmark SAR Module

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import unittest
from typing import Tuple

import navlib.math as nm
import numpy as np

from magyc.benchmark_methods import (
    sar_ls,
    sar_aid,
    sar_kf,
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


class TestSarLs(unittest.TestCase):

    def test_valid_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        hard_iron, calibrated_magnetic_field = sar_ls(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)

    def test_list_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        angular_rate = TEST_ANGULAR_RATE.tolist()
        time = TEST_TIME.tolist()
        hard_iron, calibrated_magnetic_field = sar_ls(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, np.array(magnetic_field).shape)

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = "invalid"
        time = np.array([0, 1, 2])
        with self.assertRaises(TypeError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_invalid_time_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = "invalid"
        with self.assertRaises(TypeError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([0.1, 0.2, 0.3, 0.4])
        time = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_invalid_time_shape(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            sar_ls(magnetic_field, angular_rate, time)

    def test_mismatched_samples(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        angular_rate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        time = np.array([0, 1])
        with self.assertRaises(ValueError):
            sar_ls(magnetic_field, angular_rate, time)


class TestSarAid(unittest.TestCase):

    def test_valid_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        hard_iron, calibrated_magnetic_field, filtered_magnetic_field = sar_aid(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)
        self.assertEqual(filtered_magnetic_field.shape, (len(magnetic_field), 3))

    def test_list_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        angular_rate = TEST_ANGULAR_RATE.tolist()
        time = TEST_TIME.tolist()
        hard_iron, calibrated_magnetic_field, filtered_magnetic_field = sar_aid(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, (len(magnetic_field), 3))
        self.assertEqual(filtered_magnetic_field.shape, (len(magnetic_field), 3))

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = "invalid"
        time = TEST_TIME
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_time_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = "invalid"
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        with self.assertRaises(ValueError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = np.array([0.1, 0.2, 0.3, 0.4])
        time = TEST_TIME
        with self.assertRaises(ValueError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_time_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_mismatched_samples(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME[:, :-1]
        with self.assertRaises(ValueError):
            sar_aid(magnetic_field, angular_rate, time)

    def test_invalid_gains_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        gains = "invalid"
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time, gains=gains)

    def test_invalid_gains_elements(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        gains = (1.0, "invalid")
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time, gains=gains)

    def test_invalid_f_normalize_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        f_normalize = "invalid"
        with self.assertRaises(TypeError):
            sar_aid(magnetic_field, angular_rate, time, f_normalize=f_normalize)


class TestSarKf(unittest.TestCase):

    def test_valid_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        hard_iron, calibrated_magnetic_field, filtered_magnetic_field = sar_kf(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, magnetic_field.shape)
        self.assertEqual(filtered_magnetic_field.shape, (len(magnetic_field), 3))

    def test_list_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD.tolist()
        angular_rate = TEST_ANGULAR_RATE.tolist()
        time = TEST_TIME.tolist()
        hard_iron, calibrated_magnetic_field, filtered_magnetic_field = sar_kf(magnetic_field, angular_rate, time)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(calibrated_magnetic_field.shape, (len(magnetic_field), 3))
        self.assertEqual(filtered_magnetic_field.shape, (len(magnetic_field), 3))

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = "invalid"
        time = TEST_TIME
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_time_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = "invalid"
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        with self.assertRaises(ValueError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_angular_rate_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = np.array([0.1, 0.2, 0.3, 0.4])
        time = TEST_TIME
        with self.assertRaises(ValueError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_time_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_mismatched_samples(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME[:, :-1]
        with self.assertRaises(ValueError):
            sar_kf(magnetic_field, angular_rate, time)

    def test_invalid_gains_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        gains = "invalid"
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time, gains=gains)

    def test_invalid_gains_elements(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        gains = (1.0, "invalid")
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time, gains=gains)

    def test_invalid_f_normalize_type(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        angular_rate = TEST_ANGULAR_RATE
        time = TEST_TIME
        f_normalize = "invalid"
        with self.assertRaises(TypeError):
            sar_kf(magnetic_field, angular_rate, time, f_normalize=f_normalize)
