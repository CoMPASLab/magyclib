"""
Test Module for the Benchmark Two-Step Module

Authors: Sebastián Rodríguez-Martínez
Contact: srodriguez@mbari.org
"""
import unittest
from typing import Tuple

import navlib.math as nm
import numpy as np

from magyc.benchmark_methods import twostep_hi, twostep_hsi


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


class TestTwostepHi(unittest.TestCase):

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(TypeError):
            twostep_hi(magnetic_field, reference_magnetic_field)

    def test_invalid_reference_magnetic_field_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reference_magnetic_field = "invalid"
        with self.assertRaises(TypeError):
            twostep_hi(magnetic_field, reference_magnetic_field)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hi(magnetic_field, reference_magnetic_field)

    def test_invalid_reference_magnetic_field_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0])
        with self.assertRaises(ValueError):
            twostep_hi(magnetic_field, reference_magnetic_field)

    def test_invalid_max_iterations(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hi(magnetic_field, reference_magnetic_field, max_iterations=-1)

    def test_invalid_measurement_noise_std(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hi(magnetic_field, reference_magnetic_field, measurement_noise_std=-0.1)


class TestTwostepHsi(unittest.TestCase):

    def test_invalid_magnetic_field_type(self):
        magnetic_field = "invalid"
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(TypeError):
            twostep_hsi(magnetic_field, reference_magnetic_field)

    def test_invalid_reference_magnetic_field_type(self):
        magnetic_field = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        reference_magnetic_field = "invalid"
        with self.assertRaises(TypeError):
            twostep_hsi(magnetic_field, reference_magnetic_field)

    def test_invalid_magnetic_field_shape(self):
        magnetic_field = np.array([1, 2, 3, 4])
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hsi(magnetic_field, reference_magnetic_field)

    def test_invalid_reference_magnetic_field_shape(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0])
        with self.assertRaises(ValueError):
            twostep_hsi(magnetic_field, reference_magnetic_field)

    def test_invalid_max_iterations(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hsi(magnetic_field, reference_magnetic_field, max_iterations=-1)

    def test_invalid_measurement_noise_std(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            twostep_hsi(magnetic_field, reference_magnetic_field, measurement_noise_std=-0.1)

    def test_input(self):
        magnetic_field = TEST_MAGNETIC_FIELD
        reference_magnetic_field = np.array([[0.227207, 0.051796, 0.411731]])
        hard_iron, soft_iron, corrected_magnetic_field = twostep_hsi(magnetic_field, reference_magnetic_field)

        self.assertEqual(hard_iron.shape, (3,))
        self.assertEqual(soft_iron.shape, (3, 3))
        self.assertEqual(corrected_magnetic_field.shape, magnetic_field.shape)
