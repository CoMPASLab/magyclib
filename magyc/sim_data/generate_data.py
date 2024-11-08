"""
MAGYC - Synthetic Data Generator

This module generates synthetic data for the calibration of magnetometers and
gyroscopes. The data is generated using a constant magnetic vector that is
rotated in three different degrees of motion: low, mid, and high. The created
data is saved in the folderPath directory as a pickle (.pkl) file. The file's
name uses the date of its creation as: yyyymmdd_hhmm.pkl

The data provided is: magnetic field (m), magnetic field with added noise (mt),
magnetic field with HSI (mm), magnetic field with HSI and noise (mmt), angular
rates (w), angular rates with noise (wt), angular rates with gyroscope bias (wm),
angular rate with gyroscope bias and noise (wmt), attitude (rph), time (t),
ground truth magnetic vector (m0), soft iron (si), hard iron (hi), and gyroscope
bias (wb).

For each level of movement the data is provided as a multi-dimensional array,
where the shape is: (simulation, samples, 3).

Functions:
    create_synthetic_dataset: Creates a synthetic dataset using a constant
    magnetic vector

Classes:
    _SyntheticData: Synthetic dataset generator

Authors: Sebastián Rodríguez-Martínez and Giancarlo Troni
Contact: srodriguez@mbari.org
"""
import pickle
from datetime import datetime
from os.path import isdir
from pathlib import Path
from typing import Dict, List, Union
import navlib.math as nm

import numpy as np


# TODO: Add test for create_synthetic_dataset
def create_synthetic_dataset(folder_path: Path, niter: int = 100, nsamples: int = 10000, frequency: float = 25.0,
                             mag_noise_G: float = 0.01, gyro_noise_rad_s: float = 0.005, random: bool = False) -> None:
    """
    Creates a synthetic dataset using a constant magnetic vector that is
    randomly rotated in three different degrees of motion: low, mid, and high.
    The created data is saved in the folderPath directory as a pickle (.pkl)
    file. The file's name uses the date of its creation as: yyyymmdd_hhmm.pkl

    The data provided is: magnetic field (m), magnetic field with added noise
    (mt), magnetic field with HSI (mm), magnetic field with HSI and noise (mmt),
    angular rates (w), angular rates with noise (wt), angular rates with gyroscope
    bias (wm), angular rate with gyroscope bias and noise (wmt), attitude (rph),
    time (t), ground truth magnetic vector (m0), soft iron (si), hard iron (hi),
    and gyroscope bias (wb).

    For each level of movement the data is provided as a multi-dimensional array,
    where the shape is: (simulation, samples, 3).

    Args:
        folder_path (Path): Folder to save the data as a pickle (.pkl) file.
        niter (int): Number of simulations per motion level, by default 100
        nsamples (int): Number of samples per simulation, by default 10000
        frequency (float): Simulated sensor frequency, by default 25
        mag_noise_G (float): Noise for the magnetic measurements in G.
        gyro_noise_rad_s (float): Noise for the gyroscope measurements in rad/s
        random (bool): If True, the soft iron, hard iron, and gyroscope biases
            are randomly generated. If False, the soft iron, hard iron, and
            gyroscope biases are fixed, by default False.
    """
    if not random:
        # Adimensional and positive definite symmetric (PDS) matrix
        SI = np.array([[1.10, 0.10, 0.04],
                       [0.10, 0.88, 0.02],
                       [0.04, 0.02, 1.22]])
        SI *= 1.8
        # G
        HI = np.array([[0.020],
                       [0.120],
                       [0.090]])
        HI *= 1.8
        # rad/s
        WB = np.array([[0.004],
                       [-0.005],
                       [0.002]])
        WB *= 1.8

    # Check if folder is a directory
    if not isdir(folder_path):
        raise ValueError("The provided folder is not a directory in the system")

    # Create dataset generator object
    sd = _SyntheticData(niter=niter, nsamples=nsamples, frequency=frequency, mag_noise_G=mag_noise_G,
                        gyro_noise_rad_s=gyro_noise_rad_s)

    # Set calibration parameters
    if random:
        sd.random_parameters()
    else:
        sd.fixed_parameters(SI, HI, WB)

    # Generate Dataset
    sd.generate_data(folder_path)


class _SyntheticData():
    def __init__(
        self,
        niter: int = 100,  # Number of simulations per motion level
        nsamples: int = 10000,  # Number of samples per simulation
        frequency: float = 25,  # Sensor frequency in Hz
        mag_noise_G: float = 0.001,  # Magnetic noise in Gauss
        gyro_noise_rad_s: float = 0.005,  # Gyroscope noise in rad/s
    ) -> None:
        """
        Synthetic dataset.

        Args:
            niter (int): Number of simulations per motion level, by default 100
            nsamples (int): Number of samples per simulation, by default 10000
            frequency (float): Simulated sensor frequency, by default 25
            mag_noise (float): Noise for the magnetic measurements in G.
            gyro_noise (float): Noise for the gyroscope measurements in rad/s
        """
        self._niter = niter
        self._nsamples = nsamples
        self._frequency = frequency
        self._mag_noise = mag_noise_G
        self._gyro_noise = gyro_noise_rad_s
        self._data_dict = dict()

    ####################################################################################################################
    # Properties                                                                                                       #
    ####################################################################################################################

    @property
    def data_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Data dictionary"""
        return self._data_dict

    @property
    def motion_levels(self) -> List[str]:
        """Motion level for the data dictionaries"""
        return ["high", "mid", "low", "tiny", "cross"]

    @property
    def initial_dict(self) -> Dict[str, np.ndarray]:
        """Initial zero matrices to initialize the keys"""
        n = self.niter
        m = self.nsamples
        dshape = (n, m, 3)
        tshape = (n, m, 1)
        shape = {
            "m0": (n, 3, 1),
            "m": dshape,
            "mt": dshape,
            "mm": dshape,
            "mmt": dshape,
            "w": dshape,
            "wt": dshape,
            "wm": dshape,
            "wmt": dshape,
            "rph": dshape,
            "t": tshape,
            "si": (n, 3, 3),
            "hi": (n, 3, 1),
            "wb": (n, 3, 1),
        }
        initial_dict = dict(zip([*shape.keys()], [np.zeros(s) for s in [*shape.values()]]))
        return initial_dict

    @property
    def niter(self) -> float:
        """Number of simulations per motion level"""
        return self._niter

    @property
    def nsamples(self) -> int:
        """Number of samples per simulation level"""
        return self._nsamples

    @property
    def mag_noise(self) -> float:
        """Magnetometer measurements noise in mG"""
        return self._mag_noise

    @property
    def gyro_noise(self) -> float:
        """Gyroscope measurements noise in rad/s"""
        return self._gyro_noise

    @property
    def timestep(self) -> float:
        """Time step between samples in each simulation"""
        return 1 / self._frequency

    @property
    def simulation_length(self) -> float:
        """Simulation length in seconds"""
        return self.timestep * self.nsamples

    @property
    def magnetic_vector(self) -> np.ndarray:
        """Magnetic vector as a (3,1) array in G"""
        return np.array([[227.207], [51.796], [411.731]]) / 1000

    @property
    def motion_limits(self) -> List[List[float]]:
        """
        Nested list with the amplitude of the sinusoidal function that models RPH
        for each level of motion, where the first row is high motion, the second
        is mid motion, the third is low motion and the last is tiny motion.
        Each nested list has the RPH amplitude as [A_roll, A_pitch, A_heading].
        """
        high_motion = np.deg2rad([5, 45, 360]).tolist()
        mid_motion = np.deg2rad([5, 5, 360]).tolist()
        low_motion = np.deg2rad([5, 45, 90]).tolist()
        tiny_motion = np.deg2rad([5, 5, 90]).tolist()
        cross_motion = np.deg2rad([5, 80, 180]).tolist()
        return [high_motion, mid_motion, low_motion, tiny_motion, cross_motion]

    @property
    def HI(self) -> np.ndarray:
        """Hard Iron as a (3,1) array"""
        return self._HI

    @property
    def SI(self) -> np.ndarray:
        """Soft Iron as a (3,3) array"""
        return self._SI

    @property
    def Wb(self) -> np.ndarray:
        """Gyroscope bias as a (3,1) array"""
        return self._Wb

    ####################################################################################################################
    # Properties setter                                                                                                #
    ####################################################################################################################

    @data_dict.setter
    def data_dict(self, value: List[Union[str, Dict[str, np.ndarray]]]) -> None:
        """Gets a dictionary for a motion level"""
        self._data_dict[value[0]] = value[1]

    @HI.setter
    def HI(self, value: np.ndarray) -> None:
        """Gets the HI as a (3,1), or (1,3), or (3,) array"""
        self._HI = value.reshape(3, 1)

    @SI.setter
    def SI(self, value: np.ndarray) -> None:
        """Gets the SI as a (3,3), or (9,) array"""
        self._SI = value.reshape(3, 3)

    @Wb.setter
    def Wb(self, value: np.ndarray) -> None:
        """Gets the Wb as a (3,1), or (1,3), or (3,) array"""
        self._Wb = value.reshape(3, 1)

    ####################################################################################################################
    # Methods                                                                                                          #
    ####################################################################################################################

    def random_parameters(self) -> None:
        """
        Generate random parameters for the soft iron, hard iron, and gyroscope
        biases.
        """
        # Soft Iron
        SI_diagonal = np.random.normal(1.0, 0.1, (3,))
        SI_offdiagonal = np.abs(1 - np.random.normal(1.0, 0.05, (3,)))
        self.SI = np.array(
            [
                [SI_diagonal[0], SI_offdiagonal[0], SI_offdiagonal[1]],
                [SI_offdiagonal[0], SI_diagonal[1], SI_offdiagonal[2]],
                [SI_offdiagonal[1], SI_offdiagonal[2], SI_diagonal[2]],
            ]
        )

        # Hard Iron
        self.HI = np.random.normal(0.0, 0.08, (3, 1))

        # Gyroscope Bias
        self.Wb = np.random.normal(0.0, 0.02, (3, 1))

    def fixed_parameters(self, SI: np.ndarray, HI: np.ndarray, Wb: np.ndarray) -> None:
        """
        Generate fixed parameters for the soft iron, hard iron, and gyroscope
        biases.

        Args:
            SI (np.ndarray): Soft Iron positive definite matrix as a (3, 3) numpy array.
            HI (np.ndarray): Hard Iron in G as a (3, 1) numpy array.
            Wb (np.ndarray): Gyroscope bias in rad/s as a (3, 1) numpy array.
        """
        # Check that the inputs are numpy arrays
        if not isinstance(SI, np.ndarray):
            raise ValueError("SI must be a numpy array")
        if not isinstance(HI, np.ndarray):
            raise ValueError("HI must be a numpy array")
        if not isinstance(Wb, np.ndarray):
            raise ValueError("Wb must be a numpy array")
        # Check if SI is PSD
        if any(np.linalg.eigvals(SI) < 0) or not np.allclose(SI, np.transpose(SI)):
            raise ValueError("SI must be a PDS matrix")

        self.SI = SI
        self.HI = HI
        self.Wb = Wb

    def save_data(self, folder_path: Path):
        """
        Saves the synthetic data as a pickle file in the folder path with the
        name yyyymmdd_hhmm.pkl.

        Args
            folder_path (Path): Folder to save the data as a pickle (.pkl) file.
        """
        if not isdir(folder_path):
            raise ValueError("The provided directory is not in the system.")

        file_name = Path(datetime.now().strftime("%Y%m%d_%H%M") + ".pkl")
        with open(folder_path / file_name, "wb") as handle:
            pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_data(self, folder_path: str):
        """
        Creates a synthetic dataset using a constant magnetic vector that is
        randomly rotated in three different degrees of motion: low, mid, and high.
        The created data is saved in the folderPath directory as a pickle (.pkl)
        file. The file's name uses the date of its creation as: yyyymmdd_hhmm.pkl

        The data provided is: magnetic field (m), magnetic field with added noise
        (mt), magnetic field with HSI (mm), magnetic field with HSI and noise (mmt),
        angular rates (w), angular rates with noise (wt), angular rates with gyroscope
        bias (wm), angular rate with gyroscope bias and noise (wmt), attitude (rph),
        time (t), ground truth magnetic vector (m0), soft iron (si), hard iron (hi),
        and gyroscope bias (wb).

        For each level of movement the data is provided as a multi-dimensional array.
        For example, for the magnetic field, for any motion level the provided array's
        shape is: (simulation, samples, 3).

        Args
            folder_path (str): Folder to save the data as a pickle (.pkl) file.
        """
        for motion_level in range(len(self.motion_levels)):
            print(f"Creating {self.motion_levels[motion_level]} movement data.")

            # Create data dictionary
            data = dict(self.initial_dict)

            # Set kinematic conditions based on the following. The angular rates
            # of an ROV are defined based on the following ranges:
            # Heading rate: 0.2 to 0.4 rad/s
            # Pitch rate: 0.1 to 0.3 rad/s
            # Roll rate: 0.05 to 0.08 rad/s
            # If for an axis i we model the movement as:
            # r_i = A_i * sin((w_i/A_i) * t + phi_i), where A_i is the amplitude,
            # w_i is the angular rate, and phi_i is the phase shift, then the
            # angular rate is modelled as its derivative:
            # w_i = w_i * cos((w_i/A_i) * t + phi_i)

            # Amplitude ranges
            r_A, p_A, h_A = self.motion_limits[motion_level]
            # horizontal shift
            r_phi = np.random.uniform(-np.pi, np.pi, (self.niter,))
            p_phi = np.random.uniform(-np.pi, np.pi, (self.niter,))
            h_phi = np.random.uniform(-np.pi, np.pi, (self.niter,))
            # Angular rate
            r_w = np.random.uniform(0.05, 0.08, (self.niter,))
            p_w = np.random.uniform(0.1, 0.3, (self.niter,))
            h_w = np.random.uniform(0.2, 0.4, (self.niter,))
            # time
            t = np.linspace(0, self.simulation_length, self.nsamples)

            # Start simulations loop
            for simulation in range(self.niter):
                if simulation % 10 == 0:
                    header = f"Motion Level: {self.motion_levels[motion_level]}"
                    print(f"{header} | Simulation {simulation} our of {self.niter}.")

                # Roll, pitch and heading
                r = r_A * np.sin((r_w[simulation] / r_A) * t + r_phi[simulation])
                p = p_A * np.sin((p_w[simulation] / p_A) * t + p_phi[simulation])
                h = h_A * np.sin((h_w[simulation] / h_A) * t + h_phi[simulation])
                rph = np.array([r, p, h]).T

                # Rotation Matrices
                rot_mat = np.apply_along_axis(nm.rph2rot, 1, rph)
                rot_mat_t = np.einsum("ijk->ikj", rot_mat)

                # Magnetic field
                m = (rot_mat_t @ self.magnetic_vector).squeeze()

                # compute angular rates in the sensor frame
                rot_ij = (rot_mat_t[:-1, :, :] @ rot_mat[1:, :, :]).reshape(-1, 9)
                skew_w = np.apply_along_axis(lambda x: nm.matrix_log3(x.reshape(3, 3)), 1, rot_ij).reshape(-1, 9)
                w_prime = np.apply_along_axis(lambda x: nm.so3_to_vec(x.reshape(3, 3)).reshape(3, 1), 1, skew_w)
                w = ((1 / np.diff(t)).reshape(-1, 1, 1) * w_prime).reshape(-1, 3)
                w = np.concatenate([np.atleast_2d(w[0]), w], axis=0)

                # Measurements Noise
                mNoise = np.random.randn(self.nsamples, 3) * self.mag_noise
                wNoise = np.random.randn(self.nsamples, 3) * self.gyro_noise

                # add HSI and Wb
                mm = ((self.SI @ m.T) + self.HI).T
                wm = (w.T + self.Wb).T

                # Add data to dictionary
                data["m"][simulation, :, :] = m
                data["mt"][simulation, :, :] = m + mNoise
                data["mm"][simulation, :, :] = mm
                data["mmt"][simulation, :, :] = mm + mNoise
                data["w"][simulation, :, :] = w
                data["wt"][simulation, :, :] = w + wNoise
                data["wm"][simulation, :, :] = wm
                data["wmt"][simulation, :, :] = wm + wNoise
                data["rph"][simulation, :, :] = rph
                data["t"][simulation, :, :] = t.reshape(-1, 1)
                data["m0"][simulation, :, :] = self.magnetic_vector
                data["si"][simulation, :, :] = self.SI
                data["hi"][simulation, :, :] = self.HI
                data["wb"][simulation, :, :] = self.Wb

            self.data_dict = [self.motion_levels[motion_level], data]
            print(f"{header} | Completed.")

        self.save_data(folder_path)
        print("Synthetic Dataset Completed.")
