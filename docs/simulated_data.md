# Monte Carlo Simulation

[Sim Data :material-google-drive:](https://drive.google.com/file/d/1c5Y1y3PU0pYVrRuZwQGtYYlmj3X2twRm/view?usp=drive_link){ .md-button }

A Monte Carlo numerical simulation was conducted to replicate 6,000 measurements from a MEMS AHRS during the sinusoidal motions of a vehicle. This simulation was designed to emulate a magnetometer calibration platform for an articulated vehicle or one with pitching capabilities in the Wide Angular Movement (WAM) dataset, a magnetometer calibration for a roll-and-pitch stable vehicle in the Mid Angular Movement (MAM) dataset, and a survey for a vehicle with the same capabilities as described in WAM for the Low Angular Movement (LAM) dataset.

Each experiment lasted 600 s, with simulated data generated at a 10 Hz rate and magnetometer measurements ($\sigma_{mag} = 10$ mG) and angular rate sensor ($\sigma_{gyro} = 10$ mrad/s) corrupted by Gaussian noise.

The true magnetic field vector is $\mathbf{m_0} = [227,\, 52, \,412]^T$ mG, the soft-iron upper triangular terms are given by $\mathbf{a} = [1.10,\, 0.10,\, 0.04,\, 0.88,\, 0.02,\, 1.22]^T$, the hard-iron bias is $\mathbf{m_b} = [20,\, 120,\, 90]^T$ mG, and the gyroscope bias is $\mathbf{w_b} = [4,\, -5,\, 2]^T$ mrad/s.

!!! note
    For further details on the simulation, please refer to the section `V. Numerical Simulation Evaluation` in the paper.
