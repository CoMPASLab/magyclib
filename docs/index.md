# Full Magnetometer and Gyroscope Bias Estimation using Angular Rates: Theory and Experimental Evaluation of a Factor Graph-Based Approach

**Authors:**

- [Sebastián Rodríguez-Martínez](https://scholar.google.com/citations?user=VITIKcsAAAAJ&hl=en) ([srodriguez@mbari.org](mailto:srodriguez@mbari.org)), Monterey Bay Aquarium Research Institute

- [Giancarlo Troni](https://scholar.google.com/citations?user=7nLHDsMAAAAJ&hl=en) ([gtroni@mbari.org](mailto:gtroni@mbari.org)), Monterey Bay Aquarium Research Institute

!!! info
    This work was published in the IEEE Journal of Oceanic Engineering. The paper is currently under publication process. The preprint version is available on arXiv.

---

[Paper :simple-arxiv:](#){ .md-button } [Code :simple-github:](https://github.com/CoMPASLab/magyclib){ .md-button } [Pypi :simple-python:](https://pypi.org/project/magyc/){ .md-button } [Sim Data :material-google-drive:](https://drive.google.com/file/d/1c5Y1y3PU0pYVrRuZwQGtYYlmj3X2twRm/view?usp=drive_link){ .md-button }

## Abstract

Despite their widespread use in determining system attitude, Micro-Electro-Mechanical Systems (MEMS) Attitude and Heading Reference Systems (AHRS) are limited by sensor measurement biases. This paper introduces a method called MAgnetometer and GYroscope Calibration (MAGYC), leveraging three-axis angular rate measurements from an angular rate gyroscope to estimate both the hard- and soft-iron biases of magnetometers as well as the bias of gyroscopes. We present two implementation methods of this approach based on batch and online incremental factor graphs. Our method imposes fewer restrictions on instrument movements required for calibration, eliminates the need for knowledge of the local magnetic field magnitude or instrument's attitude, and facilitates integration into factor graph algorithms for Smoothing and Mapping frameworks. We validate the proposed methods through numerical simulations and in-field experimental evaluations with a sensor onboard an underwater vehicle. By implementing the proposed method in field data of a seafloor mapping dive, the dead reckoning-based position estimation error of the underwater vehicle was reduced from 10% to 0.5% of the distance traveled.

!!! note "Related Work"

    This paper builds on the prior work of the authors, extending the foundational approach introduced in [Rodriguez and Troni (2024)](https://arxiv.org/abs/2410.13827), where a factor graph framework was proposed to estimate the full calibration of a three-axis magnetometer (including hard-iron and soft-iron effects) and a three-axis gyroscope using magnetometer and angular rate measurements. The primary advancement over the previous approach is the incorporation of system constraints directly into the model residual, leveraging the mathematical properties of the calibration parameters. This refinement enhances convergence and, consequently, improves post-calibration performance. Additionally, by applying the method to field data from mapping surveys, this work expands the evaluation to real-world navigation outcomes, demonstrating the robustness and applicability of the proposed method in operational settings.

## Acknowledgments

This work was supported by the David and Lucile Packard Foundation and FONDECYT-Chile under grant 11180907. The field experimental data used in this study were collected in oceanographic surveys conducted by the Monterey Bay Aquarium Research Institute, led by Chief Scientist Dr. David Caress.

## Bibtex

```bibtex
TBD
```
