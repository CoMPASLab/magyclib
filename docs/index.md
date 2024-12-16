# MAGYC: Magnetometer and Gyroscope Calibration

[Code :simple-github:](https://github.com/CoMPASLab/magyclib){ .md-button } [Pypi :simple-python:](https://pypi.org/project/magyc/){ .md-button }

This site serves as documentation for the `magyc` library. The goal of this library is to provide a set of tools for the calibration of Attitude and Heading Reference System (AHRS) magnetometers and gyroscopes. The proses of calibration consist of determine the scale and non-orthogonality vectors for the magnetometer, soft-iron (SI), and the biases for the gyroscope and the magnetometer, hard-iron (HI).

To solve the calibration problem, this library provides a set of least squares and factor graph method that need the magnetometer and gyroscope measurements, and the timestamp for each one of this samples. As both measurement are from the same device, the timestamp will be the same for both. This library was developed in the context of a research publication in the IEEE Journal of Oceanic Engineering. In this library the user can find the methods developed for this research under the MAGYC: Magnetometer and Gyroscope Calibration novel approach, and as well the benchmark methods implemented.

The documentation for the different modules of this library can be found via the navigation bar.

If you use this library in your research, please cite the following publication:

```bibtex
@misc{rodríguezmartínez2024magnetometergyroscopebiasestimation,
      title={Full Magnetometer and Gyroscope Bias Estimation using Angular Rates: Theory and Experimental Evaluation of a Factor Graph-Based Approach},
      author={Sebastián Rodríguez-Martínez and Giancarlo Troni},
      year={2024},
      eprint={2412.09690},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.09690},
}
```
