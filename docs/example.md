# Simulated Dataset Calibration

We provide two examples for calibration and evaluation of the MAGYC and benchmark methods as jupyter notebooks. The notebooks are available in the `examples` directory, and to use them is neccesary to build [poetry](https://python-poetry.org/) as:

```bash
poetry build
```
Then, you will have all the required dependencies installed.

## MAGYC Example - Simulated Data Calibration & Self Evaluation

[Jupyter Notebook :simple-jupyter:](https://github.com/CoMPASLab/magyclib/blob/main/example/20241107_sim_calibration.ipynb){ .md-button }

This notebook demonstrates how to use the MAGYC algorithms to calibrate a magnetometer and gyroscope using simulated data and provides a comparison with benchmark methods for calibration. Then, the results are self evaluated.

The calibration dataset corresponds to the simulated data used in the paper: "Full Magnetometer and Gyroscope Bias Estimation Using Angular Rates: Theory and Experimental Evaluation of a Factor Graph-Based Approach" by S. Rodríguez-Martínez and G. Troni, 2024. The dataset is available on Google Drive.

## MAGYC Example - Simulated Cross-Validation

[Jupyter Notebook :simple-jupyter:](https://github.com/CoMPASLab/magyclib/blob/main/example/20241107_sim_evaluation.ipynb){ .md-button }

This notebook evaluates the MAGYC and benchmark algorithms formagnetometer and gyroscope calibration using simulated data. The calibration dataset corresponds to the simulated data used in the paper: "Full Magnetometer and Gyroscope Bias Estimation Using Angular Rates: Theory and Experimental Evaluation of a Factor Graph-Based Approach" by S. Rodríguez-Martínez and G. Troni, 2024. The dataset is available on Google Drive.
