# MAGYC: Magnetometer and Gyroscope Calibration

The goal of this library is to provide a set of tools for the calibration of Attitude and Heading Reference System (AHRS) magnetometers and gyroscopes. The proses of calibration consist of determine the scale and non-orthogonality vectors for the magnetometer, soft-iron (SI), and the biases for the gyroscope and the magnetometer, hard-iron (HI).

To solve the calibration problem, this library provides a set of least squares and factor graph method that need the magnetometer and gyroscope measurements, and the timestamp for each one of this samples. As both measurement are from the same device, the timestamp will be the same for both. This library was developed in the context of a research publication in the IEEE Journal of Oceanic Engineering. In this library the user can find the methods developed for this research under the MAGYC: Magnetometer and Gyroscope Calibration novel approach, and as well the benchmark methods implemented.

## Dependencies

Poetry will be used for dependencies handling. For further details on the required dependencies, please refer to the `pyproject.toml` file.

## Build

To isolate this library's use in the local environment, this project is built with [Poetry](https://python-poetry.org/docs/). If you do not have poetry installed on your machine, you can follow the steps in this [tutorial](https://python-poetry.org/docs/). Once you have poetry installed, you can follow the next steps.

If you want to use the library as a pip module, you can build the project with the following command:

``` bash
poetry build
```

This will create a `dist` directory with the built `magical` package. Then, you can install the built package with the following command:

``` bash
pip3 install dist/magyclib-<VERSION>-py3-none-any.whl
```

## Development

If you want to install the library for development purposes, you can install the required packages in the virtual environment with the following command inside the repository:

``` bash
poetry install
```

> If you are installing the library through an SSH server, there might be a problem related to the python keyring. To avoid this problem, run `keyring --disable` before the last command in your terminal.

If you want to avoid the development, documentation and test packages, you can use the following command:

```bash
poetry install --without dev docs test
```

Now, to run and modify the code, you have to spawn a new shell with the environment activated:

```bash
poetry shell
```

### VSCode Development

If you use VSCode as your development tool, you need to consider that the `python interpreter` must be set to the virtual environment activated for this project. By default, `poetry` saves the virtual environment in the directory: `/home/<user>/.cache/pypoetry/virtualenvs/`. To have this directory available for our interpreter selection, you need to open VSCode preferences (`ctrl + ,`) and search for `venv`. You will find `Python: Venv Folders` and `Python: Venv Path`. You must add the virtual environments directory path in both sections, including the `/` at the end.

Once the path is added, you can select the `python interpreter` in the lower right corner for the `python scripts` and the upper right corner for the `python notebooks`.

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/). To create it, you will need to install the project development dependencies:

```bash
poetry install --no-root
```

Then, run the following command:

```bash
mkdocs build
```

By default, this will create a `site/` directory with the built documentation in html format. However, if you want to build the documentation and serve it locally, you can run the following command:

```bash
mkdocs serve
```

Then, navigate to [http://localhost:8000/](http://localhost:8000/) to view the documentation.
