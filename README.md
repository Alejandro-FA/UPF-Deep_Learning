# UPF-Deep_Learning
Repository for sharing lab projects of Universitat Pompeu Fabra (UPF) Deep Learning course.

## Environment configuration

It is very recommended to use a python virtual environment for all the labs of the subject. There are 2 recommended environments for doing so, [`miniconda`](https://docs.conda.io/en/latest/miniconda.html#) or [`venv`](https://docs.python.org/3/library/venv.html). Below you can find the steps for both types of environments:

### Environment configuration using `venv`

1. First open a terminal in the labs folder and create a virtual environment:

```bash
python3 -m venv .venv
```

2. Then activate the environment. From now on everything that we install will remain inside the `.venv` folder, without polluting the user installation.

```bash
source .venv/bin/activate
```

3. You can now start working normally! Remember to **select the python interpreter** appripriately. You can do so in Visual Studio Code using the toolbar for `.py` files:

![](assets/Screenshot%202023-04-20%20at%2013.51.38.png)

Or using the `Select Kernel` option for Jupyter Notebooks:

![](assets/Screenshot%202023-04-20%20at%2013.52.49.png)

4. It is recommended to install some basic DL packages as well:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy google pandas matplotlib opencv-python torch
```

## Installing tensorflow

To install tensorflow, you can follow the instructions [here](https://www.tensorflow.org/install/pip#step-by-step_instructions).





