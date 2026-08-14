# Linear Regression from Scratch

A small, practical machine-learning project that implements linear regression with NumPy and explores the complete workflow in a Jupyter notebook.

## Project Overview

This repository is a from-scratch implementation of linear regression. The goal is to understand the mechanics behind model training instead of treating the estimator as a black box.

The notebook walks through:

- Loading and inspecting a dataset.
- Preparing feature and target values.
- Implementing the hypothesis function.
- Defining a cost function.
- Computing gradients.
- Updating parameters with gradient descent.
- Evaluating predictions on the available data.
- Visualizing the fitted model and learning process.

## Repository Structure

```text
.
├── Linear_Regression.ipynb  # Main notebook
├── train.csv                # Training dataset
├── README.md                # Project documentation
└── .gitignore
```

## Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Getting Started

### Clone the repository

```bash
git clone https://github.com/LooninS/Linear_Regression.git
cd Linear_Regression
```

### Create an environment

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

### Run the notebook

```bash
jupyter notebook Linear_Regression.ipynb
```

You can also open the notebook in VS Code or JupyterLab.

## How It Works

For an input feature vector \(x\), the model predicts a value using:

\[
\hat{y} = wx + b
\]

where `w` is the model parameter and `b` is the bias. Training starts with initial parameter values and repeatedly applies gradient descent to reduce the cost:

\[
w := w - \alpha \frac{\partial J}{\partial w}, \qquad
b := b - \alpha \frac{\partial J}{\partial b}
\]

Here, `alpha` is the learning rate and `J` is the cost function.

The notebook keeps the implementation explicit so each part of the optimization loop can be inspected and modified.

## Dataset

The repository includes `train.csv`, which is used by the notebook. Check the notebook's loading and preprocessing cells for the expected column names and data format.

## Notes

This project is intended for learning and experimentation. For production work, compare the from-scratch implementation with a tested library implementation and add validation, error handling, and reproducible evaluation.

## Future Improvements

- Add a separate test split and report evaluation metrics.
- Compare batch and mini-batch gradient descent.
- Add regularization.
- Track and plot the cost across iterations.
- Add automated tests for the cost and gradient functions.
- Export learned parameters for later inference.

## License

No license has been specified yet. Add a license file before accepting external contributions or reusing the project publicly.

## Author

Maintained by [LooninS](https://github.com/LooninS).