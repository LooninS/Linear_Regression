# Linear Regression from Scratch

A simple implementation of **linear regression from scratch** using only:

- [NumPy](https://numpy.org/) for numerical computation.
- [Matplotlib](https://matplotlib.org/) for visualization.

The project does not use scikit-learn or any machine-learning library. The model learns the relationship between years of experience and salary using gradient descent.

## Project Overview

The model fits a straight line to the training data:

\[
\hat{y} = wx + b
\]

where:

- `x` is the number of years of experience.
- `\hat{y}` is the predicted salary.
- `w` is the learned weight or slope.
- `b` is the learned bias or intercept.

The parameters are learned by minimizing the mean squared error using gradient descent.

## Dataset

The project expects a file named `train.csv` with the following structure:

```csv
YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
2.0,43525.00
```

The complete dataset contains two columns:

- `YearsExperience`: The input feature.
- `Salary`: The target value to predict.

## Requirements

- Python 3.8 or newer.
- NumPy.
- Matplotlib.

Install the dependencies with:

```bash
python -m pip install numpy matplotlib
```

## Project Structure

```text
.
├── train.csv
├── linear_regression.py
└── README.md
```

## Loading the Dataset

Pandas is not required. NumPy reads the CSV file and uses the first row as the field names:

```python
import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(
    "train.csv",
    delimiter=",",
    names=True,
    dtype=float
)

x_train = data["YearsExperience"]
y_train = data["Salary"]
```

## Gradient Descent

The model uses the mean squared error cost function:

\[
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(wx_i + b - y_i)^2
\]

The gradients are:

\[
\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(wx_i+b-y_i)x_i
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(wx_i+b-y_i)
\]

The parameters are updated after every iteration:

\[
w := w - \alpha\frac{\partial J}{\partial w}
\]

\[
b := b - \alpha\frac{\partial J}{\partial b}
\]

Here, `alpha` is the learning rate and `m` is the number of training samples.

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(
    "train.csv",
    delimiter=",",
    names=True,
    dtype=float
)

x_train = data["YearsExperience"]
y_train = data["Salary"]


def gradient_function(x, y, w, b):
    m = len(x)

    dc_dw = 0.0
    dc_db = 0.0

    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]

        dc_dw += error * x[i]
        dc_db += error

    return dc_dw / m, dc_db / m


def gradient_descent(x, y, alpha, iterations):
    w, b = 0.0, 0.0

    for _ in range(iterations):
        dc_dw, dc_db = gradient_function(x, y, w, b)

        w -= alpha * dc_dw
        b -= alpha * dc_db

    return w, b


learning_rate = 0.000001
iterations = 20000

final_w, final_b = gradient_descent(
    x_train,
    y_train,
    learning_rate,
    iterations
)

print(f"w: {final_w:.4f}")
print(f"b: {final_b:.4f}")


x_values = np.linspace(
    np.min(x_train),
    np.max(x_train),
    100
)

y_values = final_w * x_values + final_b

plt.scatter(x_train, y_train, label="Training data")
plt.plot(x_values, y_values, color="red", label="Regression line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
```

## Running the Project

Place `train.csv` in the same directory as the Python file, then run:

```bash
python linear_regression.py
```

The program will:

1. Load the training data.
2. Initialize `w` and `b` to zero.
3. Calculate the gradients.
4. Update the parameters using gradient descent.
5. Print the learned weight and bias.
6. Display the training points and fitted regression line.

## Making Predictions

After training, predictions can be generated using:

```python
experience = 7.5
predicted_salary = final_w * experience + final_b

print(f"Predicted salary: {predicted_salary:.2f}")
```

## Important Notes

- The feature name must be exactly `YearsExperience`.
- The target column must be exactly `Salary`.
- The learning rate is deliberately small because salary values are relatively large.
- A learning rate that is too large can make the cost diverge.
- A learning rate that is too small can make training slow.
- This implementation is intended for learning and does not include regularization, train/test splitting, or production-level data validation.

## Learning Outcomes

This project demonstrates:

- Reading structured data without pandas.
- Representing a linear model mathematically.
- Calculating derivatives manually.
- Implementing gradient descent.
- Using NumPy arrays for numerical computation.
- Visualizing model predictions with Matplotlib.

## License

This project is available for educational and personal use.