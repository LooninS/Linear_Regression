# Linear Regression from Scratch

A simple implementation of **linear regression from scratch** using only:

- [NumPy](https://numpy.org/) for numerical computation.
- [Matplotlib](https://matplotlib.org/) for visualization.

The project does not use pandas, scikit-learn, or any other machine-learning library. The model learns the relationship between years of experience and salary using gradient descent.

## Project Overview

The model predicts salary using the following equation:

$$
\hat{y} = wx + b
$$

where:

- \(x\) is the number of years of experience.
- \(\hat{y}\) is the predicted salary.
- \(w\) is the learned weight or slope.
- \(b\) is the learned bias or intercept.

The parameters are learned by minimizing the mean squared error using gradient descent.

## Dataset

The project expects a file named `train.csv` with the following structure:

```csv
YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
2.0,43525.00
2.2,39891.00
```

The dataset contains two columns:

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



## Cost Function and Gradients

The model uses the following cost function:

$$
J(w, b) =
\frac{1}{2m}
\sum_{i=1}^{m}
\left(wx_i + b - y_i\right)^2
$$

The gradients are:

$$
\frac{\partial J}{\partial w} =
\frac{1}{m}
\sum_{i=1}^{m}
\left(wx_i + b - y_i\right)x_i
$$

$$
\frac{\partial J}{\partial b} =
\frac{1}{m}
\sum_{i=1}^{m}
\left(wx_i + b - y_i\right)
$$

The parameters are updated after every iteration:

$$
w \leftarrow
w - \alpha \frac{\partial J}{\partial w}
$$

$$
b \leftarrow
b - \alpha \frac{\partial J}{\partial b}
$$



## Making Predictions

After training, predictions can be generated using:

```python
experience = 7.5
predicted_salary = final_w * experience + final_b

print(f"Predicted salary: {predicted_salary:.2f}")
```

The prediction follows:

$$
\hat{y} = wx + b
$$

## Important Notes

- The feature name must be exactly `YearsExperience`.
- The target column must be exactly `Salary`.
- The learning rate is small because salary values are relatively large.
- A learning rate that is too large can make the cost diverge.
- A learning rate that is too small can make training slow.
- This implementation uses batch gradient descent.
- This project is intended for learning and does not include regularization, train/test splitting, or production-level data validation.

## Learning Outcomes

This project demonstrates:

- Reading structured data without pandas.
- Representing a linear model mathematically.
- Implementing batch gradient descent.
- Using NumPy arrays for numerical computation.
- Visualizing model predictions with Matplotlib.

## License

This project is available for educational and personal use.
