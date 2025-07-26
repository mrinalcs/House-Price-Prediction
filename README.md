# House Prices Regression Analysis

This repository contains a Jupyter Notebook for analyzing and predicting house prices using regression models. Using the popular [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and performs data preprocessing, feature selections, and modeling to estimate property values.
 

## Covered
- Feature selection
- Handling missing values
- Feature scaling and encoding
- Simplifying categorical features
- Model training and evaluation:
  - Multiple Linear Regression (MLR)
  - Ridge Regression
  - Lasso Regression
- VIF (Variance Inflation Factor) analysis for multicollinearity
- Model comparison using RMSE, R², and Adjusted R²
- Streamlit price prediction app

## Conclusion

Two modeling approaches were applied:

- A manual multiple linear regression (MLR) using a small set of handpicked features.
- A full-feature model comparison using regularized regressions (Ridge and Lasso).

Key takeaways:

Lasso Regression achieved the best performance on the test data, with the lowest RMSE and highest R², demonstrating its ability to generalize well while reducing overfitting.

The manual MLR model, though simpler, performed reasonably well and offered interpretability, making it a good baseline.

Linear Regression on all features overfit the training data, while Ridge Regression improved stability by controlling coefficient magnitudes.

[![View Notebook](https://img.shields.io/badge/View-Notebook-blue?logo=github)](https://github.com/mrinalcs/House-Price-Prediction/blob/main/notebook.ipynb)
[![Open in Streamlit](https://img.shields.io/badge/Try%20it%20Live-Streamlit-brightgreen?logo=streamlit)](https://house-price-prediction-mrinalcs.streamlit.app/)
