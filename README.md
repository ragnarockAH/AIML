# House Price Prediction with NumPy, Pandas and Linear Regression

A beginner-friendly ML project: clean and explore a house-price dataset (Task 1), then predict prices with Linear Regression (Task 2).

**Colab notebook:** https://colab.research.google.com/drive/1U686YTF5N7aV69ikIc05xjqR8zLOyX2B
**Data file:** `house_prices.csv`

## 1. Dataset
A **synthetic dataset of 500 houses** generated with NumPy (fixed random seed, so it is reproducible). About 5% of the values in `area`, `bathrooms` and `age` were removed at random (25 each) to practise handling missing data.

| Column | Description |
|---|---|
| `area` | House size in square feet |
| `bedrooms` | Number of bedrooms (1-5) |
| `bathrooms` | Number of bathrooms (1-4) |
| `age` | Age of the house in years (0-50) |
| `price` | Sale price (the target we predict) |

## 2. Steps followed

### Task 1: Cleaning and exploration
1. **Load and inspect:** checked shape (500 x 5), column types, and `describe()` statistics.
2. **Missing values:** found 25 missing values each in `area`, `bathrooms` and `age`, and filled them:
   - `area`: **mean** (continuous, and mean ≈ median here, so the choice makes almost no difference)
   - `age` and `bathrooms`: **median** (not affected by extreme values; for bathrooms it gives a real whole number)
   - `price` (target) was complete and never filled.
3. **NumPy operations:** converted columns to arrays and computed price per square foot with vectorised division (about 151 on average).
4. **Statistics:** mean, median, std, min, max, and a correlation matrix.
5. **Grouping:** `groupby("bedrooms")` for average price and area.
6. **Plots:** price histogram, area vs price scatter, correlation heatmap.

### Task 2: Linear Regression
1. **Features (X):** `area`, `bedrooms`, `bathrooms`, `age`. **Target (y):** `price`.
2. **Split:** 80% training (400 rows), 20% test (100 rows), `random_state=42`.
3. **Train** a scikit-learn `LinearRegression`.
4. **Evaluate** on the unseen test set with MAE, RMSE and R².
5. **Predict** a new house and plot actual vs predicted prices and residuals.

## 3. Results

### Key findings from exploration
- **Area is the strongest predictor of price** (correlation ≈ 0.90).
- **Bedrooms are strongly related to price** (≈ 0.76), but they are also correlated with area (≈ 0.71), so part of the bedroom effect is really an area effect. Average price rises from about 182k (1 bedroom) to 446k (5 bedrooms).
- **Age has a weak negative relationship with price** (≈ -0.14): older houses are slightly cheaper.

### Model coefficients
| Feature | Coefficient | Meaning (other features held fixed) |
|---|---|---|
| area | ≈ +104.5 | each extra sq ft adds about 104 to the price |
| bedrooms | ≈ +12,413 | each extra bedroom adds about 12.4k |
| bathrooms | ≈ +12,697 | each extra bathroom adds about 12.7k |
| age | ≈ -934 | each extra year of age lowers the price by about 934 |
| intercept | ≈ 44,229 | starting point of the line; has no real-world meaning alone |

### Performance (test set)
| Metric | Value | What it says |
|---|---|---|
| MAE | ≈ 20,600 | predictions are off by about 20.6k on average (roughly 8% of the mean price) |
| RMSE | ≈ 26,600 | slightly larger than MAE, so a few predictions are noticeably worse |
| R² | ≈ 0.86 | the model explains about 86% of the variation in price on unseen houses |

A baseline that always guesses the average price has an MAE of about 58.7k, so the model cuts the error by roughly two-thirds. Training R² (0.88) and test R² (0.86) are close, so the model is not overfitting.

### Example prediction
A **1,200 sq ft, 3-bedroom house** (assuming 2 bathrooms, 10 years old) is predicted at about **222,900**, with a typical error range of roughly 196k to 249k (± 1 RMSE).

## 4. Limitations
- The data is synthetic, so real-world factors such as location and condition are not represented.
- Bedrooms and area are correlated (multicollinearity), so individual coefficients, especially for bedrooms, should be interpreted carefully.
- A linear model cannot capture curved or interaction effects.

## 5. How to run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook AWS_SBG_AIML.ipynb
```
Or open the Colab link above and choose **Runtime → Run all**.

## 6. What I would explore next
Real datasets with location features, tree-based models such as Random Forests, and cross-validation for more reliable evaluation.
