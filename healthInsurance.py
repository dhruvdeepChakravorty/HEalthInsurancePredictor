import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    data = pd.read_csv("data\\Insurance.csv")
    data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)
    X = data.drop("charges", axis=1)
    y = data["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-Squared Value:", r2)
    plt.scatter(y_test, y_pred, color="blue")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Actual vs. Predicted Health Insurance Charges")
    plt.show()
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

except FileNotFoundError as e:
    print(f"Error: The file was not found. Please check the file path. {e}")
except pd.errors.EmptyDataError as e:
    print(f"Error: The CSV file is empty. {e}")
except KeyError as e:
    print(f"Error: The column {e} is missing in the dataset.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
