# Loading the data into Pandas DataFrame object. Remember to use public URLs to read the file.
import pandas as pd
import numpy as np

# define column names manually
column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
                "density", "pH", "sulphates", "alcohol", "quality"]

df  = pd.read_csv("https://raw.githubusercontent.com/uy-seng/cs4372/main/assignment-1/dataset/winequality.csv", skiprows=1, names=column_names, delimiter=";")

    
x = df.drop(columns=["quality"]).values
y = df["quality"].values

# standardlize the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

# 80/20 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(x_train, y_train)

sgd.score(x_test, y_test)