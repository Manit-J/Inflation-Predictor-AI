# Imports 
import pandas as pd #imports panda library
from sklearn.model_selection import train_test_split # 
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("INDINF_CPI.csv")

x = dataset.drop(columns=["INDINF_CPI_COMMON_Q"])
y = dataset["INDINF_CPI_COMMON_Q"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1)

clf = LinearRegression()
clf.fit(x_train, y_train)
print(clf.predict(x_test))
print(clf.score(x_test, y_test))

