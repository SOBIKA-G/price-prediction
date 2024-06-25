import pandas as pd
import numpy as np

d = pd.read_csv("D:\Python\Advertising.csv")
print(d.head())

print(d.isna().sum())

print(d.corr())

x = d.drop('Sales',axis = 1)
y = d[['Sales']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=32)

from sklearn.ensemble import RandomForestRegressor
rg = RandomForestRegressor(n_estimators = 100)
rg.fit(x_train,y_train)

from sklearn.metrics import mean_squared_error,r2_score
print("Mean squared Error - ",mean_squared_error(y_test,rg.predict(x_test)))
print("R square score - ",r2_score(y_test,rg.predict(x_test)))
