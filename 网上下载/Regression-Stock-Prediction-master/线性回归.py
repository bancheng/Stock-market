import csv
import numpy as np
import pandas as pd
from sklearn import linear_model

dates=[]
prices=[]

df = pd.read_csv("goog.csv")
prices = df.values[:,1]
prices = list(prices)
i=0
while i<df.shape[0]:
    dates.append(int(str(df.values[:,0][i]).split('-')[0]))
    i=i+1
dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
prices = np.reshape(prices, (len(prices), 1))
linearmod=linear_model.LinearRegression()
linearmod.fit(dates,prices)
print linearmod.predict(29)[0][0], linearmod.coef_[0][0], linearmod.intercept_[0]

