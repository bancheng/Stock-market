import pandas as pd

df = pd.read_csv("goog.csv")
dates=[]
print(df.values.shape)
i=0
while i<df.shape[0]:
    dates.append(str(df.values[:,0][i]).split('-')[0])
    i=i+1
print(dates)
