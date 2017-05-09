#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data, wb
from matplotlib.dates import DateFormatter
import datetime

start = datetime.date(2012,1,1)
end = datetime.date(2012,11,11)
daysFmt = DateFormatter('%m-%d-%Y')
alibaba = data.DataReader('000001', 'yahoo', start, end)
alibaba['Adj Close'].plot(legend=True, figsize=(10,4))
plt.show()