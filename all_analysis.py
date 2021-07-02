#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import glob



files = glob.glob('ETFs/*.txt')
for file_names in files:
    df = pd.read_csv(file_names)
    df.rename(columns = {'Date' : 'ds', 'Close' : 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.drop(df.columns[[1, 2, 3, 5, 6]], axis=1, inplace=True)
    
    while True:
        X = df['y'].values
        result = adfuller(X)
        if result[0] < result[4]['5%']:
            break
        else:
            df['y'] = df['y'] - df['y'].shift(1)
            df.dropna(axis=0, inplace=True)
    
    model = Prophet()
    model.fit(df)
    future_days = model.make_future_dataframe(periods=365)
    prediction = model.predict(future_days)
    fig1 = model.plot(prediction)
    fig2 = model.plot_components(prediction)



files = glob.glob('Stocks/*.txt')
for file_names in files:
    df = pd.read_csv(file_names)
    df.rename(columns = {'Date' : 'ds', 'Close' : 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.drop(df.columns[[1, 2, 3, 5, 6]], axis=1, inplace=True)
    
    while True:
        X = df['y'].values
        result = adfuller(X)
        if result[0] < result[4]['5%']:
            break
        else:
            df['y'] = df['y'] - df['y'].shift(1)
            df.dropna(axis=0, inplace=True)
    
    model = Prophet()
    model.fit(df)
    future_days = model.make_future_dataframe(periods=365)
    prediction = model.predict(future_days)
    fig1 = model.plot(prediction)
    fig2 = model.plot_components(prediction)

