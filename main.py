import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self):
        self.stock_data = None
        self.symbol = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = None

    def fetch_stock_data(self, symbol, period = '2y'):
        self.symbol = symbol
        ticker = yf.Ticker(symbol)
        self.stock_data= ticker.history(period=period)
        return self.stock_data
    
    def prepare_data(self,sequence_length = 60):
        data = self.stock_data['close'].values.reshape(-1,1)
        scaled_data = self.scaler.fit_transform(data)
        X , Y = [],[]
        for i in range(sequence_length,len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i,0])
            Y.append(scaled_data[i,0])
        
        X = np.array(X)
        Y = np.array(Y)