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

        train_size = int(len(X)*0.8)
        X_train = X[:train_size]
        Y_train = Y[:train_size]
        X_test = X[train_size:]
        Y_test = Y[train_size:]

        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

        return X_train, Y_train, X_test, Y_test

    def build_model(self,seq_length):
        model = Sequential([
            LSTM(units = 50,return_sequences=True,input_shape=(seq_length,1)),
            Dropout(0.2),
            LSTM(units = 50, return_sequences = False),
            Dense(units = 1)
        ])
        model.compile(optimizer='adam',loss='mean_squared_error')
        self.model = model
        return model
    
    def train_model(self,X_train, Y_train, epochs = 50,batch_size =32):
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        history = self.model.fit(
            X_train,Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split = 0.1,
            verbose = 1
        )
        return history
    
    def predict_future(self,days =30):
        if self.model is None:
            raise ValueError("Model not trained yet")

        seq_length =60
        last_seq = self.stock_data['Close'].values[-seq_length:]

        last_seq_scaled = self.scaler.transform(last_seq.reshape(-1,1))

        predictions=[]
        current_sequence = last_seq_scaled.copy()

        for _ in range(days):
            current_reshaped = current_sequence.reshape(1,seq_length,1)
            next_day_scaled = self.model.predict(current_reshaped)
            predictions.append(next_day_scaled[0,0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_day_scaled

        predictions_rescaled = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1,1)
        )

        return predictions_rescaled.flatten()

    def main():
        