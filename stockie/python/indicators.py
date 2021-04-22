#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This file contains all the technical indicator function calls

# Relative Strength Index (RSI)

import numpy as np
import pandas as pd

def RSI_Frank(df, window):
    rsi=df['Close'].copy()
    rsi[:]=np.nan

    change = df.Close - df.Close.shift(1)
    change = change.fillna(0)

    gain = change.copy()
    gain [ gain <= 0] = 0
    
    loss = change.copy()
    loss [ loss > 0] = 0
    loss = np.abs(loss)
    
    Avg_gain = gain.fillna(0).rolling(window).mean()
    Avg_loss = loss.fillna(0).rolling(window).mean()
    rsi = 100 * (Avg_gain / (Avg_gain + Avg_loss))

    return rsi    

def RSI_Harshad(df, window):
    Gain=df['Close'].copy()
    Loss=df['Close'].copy()
    Avg_gain=df['Close'].copy()
    Avg_loss=df['Close'].copy()
    rsi=df['Close'].copy()

    Gain[:]=0.0
    Loss[:]=0.0
    Avg_gain[:]=0.0
    Avg_loss[:]=0.0
    rsi[:]=np.nan

    for i in range(1,len(df)):
        if df.Close.iloc[i] > df.Close.iloc[i-1]:
            Gain[i]=df.Close.iloc[i]-df.Close.iloc[i-1]
        else:
            # For loss save the absolute value on loss
            Loss[i]=abs(df.Close.iloc[i]-df.Close.iloc[i-1])
        if i>window:
            Avg_gain[i]=(Avg_gain[i-1]*(window-1)+Gain[i])/window
            Avg_loss[i]=(Avg_loss[i-1]*(window-1)+Loss[i])/window
            rsi[i]=(100*Avg_gain[i]/(Avg_gain[i]+Avg_loss[i])).round(2)

    return rsi

def RSI(df, window):
    rsi=df['Close'].copy()
    rsi[:]=np.nan

    change = df.Close - df.Close.shift(1)
    change = change.fillna(0)

    gain = change.copy()
    gain [ gain <= 0] = 0

    loss = change.copy()
    loss [ loss > 0] = 0
    loss = np.abs(loss)

    avg_gain = gain[:window].mean()
    avg_loss = loss[:window].mean()

    i=window
    for g, l in zip(gain[window:], loss[window:]):
        avg_gain = (avg_gain * (window-1) + g) / window
        avg_loss = (avg_loss * (window-1) + l) / window
        rsi[i] = 100 * (avg_gain / (avg_gain + avg_loss) )
        i+=1

    return rsi

# Williams Percent Range (WPR)
def WPR(df, window):
    Highest_High = df['High'].rolling(window,min_periods=window).max()
    Lowest_Low   = df['Low'].rolling(window,min_periods=window).min()

    denom = (Highest_High - Lowest_Low).values
    denom[denom == 0] = 0.00001
    wpr = 100 * (df['Close'] - Highest_High) / denom

    return wpr

# Money Flow Index (MFI)
def MFI(df, window):
    
    typical_price = (df['High'] + df['Low'] + df['Close'])/3
    raw_money_flow = typical_price*df['Volume']

    idx = typical_price>typical_price.shift(1)

    pos_money_flow = raw_money_flow.copy()
    pos_money_flow.iloc[:] = 0.0
    pos_money_flow.loc[idx] = raw_money_flow.loc[idx]

    neg_money_flow = raw_money_flow.copy()
    neg_money_flow.iloc[:] = 0.0
    neg_money_flow.loc[~idx] = raw_money_flow.loc[~idx]

    mfi_pos = pos_money_flow.rolling(window).sum()
    mfi_neg = neg_money_flow.rolling(window).sum()

    denom = (mfi_pos+mfi_neg).values
    denom[denom == 0] = 0.00001
    mfi = 100 * mfi_pos / denom

    return mfi

# Bollinger Bands (BB)
def BBP(df, window):
    MA = df['Close'].rolling(window).mean()
    Std_Dev = df['Close'].rolling(window).std()

    BOLU=MA + 2 * Std_Dev
    BOLL=MA - 2 * Std_Dev

    denom = (BOLU - BOLL).values
    denom[denom == 0.0] = 0.00001
    bbp = ( df['Close'] - BOLL) / denom

    return bbp

