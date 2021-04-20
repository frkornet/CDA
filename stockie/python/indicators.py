#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This file contains all the technical indicator function calls

# Relative Strength Index (RSI)

import numpy as np
import pandas as pd

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
    
    Avg_gain = gain.fillna(0).rolling(14).mean()
    Avg_loss = loss.fillna(0).rolling(14).mean()
    rsi = 100 * (Avg_gain / (Avg_gain + Avg_loss))

    return rsi    

# Williams Percent Range (WPR)
def WPR(df, window):
    Highest_High = df['High'].rolling(window,min_periods=window).max()
    Lowest_Low   = df['Low'].rolling(window,min_periods=window).min()
    wpr=100 *(df['Close'] - Highest_High)/(Highest_High - Lowest_Low)

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
    mfi = 100 * mfi_pos / (mfi_pos+mfi_neg)

    return mfi

# Bollinger Bands (BB)
def BBP(df, window):
    MA=df['Close'].rolling(window).mean()
    Std_Dev=df['Close'].rolling(window).std()

    BOLU=MA+2*Std_Dev
    BOLL=MA-2*Std_Dev

    return ( df['Close'] - BOLL) / (BOLU - BOLL)

