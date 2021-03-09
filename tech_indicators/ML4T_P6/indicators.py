import pandas as pd
import numpy as np
from util import get_data
import matplotlib.pyplot as plt	

def author():
    return "fkornet3"

################################################
#                                              #
#     Bollinger Position Indicator (BBP))      #
#                                              #
################################################

def BBP(price, lookback):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band    = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    return bbp

def plot_BBP(col_to_plot, bbp, pdf):
    col_df = pdf[col_to_plot]
    col_df = col_df / col_df.iloc[0]
    col_df.columns = [ col_to_plot ]

    indicator = bbp[col_to_plot] - 1
    indicator.columns = [ col_to_plot ]

    ax = col_df.plot(label=col_to_plot + " Adj Close", color='green')
    s=col_to_plot + " (Normalized)"
    plt.text(0.4, 0.94, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    indicator.plot(label=col_to_plot + " RSI", color='blue')
    s = "BBP"
    plt.text(0.22, 0.3, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    h_upper = pdf.copy()
    h_upper[col_to_plot] = 0.0
    h_upper[col_to_plot].plot()

    h_lower = pdf.copy()
    h_lower[col_to_plot] = -1.0
    h_lower[col_to_plot].plot()

    ax.set_yticks([-1.0, 0.0, 1.5, 1.0, 0.5])
    plt.ylim(-1.2, 1.5)

    plt.savefig("BBP.png")
    plt.close()

################################################
#                                              #
#        Relative Strength Index (RSI)         #
#                                              #
################################################

def RSI_indicator_v1(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    for day in range(price.shape[0]):
        for sym in symbols:
            up_gain   = 0
            down_loss = 0

            for prev_day in range(day-lookback+1, day+1):
                delta = price.ix[prev_day, sym] - price.ix[prev_day-1, sym]

                if delta >= 0:
                    up_gain = up_gain + delta
                else:
                    down_loss = down_loss + (-1 * delta)

            if down_loss == 0:
                rsi.ix[day, sym] = 100
            else:
                rs = (up_gain / lookback) / (down_loss / lookback)
                rsi.ix[day, sym] = 100 - (100 / (1 + rs))

    # print(rsi)
    return rsi

def RSI_indicator_v2(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:]  = np.nan

    for day in range(price.shape[0]):
        for sym in symbols:
            up_gain   = 0
            down_loss = 0

            for prev_day in range(day-lookback+1, day+1):
                delta = daily_rets.ix[prev_day, sym]

                if delta >= 0:
                    up_gain = up_gain + delta
                else:
                    down_loss = down_loss + (-1 * delta)

            if down_loss == 0:
                rsi.ix[day, sym] = 100
            else:
                rs = (up_gain / lookback) / (down_loss / lookback)
                rsi.ix[day, sym] = 100 - (100 / (1 + rs))

    # print(rsi)
    return rsi

def RSI_indicator_v3(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:]  = np.nan

    for day in range(price.shape[0]):
        up_gain = daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets >= 0.0).sum()
        down_loss = -1 * daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets < 0.0).sum()

        for sym in symbols:
            if down_loss[sym] == 0:
                rsi.ix[day, sym] = 100
            else:
                rs = (up_gain[sym] / lookback) / (down_loss[sym] / lookback)
                rsi.ix[day, sym] = 100 - (100 / (1 + rs))

    # print(rsi)
    return rsi

def RSI_indicator_v4(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:]  = np.nan

    for day in range(price.shape[0]):
        if day < lookback:
            rsi.ix[day, :] = np.nan
            continue

        up_gain = daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets >= 0.0).sum()
        down_loss = -1 * daily_rets.ix[day-lookback+1:day+1,:].where(daily_rets < 0.0).sum()

        rs = (up_gain / lookback) / (down_loss / lookback)
        rsi.ix[day,:] = 100 - (100 / (1 + rs))

    rsi[rsi == np.inf] == 100
    # print(rsi)
    return rsi

def RSI_indicator_v5(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:]  = np.nan

    up_rets   = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = daily_rets[daily_rets <  0].fillna(0).cumsum() * -1.0

    up_gain = price.copy()
    up_gain.ix[:,:] = 0
    up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]

    down_loss = price.copy()
    down_loss.ix[:,:] = 0
    down_loss.values[lookback:,:] = down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]

    for day in range(price.shape[0]):
        # if day < lookback:
        #     rsi.ix[day, :] = np.nan
        #     continue

        up   = up_gain.ix[day,:]
        down = down_loss.ix[day,:]

        rs = (up / lookback) / (down / lookback)
        rsi.ix[day,:] = 100 - (100 / (1 + rs))

    rsi[rsi == np.inf] == 100
    # print(rsi)
    return rsi

def RSI_indicator_v6(price, lookback):
    symbols = price.columns
    rsi = price.copy()

    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:]  = np.nan

    up_rets   = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = daily_rets[daily_rets <  0].fillna(0).cumsum() * -1.0

    up_gain = price.copy()
    up_gain.iloc[:,:] = 0
    up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]

    down_loss = price.copy()
    down_loss.iloc[:,:] = 0
    down_loss.values[lookback:,:] = down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]

    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:lookback,:] = np.nan

    rsi[rsi == np.inf] == 100
    # print(rsi)
    return rsi

def RSI(price, lookback):
    return RSI_indicator_v6(price, lookback)

def plot_RSI(col_to_plot, rsi, pdf):
    col_df = pdf[col_to_plot]
    col_df = col_df / col_df.iloc[0]
    col_df.columns = [ col_to_plot ]

    indicator = rsi[col_to_plot]/100.0 - 1
    indicator.columns = [ col_to_plot ]

    ax = col_df.plot(label=col_to_plot + " Adj Close", color='green')
    s=col_to_plot + " (Normalized)"
    plt.text(0.4, 0.92, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    indicator.plot(label=col_to_plot + " RSI", color='blue')
    s = "RSI"
    plt.text(0.63, 0.16, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    h_upper = pdf.copy()
    h_upper[col_to_plot] = -0.3
    h_upper[col_to_plot].plot()

    h_lower = pdf.copy()
    h_lower[col_to_plot] = -0.7
    h_lower[col_to_plot].plot()

    ax.set_yticks([-0.7, -0.3, 1.5, 1.0, 0.5])
    plt.ylim(-1.0, 1.5)

    plt.savefig("RSI.png")
    plt.close()

################################################
#                                              #
#           Money Flow Index (MFI)             #
#                                              #
################################################

def MFI(vdf, hdf, ldf, pdf, lookback):
    tpdf = (hdf + ldf + pdf) / 3.0

    money_flow = tpdf * vdf
    money_ratio = tpdf.copy()
    money_ratio.iloc[:,:] = 0

    money_flow_positive = money_ratio.copy()
    money_flow_negative = money_ratio.copy()

    idx = tpdf < tpdf.shift(1)
    for i, s in enumerate(tpdf.columns):
        i_1d = idx.iloc[:,i]
        money_flow_positive[s].loc[i_1d] = money_flow[s].loc[i_1d]
        money_flow_negative[s].loc[~i_1d] = money_flow[s].loc[~i_1d]

    mfi_pos = money_flow_positive.rolling(window=lookback, min_periods=lookback).sum()
    mfi_neg = money_flow_negative.rolling(window=lookback, min_periods=lookback).sum()
    mfi = mfi_pos / (mfi_pos + mfi_neg)

    return mfi

def plot_MFI(col_to_plot, mfi, pdf):
    col_df = pdf[col_to_plot]
    col_df = col_df / col_df.iloc[0]
    col_df.columns = [ col_to_plot ]

    indicator = mfi[col_to_plot] - 1
    indicator.columns = [ col_to_plot ]

    ax = col_df.plot(label=col_to_plot + " Adj Close", color='green')
    s=col_to_plot + " (Normalized)"
    plt.text(0.4, 0.92, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    indicator.plot(label=col_to_plot + " RSI", color='blue')
    s = "MFI"
    plt.text(0.23, 0.16, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    h_upper = pdf.copy()
    h_upper[col_to_plot] = -0.2
    h_upper[col_to_plot].plot()

    h_lower = pdf.copy()
    h_lower[col_to_plot] = -0.8
    h_lower[col_to_plot].plot()

    ax.set_yticks([-0.8, -0.2, 1.5, 1.0, 0.5])
    plt.ylim(-1.0, 1.5)

    plt.savefig("MFI.png")
    plt.close()

################################################
#                                              #
#               Williams %R (%R)               #
#                                              #
################################################

def WilliamsR(hdf, ldf, cdf, lookback):
    hh = hdf.rolling(window=lookback, min_periods=lookback).max()
    ll = ldf.rolling(window=lookback, min_periods=lookback).min()
    wr = (cdf - hh) / (hh - ll) # + 1
    return wr

def plot_WilliamsR(col_to_plot, wr, pdf):
    col_df = pdf[col_to_plot]
    col_df = col_df / col_df.iloc[0]
    col_df.columns = [ col_to_plot ]

    indicator = wr[col_to_plot] # - 1
    indicator.columns = [ col_to_plot ]

    ax = col_df.plot(label=col_to_plot + " Adj Close", color='green')
    s=col_to_plot + " (Normalized)"
    plt.text(0.4, 0.92, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    indicator.plot(label=col_to_plot + " RSI", color='blue')
    s = "%R"
    plt.text(0.63, 0.16, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    h_upper = pdf.copy()
    h_upper[col_to_plot] = -0.2
    h_upper[col_to_plot].plot()

    h_lower = pdf.copy()
    h_lower[col_to_plot] = -0.8
    h_lower[col_to_plot].plot()

    ax.set_yticks([-0.8, -0.2, 1.5, 1.0, 0.5])
    plt.ylim(-1.0, 1.5)

    plt.savefig("WilliamsR.png")
    plt.close()


################################################
#                                              #
#  Moving Average Convergence Divergion (MACD) #
#                                              #
################################################

def MACD(fast_period, slow_period, signal_period, pdf):

    ema_slow = pdf.ewm(ignore_na=False, min_periods=slow_period, com=slow_period, adjust=True).mean()
    ema_fast = pdf.ewm(ignore_na=False, min_periods=fast_period, com=fast_period, adjust=True).mean()
    macd = ema_fast - ema_slow
    signal   = macd.ewm(ignore_na=False, min_periods=signal_period, com=signal_period, adjust=True).mean()
    macd = (macd - signal)

    return macd

def plot_MACD(col_to_plot, macd, pdf):
    col_df = pdf[col_to_plot]
    col_df = col_df / col_df.iloc[0]
    col_df.columns = [ col_to_plot ]

    indicator = macd[col_to_plot] - 1
    indicator.columns = [ col_to_plot ]

    ax = col_df.plot(label=col_to_plot + " Adj Close", color='green')
    s=col_to_plot + " (Normalized)"
    plt.text(0.4, 0.95, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    indicator.plot(label=col_to_plot + " RSI", color='blue')
    s = "MACD"
    plt.text(0.45, 0.4, s, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=12 )

    h_upper = pdf.copy()
    h_upper[col_to_plot] = -1.0
    h_upper[col_to_plot].plot()

    ax.set_yticks([-1.0, 1.5, 1.0, 0.5])
    plt.ylim(-2.0, 1.5)

    plt.savefig("MACD.png")
    plt.close()

################################################
#                                              #
#     S&P 500 (SPY) versus JP Morgan (JPM)     #
#                                              #
################################################

def plot_SPY_JPM(col_to_plot, pdf):
    col_df = pdf
    col_df = col_df / col_df.iloc[0]

    ax = col_df.plot(label=col_to_plot + " Adj Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")

    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    plt.ylim(0.5, 1.3)
    ax.legend(loc='upper center')
    plt.savefig("SPY_JPM.png")
    plt.close()

################################################
#                                              #
#              Main Test Program               #
#                                              #
################################################

if __name__ == "__main__":  
    sd  = "2008-01-01"
    ed  = "2009-12-31"
    sv  = 100000
    sym = "JPM"
    lookback = 14

    cdf = get_data([sym], pd.date_range(sd, ed), colname='Close')
    sd = cdf.index.min()
    ed = cdf.index.max()
    # print('')
    # print('Daily Close dataframe:')
    # print('======================')
    # print(cdf.tail(30))
    # print('')

    pdf = get_data([sym], pd.date_range(sd, ed))
    print('')
    print('Daily Price dataframe:')
    print('======================')
    print(pdf.tail(30))
    print('')

    plot_SPY_JPM(sym, pdf)
    adj_close_scalar = pdf / cdf

    vdf = get_data([sym], pd.date_range(sd, ed), colname='Volume')
    # print('')
    # print('Daily Volume dataframe:')
    # print('=======================')
    # print(vdf.tail(30))
    # print('')

    ldf = get_data([sym], pd.date_range(sd, ed), colname='Low')
    ldf *= adj_close_scalar
    # print('')
    # print('Daily Low dataframe:')
    # print('====================')
    # print(ldf.tail(30))
    # print('')

    hdf = get_data([sym], pd.date_range(sd, ed), colname='High')
    hdf *= adj_close_scalar
    # print('')
    # print('Daily High dataframe:')
    # print('=====================')
    # print(hdf.tail(30))
    # print('')

    bbp = BBP(pdf, lookback)
    print('BBP:')
    print("====")
    print('')
    print(bbp.tail(30))
    print('')
    plot_BBP(sym, bbp, pdf)

    # rsi = RSI_indicator_v1(pdf, lookback)
    # rsi = RSI_indicator_v2(pdf, lookback)
    # rsi = RSI_indicator_v3(pdf, lookback)
    # rsi = RSI_indicator_v4(pdf, lookback)
    # rsi = RSI_indicator_v5(pdf, lookback)
    # rsi = RSI_indicator_v6(pdf, lookback)
    rsi = RSI(pdf, lookback)
    print('RSI:')
    print("====")
    print('')
    print(rsi.tail(30))
    print('')

    plot_RSI(sym, rsi, pdf)

    mfi = MFI(vdf, hdf, ldf, pdf, lookback)
    print('MFI:')
    print("====")
    print('')
    print(mfi.tail(30))
    print('')
    plot_MFI(sym, mfi, pdf)

    wr = WilliamsR(hdf, ldf, pdf, lookback)
    print('Williams %R:')
    print("============")
    print('')
    print(wr.tail(30))
    print('')
    plot_WilliamsR(sym, wr, pdf)

    fast_period, slow_period, signal_period = 12, 26, 9
    macd = MACD(fast_period, slow_period, signal_period, pdf)
    print('MACD:')
    print("=====")
    print('')
    print(macd.tail(30))
    print('')
    plot_MACD(sym, macd, pdf)

    print('Done')