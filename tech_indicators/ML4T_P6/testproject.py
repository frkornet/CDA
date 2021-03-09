import pandas as pd
import numpy as np
from util import get_data
import matplotlib.pyplot as plt	

from indicators import BBP, plot_BBP, RSI, plot_RSI, WilliamsR, plot_WilliamsR, \
     MFI, plot_MFI, MACD, plot_MACD, plot_SPY_JPM

from TheoreticallyOptimalStrategy import testPolicy, build_order_dataframe, \
    build_benchmark_orders, print_stats

from marketsimcode import compute_portvals

def author():
    return "fkornet3"

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

    optimal = testPolicy(sym, sd, ed, sv)
    # print("Optimal strategy:")
    # print("=================")
    # print(optimal)
    # print('')

    odf = build_order_dataframe(optimal)
    # print("Optimal order dataframe:")
    # print("========================")
    # print(odf)
    # print('')

    optimal_values = compute_portvals(orders_file=odf, start_val=sv, 
        commission=0.0, impact=0.0)
    # print("Optimal portfolio values:")
    # print("=========================")
    # print(optimal_values)
    # print('')

    odf = build_benchmark_orders("JPM", sd, ed)
    # print("Benchmark order dataframe:")
    # print("==========================")
    # print(odf)
    # print('')

    benchmark_values = compute_portvals(orders_file=odf, start_val=sv, 
        commission=0.0, impact=0.0)
    # print("Benchmark portfolio values:")
    # print("===========================")
    # print(benchmark_values)
    # print('')

    ts = "Optimal Strategy versus Benchmark (JPM)"
    optimal_values = optimal_values / optimal_values[0]
    ax = optimal_values.plot(label="Optimal Strategy", color="red")

    benchmark_values = benchmark_values / benchmark_values[0]
    benchmark_values.plot(label="Benchmark", color="green", ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Return")
    ax.legend(loc='upper left')
    plt.savefig("optimal.png")

    print('')
    print_stats("Benchmark Hold JPM:", benchmark_values)
    print('')

    print_stats("Optimal Strategy JPM:", optimal_values)
    print('')

    print('Done')