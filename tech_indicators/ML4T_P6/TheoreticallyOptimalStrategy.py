import numpy as np
import pandas as pd
from util import get_data
import datetime as dt
from marketsimcode import compute_portvals	
import matplotlib.pyplot as plt	  	   		     		  		  		    	 		 		   		 		  

def author():
    return "fkornet3"

def testPolicy(symbol = "AAPL", 
    sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), 
    sv = 100000):

    date_range = pd.date_range(sd,ed)
    syms=[symbol]
    pdf = get_data(syms, date_range)
    pdf['gain'] = pdf[symbol].shift(-1) - pdf[symbol] # tomorrow's close - today's close

    pdf['position'] = 0
    pdf.loc[pdf.gain > 0.0, 'position'] = 1000
    pdf.loc[pdf.gain < 0.0, 'position'] = -1000

    pdf['action'] = pdf.position - pdf.position.shift(1)
    sd = pdf.index.min()
    pdf.loc[sd,'action'] = pdf.loc[sd,'position']
    df = pd.DataFrame(pdf['action'])
    df.columns = [symbol]
    return df

def build_order_dataframe(df):
    sym = df.columns[0]
    idx = df[sym] == 0.0
    df = df.loc[~idx].copy()
    df['Symbol'] = sym
    df['Order'] = 'BUY'
    df.loc[df[sym] < 0.0, 'Order'] = 'SELL'
    df = df.rename(columns={sym : 'Shares'})
    df.Shares = np.abs(df.Shares)
    df['Date'] = df.index
    df = df.reset_index(drop=True)
    df = df[['Date', 'Symbol', 'Order', 'Shares']]
    return df

def build_benchmark_orders(sym, sd, ed):
    init = { 'Date': [sd, ed], 
        'Symbol': [sym, sym], 
        'Order': ['BUY', 'SELL'], 
        'Shares':[1000, 1000]}
    return pd.DataFrame(init)

def print_stats(message, values):
    df = pd.DataFrame(values, columns=["cum"])
    df['dr'] = df["cum"] / df["cum"].shift(1) - 1.0
                          
    print(message)
    print("="*len(message))

    cum_return   = df["cum"].iloc[-1] / df["cum"].iloc[0] - 1.0
    daily_std    = df["dr"].std()
    daily_mean   = df["dr"].mean()
    sharpe_ratio = daily_mean / daily_std # assumes risk free return is zero

    print(f"- cumulative return        : {cum_return:2.4f}")
    print(f"- daily standard deviation : {daily_std:2.4f}")
    print(f"- mean of daily returns    : {daily_mean:2.4f}")
    print(f"- Sharpe ratio             : {sharpe_ratio:2.4f}")

if __name__ == "__main__":  
    sd  = "2008-01-01"
    ed  = "2009-12-31"
    sv  = 100000
    sym = "JPM"

    pdf = get_data([sym], pd.date_range(sd, ed))
    sd = pdf.index.min()
    ed = pdf.index.max()
    # print("pdf:")
    # print("====")
    # print(pdf)
    # print('')

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