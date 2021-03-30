import pandas as pd
import numpy  as np
import gc; gc.collect()

from tqdm                    import tqdm
from util                    import get_stock_start, smooth, log, \
                                    get_current_day_and_time, open_logfile, \
                                    build_ticker_list

from symbols                 import YFLOAD_PATH, LOGPATH

import warnings
warnings.filterwarnings("ignore")


def get_stock_n_smooth(ticker):
    """
    Copy of what is in util.py. This is done to ensure that we use the same
    period when we call get_stock_n_smooth(). The period which is fixed
    is 2011-02-13 and 2021-02-13.
    """
    start = '2011-02-13'
    end   = '2021-02-13'
    success, hist = get_stock_start(ticker, 2, start, end)
    if success == True:
       log(f'Smoothing price data for ticker {ticker}.')
       return smooth(hist, ticker)
    else:
       log(f'Failed to get price data for {ticker}!')
       return success, hist

def load_ticker(ticker):
    gc.collect()

    success, hist = get_stock_n_smooth(ticker)
    if success == True:
       fnm = f'{YFLOAD_PATH}{ticker}.csv'
       log(f'Saving loaded and smoothed price data for {ticker} to {fnm}.')
       hist.to_csv(fnm, header=True, index=True)
    else:
       log(f'- Unable to save smoothed historical prices for {ticker}')
     
def load_tickers(tickers):
    for ticker in tqdm(tickers, desc="possible trades: "):
        log(f'Loading and saving ticker: {ticker}')
        load_ticker(ticker)

def yfload_main():
    log("Loading historical smoothed prices...")
    log('')

    tickers = build_ticker_list()
    load_tickers(tickers)
    log('Done.')

if __name__ == "__main__":

    log_fnm = "yfload"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)
    yfload_main()
