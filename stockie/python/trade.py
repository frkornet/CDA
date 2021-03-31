import pandas as pd
import numpy  as np
import gc; gc.collect()

from time                    import time
from tqdm                    import tqdm
from util                    import print_ticker_heading, get_stock_start, smooth, log, \
                                    get_current_day_and_time, open_logfile, \
                                    is_holiday, exclude_tickers, build_ticker_list, \
                                    empty_dataframe
from scipy.signal            import argrelmin, argrelmax

from sklearn.linear_model    import LogisticRegression
from category_encoders       import WOEEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline        import make_pipeline, Pipeline
from sklearn.preprocessing   import KBinsDiscretizer, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.impute          import SimpleImputer

from symbols                 import BUY, SELL, STOCKS_FNM, EXCLUDE_FNM, \
                                    FULL_TRADE_FNM, TRAIN_TRADE_FNM, \
                                    TEST_TRADE_FNM, BUY_FNM, LOGPATH, \
                                    EXCLUDE_SET, TRADE_PERIOD, \
                                    BUY_THRESHOLD, SELL_THRESHOLD, \
                                    TRADE_DAILY_RET, YFLOAD_PATH, \
                                    TRADE_COLS, TRADE_COL_TYPES

import warnings
warnings.filterwarnings("ignore")


def get_stock_n_smooth(ticker, period):
    """
    Copy of what is in util.py. Except this version reads what has been read 
    from yfinance and stored on file. The stored version is smoothed already, 
    and reading from disk should be much faster as it avoids the expensive 
    smoothing operation. The reading from file, will only return success if 
    there is at least 5 years worth of data to work with. 
    """
    gc.collect()
    try:
        hist = pd.read_csv(f'{YFLOAD_PATH}{ticker}.csv')
        hist.index = hist.Date.values
        del hist['Date']
        success = len(hist) > 5 * 252
        log(f'Successfully retrieved smoothed price data for {ticker} '+
            f'(len(hist)={len(hist)}, success={success})')
    except:
        hist = None
        success = False
        log(f'Failed to find {ticker}.csv in {YFLOAD_PATH}!')
    return success, hist


def features(data, target):
    """
    Given a standard yfinance data dataframe, add features that will help
    the balanced scorecard to recognize buy and sell signals in the data.
    The features are added as columns in the data dataframe. 
    
    The original hist dataframe from yfinance is provided, so we can copy
    the target to the data dataframe. The data dataframe with the extra 
    features is returned. The target argument contains the name of the 
    column that contains the the target.
    """
    windows = [3, 5, 10, 15, 20, 30, 45, 60]

    for i in windows:
        ma = data.Close.rolling(i).mean()
        # Moving Average Convergence Divergence (MACD)
        data['MACD_'+str(i)] = ma - data.Close
        data['PctDiff_'+str(i)] = data.Close.diff(i)
        data['StdDev_'+str(i)] = data.Close.rolling(i).std()

    exclude_cols = [target, 'smooth', 'Close', 'Date', 'Volume', 'Dividends', 'Stock Splits'] 
    factor = data.Close.copy()
    for c in data.columns.tolist():
        if c in exclude_cols:
           continue
        data[c] = data[c] / factor

    data = data.dropna()
    
    return data

def stringify(data):
    """
    Convert a Pandas dataframe with numeric columns into a dataframe with only
    columns of the data type string (in Pandas terminology an object). The 
    modified dataframe is returned. Note that the original dataframe is lost.
    """
    df = pd.DataFrame(data)
    for c in df.columns.tolist():
        df[c] = df[c].astype(str)
    return df

def split_data(stock_df, used_cols, target, train_pct):
    """
    Split data set into a training and test data set:
    - X contains the features for training and predicting. 
    - y contains the target for training and evaluating the performance.

    Used_cols contain the features (i.e. columns) that you want to use
    for training and prediction. Target contains the name of the column
    that is the target.

    Function returns X and y for cross validation, X and y for training, 
    and X and y for testing.
    """

    test_starts_at = int(len(stock_df)*train_pct)
    X = stock_df[used_cols]
    y = stock_df[target]

    X_train = stock_df[used_cols].iloc[:test_starts_at]
    X_test  = stock_df[used_cols].iloc[test_starts_at:]
    y_train = stock_df[target].iloc[:test_starts_at]
    y_test  = stock_df[target].iloc[test_starts_at:]

    return X, y, X_train, X_test, y_train, y_test

def get_signals(X_train, y_train, X_test, threshold):
    """
    Used to predict buy and sell signals. The function itself has no awareness
    what it is predicting. It is just a helper function used by 
    get_possible_trades().

    Target is the column that contains the target. The other columns are
    considered to be features to be used for training and prediction.

    The function uses a balanced weight of evidence scorecard to predict the 
    signals. It returns the signals array.

    Note that the function uses 70% for training and 30% for testing. The 
    date where the split happens is dependent on how much data the hist
    dataframe contains. So, the caller will not see a single split date for
    all tickers. 
    """

    encoder   = WOEEncoder()
    binner    = KBinsDiscretizer(n_bins=5, encode='ordinal')
    objectify = FunctionTransformer(func=stringify, check_inverse=False, validate=False)
    imputer   = SimpleImputer(strategy='constant', fill_value=0.0)
    clf       = LogisticRegression(class_weight='balanced', random_state=42)

    pipe = make_pipeline(binner, objectify, encoder, imputer, clf)
    pipe.fit(X_train, y_train.values)

    test_signals = (pipe.predict_proba(X_test)  > threshold).astype(int)[:,1]
    return y_train.values, test_signals

def merge_buy_n_sell_signals(buy_signals, sell_signals):
    """
    The functions will take two lists and produce a single list containing the 
    buy and sell signals. The merged list will always start with a buy signal.
    This is achieved by setting the state to SELL. That ensures that all sell 
    signals are quietly dropped until we get to the first buy signal.

    Note: this function does not enforce that each buy signal is matched with  
    a sell signal.

    The function implements a simple deterministic state machine that flips 
    from SELL to BUY and back from BUY to SELL.

    A buy in the merged list is 1 and a sell is 2. The merged list is
    returned to the caller at the end.
    """

    assert len(buy_signals) == len(sell_signals), "buy_signal and sell_signal lengths different!"
    
    buy_n_sell = [0] * len(buy_signals)
    length     = len(buy_n_sell)
    i          = 0
    state      = SELL
    
    while i < length:
        if state == SELL and buy_signals[i] == 1:
            state = BUY
            buy_n_sell[i] = 1
        
        elif state == BUY and sell_signals[i] == 1:
            state = SELL
            buy_n_sell[i] = 2
            #continue
        
        i = i + 1
    
    return buy_n_sell

def extract_trades(hist, buy_n_sell, ticker, verbose):
    """
    Given a merged buy and sell list, extract all complete buy and sell pairs 
    and store each pair as a trade in a dataframe (i.e. possible_trades_df). 
    
    The possible trades dataframe contains the ticker, buy date, sell date, 
    the close price at buy date, the close price at sell data, the gain 
    percentage, and the daily compounded return.

    Note that hist, contains the data from yfinance for the ticker, so we
    can calculate the above values to be stored in the possible trades
    dataframe.

    The function returns the possible trades dataframe for a single ticker
    to the caller.
    
    The function assumes that the buy_n_sell list is well formed and does 
    not carry out any checks. Since the list is typiclly created by 
    merge_buy_n_sell_signals(), this should be the case.

    TODO: extend the functionality so that the buy at the end without 
    a matching signal is storted in an open position dataframe. The caller
    is then responsible for merging all open position of all tickers into 
    a single dataframe.  
    """

    test_start_at = len(hist) - len(buy_n_sell)
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    buy_id = sell_id = -1

    for i, b_or_s in enumerate(buy_n_sell):
        
        if b_or_s == BUY:
            buy_id    = test_start_at + i
            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            
        if b_or_s == SELL:
            sell_id    = test_start_at + i
            sell_close = hist.Close.iloc[sell_id]
            sell_date  = hist.index[sell_id] 
            
            gain = sell_close - buy_close
            gain_pct = round( (gain / buy_close)*100, 2)
            
            trading_days = sell_id - buy_id
            
            daily_return = (1+gain_pct/100) ** (1/trading_days) - 1
            daily_return = round(daily_return * 100, 2)
            
            trade_dict = {'buy_date'    : [buy_date],  'buy_close'    : [buy_close],
                         'sell_date'    : [sell_date], 'sell_close'   : [sell_close],
                         'gain_pct'     : [gain_pct],  'trading_days' : [trading_days],
                         'daily_return' : [daily_return], 'ticker'    : [ticker] }
            possible_trades_df = pd.concat([possible_trades_df, 
                                           pd.DataFrame(trade_dict)])
    
    if verbose == True:
        log("****EXTRACT_TRADES****")
        log(possible_trades_df)
    
    if buy_id > 0 and buy_id > sell_id:
        buy_opportunity_df = {'ticker'    : [ticker] , 
                              'buy_date'  : [buy_date],  
                              'buy_close' : [buy_close],
                             }
        buy_opportunity_df = pd.DataFrame(buy_opportunity_df)
    else:
        cols=['ticker', 'buy_date', 'buy_close']
        buy_opportunity_df = pd.DataFrame(columns=cols)

    return possible_trades_df, buy_opportunity_df

def ticker_trades(ticker, verbose):
    target = 'target'
    gc.collect()
    if verbose == True:
        print_ticker_heading(ticker)

    success, hist = get_stock_n_smooth(ticker, TRADE_PERIOD)
    if success == False:
        return False, None, None, None, None

    secs = []
    try:

        # pre-process data, create features, and split data
        start_time = time()
        step = "pre-process"
        target = 'target'
        hist[target] = 0
        log("- calling features")
        hist = features(hist, target)
        log("- finished features() call")
        exclude_cols = [target, 'smooth', 'Close', 'Date', 'Volume', 'Dividends', 'Stock Splits'] 
        used_cols = [c for c in hist.columns.tolist() if c not in exclude_cols]
        log(f" - used_cols={used_cols}")
        X, y, X_train, X_test, y_train, y_test = split_data(hist, used_cols, target, 0.7)
        log("- finished split_data() call")
        y_train_len = len(y_train)
        secs.append(time() - start_time)

        # pre-process data, create features, and split data
        start_time = time()
        step = "local minima/maxima"
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        del hist['smooth']
        secs.append(time() - start_time)
        log(f"- len(min_ids)={len(min_ids)}")
        log(f"- len(max_ids)={len(max_ids)}")

        # get the buy signals
        start_time = time()
        step = "get buy signals"
        hist[target] = 0
        hist[target].iloc[min_ids] = 1 
        y_train = hist[target].iloc[0:y_train_len]
        train_buy_signals, test_buy_signals = \
            get_signals(X_train, y_train, X_test, BUY_THRESHOLD)
        secs.append(time() - start_time)

        # get the sell signals
        start_time = time()
        step = "get sell signals"
        hist[target] = 0
        hist[target].iloc[max_ids] = 1
        y_train = hist[target].iloc[0:y_train_len]
        train_sell_signals, test_sell_signals = \
            get_signals(X_train, y_train, X_test, SELL_THRESHOLD)
        secs.append(time() - start_time)
        
        # merge the buy and sell signals
        start_time = time()
        step = "merge buy and sell signals"
        train_buy_n_sell = merge_buy_n_sell_signals(train_buy_signals, train_sell_signals)
        test_buy_n_sell  = merge_buy_n_sell_signals(test_buy_signals, test_sell_signals)
        secs.append(time() - start_time)
        
        # extract trades
        start_time = time()
        step = "extract trades"
        train_ticker_df,   _   = extract_trades(hist, train_buy_n_sell, ticker, verbose)
        test_ticker_df, buy_df = extract_trades(hist, test_buy_n_sell, ticker, verbose)
        secs.append(time() - start_time)
     
        log(f'len(hist)                 ={len(hist)}') 
        log(f'np.sum(train_buy_signals) ={np.sum(train_buy_signals)}')
        log(f'np.sum(train_sell_signals)={np.sum(train_sell_signals)}')
        log(f'np.sum(test_buy_signals)  ={np.sum(test_buy_signals)}')
        log(f'np.sum(test_sell_signals) ={np.sum(test_sell_signals)}')
        log(f'len(train_ticker_df)      ={len(train_ticker_df)}') 
        log(f'len(test_ticker_df)       ={len(test_ticker_df)}') 
        log(f'len(buy_df)               ={len(buy_df)}') 
        log(f'ticker_trades(): secs={secs}')
 
        return True, train_ticker_df, test_ticker_df, buy_df, secs

    except:
        log(f"Failed to get possible trades for {ticker}")
        log(f"step={step}")
        return False, None, None, None, None
     
def get_possible_trades(tickers, threshold, period, verbose):
    """
    The main driver that calls other functions to do the work with the aim
    of extracting all possible trades for all tickers. For each ticker it
    performs the following steps:

    1) retrieve the historical ticker information (using yfinance),
    2) smooth the Close price curve,
    3) get the buy signals (using balanced scorecard),
    4) get the sell signals (using balanced scorecard),
    5) merge the buy and sell signals, and
    6) extract the possible trades and add that to the overall dataframe
       containing all possible trades for all tickers.

    The dataframe with all possible trades is then returned to the caller 
    at the end.
    """
    # print("tickers=", tickers)
    target = 'target'
    
    cols = TRADE_COLS
    train_possible_trades_df = empty_dataframe(cols, TRADE_COL_TYPES)
    test_possible_trades_df = empty_dataframe(cols, TRADE_COL_TYPES)

    cols=['ticker', 'buy_date', 'buy_close']
    col_types = [str, str, float]
    buy_opportunities_df = empty_dataframe(cols, col_types)
    
    #print('Determining possible trades...\n')
    tickers_ignored = 0
    ignored_l = []
    tot_secs = np.zeros((6,))
    counter = 0
    gc.collect()
    for ticker in tqdm(tickers, desc="possible trades: "):
        # if counter > 10:
        #    break
        counter += 1

        success, train_ticker_df, test_ticker_df, buy_df, secs = \
            ticker_trades(ticker, verbose)

        if success == True:
            train_possible_trades_df = pd.concat([train_possible_trades_df, 
                                                  train_ticker_df])
            test_possible_trades_df = pd.concat([test_possible_trades_df, 
                                                 test_ticker_df])
            buy_opportunities_df = pd.concat([buy_opportunities_df, buy_df])
            tot_secs += secs
        else:
           tickers_ignored += 1
           ignored_l.append(ticker)

        gc.collect()

    log(f'Tickers ignored count   : {tickers_ignored}')
    log(f'The ignored tickers are : {ignored_l}')
    log(f'tot_secs={tot_secs}\n')

    descrips = ["pre-process data", "local minima/maxima", "buy signals", 
                "sell signals", "merge signals", "extract trades"]
    for i, m in enumerate(descrips):
        tsec = int(tot_secs[i])
        tmin = tsec // 60
        tsec = tsec - tmin * 60
        log(f'Time spent on {descrips[i]}: {tmin:02d}:{tsec:02d}')

    train_possible_trades_df.trading_days = \
        train_possible_trades_df.trading_days.astype(int)
    test_possible_trades_df.trading_days  = \
        test_possible_trades_df.trading_days.astype(int)
    return train_possible_trades_df, test_possible_trades_df, \
           buy_opportunities_df

def call_get_possible_trades(tickers):
    log("Calling get_possible_trades...")
    train_possible_trades_df, test_possible_trades_df,  buy_opportunities_df \
       = get_possible_trades(tickers, 0.5, TRADE_PERIOD, False)
    log("Finished get_possible_trades")
    log("")
    return train_possible_trades_df, test_possible_trades_df, buy_opportunities_df

def drop_suspicious_trades(possible_trades_df):
    log("Checking for suspicious daily return trades...")
    idx = possible_trades_df.daily_return > TRADE_DAILY_RET
    log(f'possible_trades_df=\n{possible_trades_df[idx]}')
    if len(possible_trades_df[idx]) > 0:
        log(f'Dropping trades that exceed {TRADE_DAILY_RET}% daily return...')
    possible_trades_df = possible_trades_df[~idx]
    log('')
    return possible_trades_df


def save_files(train_possible_trades_df, test_possible_trades_df, buy_opportunities_df):

    possible_trades_df = pd.concat([train_possible_trades_df, 
                                    test_possible_trades_df])
    log(f'Saving all possible trades to {FULL_TRADE_FNM}'
        f' ({len(possible_trades_df)})')
    possible_trades_df.to_csv(FULL_TRADE_FNM, index=False)

    log(f'Saving train possible trades to {TRAIN_TRADE_FNM}'
        f' ({len(train_possible_trades_df)})')
    train_possible_trades_df.to_csv(TRAIN_TRADE_FNM, index=False)

    log(f'Saving test possible trades to {TEST_TRADE_FNM}'
        f' ({len(test_possible_trades_df)})')
    test_possible_trades_df.to_csv(TEST_TRADE_FNM, index=False)

    log(f"Saving buy opportunities to {BUY_FNM} ({len(buy_opportunities_df)})")
    buy_opportunities_df.to_csv(BUY_FNM, index=False)
    log('')

def trade_main():
    log("Generating possible trades")
    log('')

    tickers = build_ticker_list()
    train_possible_trades_df, test_possible_trades_df, buy_opportunities_df \
        = call_get_possible_trades(tickers)
    train_possible_trades_df = drop_suspicious_trades(train_possible_trades_df)
    test_possible_trades_df = drop_suspicious_trades(test_possible_trades_df)
    save_files(train_possible_trades_df, test_possible_trades_df, \
               buy_opportunities_df)
    log('Done.')

if __name__ == "__main__":

    log_fnm = "trade"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        trade_main()
