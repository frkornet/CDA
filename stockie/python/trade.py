import pandas as pd
import numpy  as np
import gc; gc.collect()

from time                    import time
from tqdm                    import tqdm
from util                    import print_ticker_heading, get_stock_start, smooth, log, \
                                    get_current_day_and_time, open_logfile, \
                                    is_holiday, exclude_tickers, empty_dataframe
from scipy.signal            import argrelmin, argrelmax

from sklearn.linear_model    import LogisticRegression
from category_encoders       import WOEEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline        import make_pipeline, Pipeline
from sklearn.preprocessing   import KBinsDiscretizer, FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute          import SimpleImputer

from symbols                 import BUY, SELL, STOCKS_FNM, EXCLUDE_FNM, \
                                    FULL_TRADE_FNM, TRAIN_TRADE_FNM, \
                                    TEST_TRADE_FNM, BUY_FNM, LOGPATH, \
                                    EXCLUDE_SET, TRADE_PERIOD, \
                                    BUY_THRESHOLD, SELL_THRESHOLD, \
                                    TRADE_DAILY_RET, YFLOAD_PATH, \
                                    TRADE_COLS, TRADE_COL_TYPES

from indicators              import RSI, WPR, MFI, BBP

from itertools import chain, combinations

import warnings
warnings.filterwarnings("ignore")

def build_ticker_list():
    fnm = '/Users/frkornet/CDA/Project/fund_indicators/name_map.csv'
    name_map = pd.read_csv(fnm)
    name_map = name_map.loc[name_map.data == 1]
    tickers = name_map.ticker.unique().tolist()
    # np.random.seed(1234)
    # idx = np.random.choice(len(tickers), size=227, replace=False)
    # ts = np.array(tickers)[idx].tolist()
    return tickers

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
    # windows = [3, 5, 10, 15, 20, 30] #, 45, 60]
    windows = [ 10, 20, 30 ] 

    for i in windows:
        ma = data.Close.rolling(i).mean()
        if 'MACD' in comb:    
           data[f'MACD_{i}']    = ma - data.Close
        if 'PctDiff' in comb: 
           data[f'PctDiff_{i}'] = data.Close.diff(i)
        if 'StdDev' in comb:  
           data[f'StdDev_{i}']  = data.Close.rolling(i).std()

    # exclude_cols = [target, 'smooth', 'Close', 'Date', 'Volume', 'Dividends', 'Stock Splits'] 
    exclude_cols = [target, 'smooth', 'Close', 'Low', 'High', 'Open', 'Date', 'Volume', 'Dividends', 'Stock Splits'] 
    factor = data.Close.copy()
    for c in data.columns.tolist():
        if c in exclude_cols:
           continue
        data[c] = data[c] / factor

    for i in windows:
        if 'RSI' in comb:
           data[f'RSI_{i}']     = RSI(data, i) / 100
        if 'WPR' in comb:
           data[f'WPR_{i}']     = WPR(data, i) / 100
        if 'MFI' in comb:
           data[f'MFI_{i}']     = MFI(data, i) / 100
        if 'BBP' in comb:
           data[f'BBP_{i}']     = BBP(data, i)


    if 'P/E Ratio' in data.columns:
       if 'P/E Ratio' not in comb:
          log(f'Deleting P/E Ratio feature: comb={comb}')
          del data['P/E Ratio']
       else:
          log(f'Keeping P/E Ratio feature: comb={comb}')

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

    log(f"- Building model with features: {X_train.columns}")

    scaler    = StandardScaler()
    encoder   = WOEEncoder()
    binner    = KBinsDiscretizer(n_bins=5, encode='ordinal')
    objectify = FunctionTransformer(func=stringify, check_inverse=False, validate=False)
    imputer   = SimpleImputer(strategy='constant', fill_value=0.0)
    clf       = LogisticRegression(class_weight='balanced', random_state=42)

    pipe = make_pipeline(scaler, binner, objectify, encoder, imputer, clf)
    pipe.fit(X_train, y_train.values)

    test_signals = (pipe.predict_proba(X_test)  > threshold).astype(int)[:,1]
    return y_train.values, test_signals.copy()

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
    log(f'type(buy_signals) ={type(buy_signals)} len()={len(buy_signals)}')
    log(f'type(sell_signals)={type(sell_signals)} len()={len(sell_signals)}')
    
    buy_n_sell = np.zeros((len(buy_signals),), dtype=int)
    length     = len(buy_n_sell)
    i          = 0
    state      = SELL
    
    buy_ids  = np.where(buy_signals != 0)[0].tolist()
    sell_ids = np.where(sell_signals != 0)[0].tolist()
    log(f'merge_buy_n_sell_signals():')
    log(f'- buy_ids ={buy_ids} len()={len(buy_ids)}')
    log(f'- sell_ids={sell_ids} len()={len(sell_ids)}')

    while i < length:
        if state == SELL and buy_signals[i] == 1:
            state = BUY
            buy_n_sell[i] = 1
        
        elif state == BUY and sell_signals[i] == 1:
            state = SELL
            buy_n_sell[i] = 2
            #continue
        
        i = i + 1
    
    buy_n_sell_ids  = np.where(buy_n_sell != 0)[0].tolist()
    log(f'- buy_n_sell_ids ={buy_n_sell_ids} len()={len(buy_n_sell_ids)}')
    return buy_n_sell

def extract_trades(hist, buy_n_sell, start_at, ticker, verbose):
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

    #test_start_at = len(hist) - len(buy_n_sell)
    log(f'extract_trades():')
    log(f'- len(hist)={len(hist)}')
    log(f'- len(buy_n_sell)={len(buy_n_sell)}')
    log(f'- start_at={start_at}')
    #test_start_at = 0
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    buy_id = sell_id = -1

    for i, b_or_s in enumerate(buy_n_sell):
        
        if b_or_s == BUY:
            buy_id    = start_at + i
            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            
        if b_or_s == SELL:
            sell_id    = start_at + i
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

def pair_buy_n_sell_signals(min_ids, max_ids):

    length_i, length_j = len(min_ids), len(max_ids)
    bns_pairs = [] 
    i, j = 0, 0
    
    while i < length_i and j < length_j:
       # log(f'i={i} j={j} length_i={length_i} length_j={length_j} bns_pairs={bns_pairs}')
       # log(f'min_ids[i]={min_ids[i]} max_ids[j]={max_ids[j]}')
       if min_ids[i] >= max_ids[j]:
          j += 1
       else:
          bns_pairs.append((min_ids[i], max_ids[j]))
          i += 1
          j += 1
    
    return bns_pairs


def process_bns_pairs(hist, bns_pairs, verbose=False):

    range_min, range_plus = -3, 3
    min_ids, max_ids, drets = [], [], []
    for (s,e) in bns_pairs:
        if verbose: log(f's={s}, e={e}')
        start_values = [s+i for i in range(range_min, range_plus+1)
                        if s+i > 0 and s+i < len(hist)]
        if verbose: log(f'start_values={start_values}')
        end_values = [e+i for i in range(range_min, range_plus+1)
                        if s+i > 0 and s+i < len(hist)]
        if verbose: log(f'end_values={end_values}')

        best_s = best_e = best_dret = None
        for si in start_values:
            for ei in end_values:
                days = ei - si
                if verbose: log(f"- days={days}")
                if days < 5 or days > 50:
                   continue

                si_close = hist.Close.iloc[si]
                ei_close = hist.Close.iloc[ei]
                gain    = ei_close - si_close
                if verbose: log(f"- gain={gain}")
                if gain <= 0.0:
                   continue

                dret = (1 + gain/si_close) ** (1/days) - 1.0
                if verbose: log(f"- dret={dret}")
                if best_dret is None or dret > best_dret:
                   best_s, best_e, best_dret = si, ei, dret
                if verbose: log(f"- best_s={best_s} best_e={best_e} best_dret={best_dret}")

        if best_dret is not None and best_dret >= 0.003:
           if verbose: log(f"adding {best_s}, {best_e}, {best_dret} to lists")
           min_ids.append(best_s)
           max_ids.append(best_e)
           drets.append(best_dret)

    log(f'process_bns_pairs(): min_ids={min_ids}')
    log(f'process_bns_pairs(): max_ids={max_ids}')
    log(f'process_bns_pairs(): drets={drets}')
    return min_ids, max_ids 



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
        hist = features(hist, target)
        exclude_cols = [target, 'smooth', 'Close', 'Open', 'Low', 'High', 'Date', 'Volume', 'Dividends', 'Stock Splits'] 
        used_cols = [c for c in hist.columns.tolist() if c not in exclude_cols]
        X, y, X_train, X_test, y_train, y_test = split_data(hist, used_cols, target, 0.7)
        y_train_len = len(y_train)
        secs.append(time() - start_time)

        # pre-process data, create features, and split data
        start_time = time()
        step = "local minima/maxima"
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        del hist['smooth']
        bns_pairs = pair_buy_n_sell_signals(min_ids, max_ids)
        new_min_ids, new_max_ids = process_bns_pairs(hist, bns_pairs)
        log(f"- len(min_ids)      ={len(min_ids)}")
        log(f"- len(max_ids)      ={len(max_ids)}")
        log(f"- len(bns_pairs)    ={len(bns_pairs)}")
        log(f"- bns_pairs         ={bns_pairs}")
        log(f"- len(new_min_ids)  ={len(new_min_ids)}")
        log(f"- len(new_max_ids)  ={len(new_max_ids)}")
        min_ids, max_ids = new_min_ids, new_max_ids
        log(f"- min_ids           ={min_ids}")
        log(f"- max_ids           ={max_ids}")
        secs.append(time() - start_time)
    
    except:
        log(f"Failed to get possible trades for {ticker}")
        log(f"step={step}")
        return False, None, None, None, None

    # get the buy signals
    start_time = time()
    step = "get buy signals"
    hist[target] = 0
    hist[target].iloc[min_ids] = 1 
    y_train = hist[target].iloc[0:y_train_len].copy()
    if len(y_train.unique()) < 2:
       log(f'y_train only contains zeros, so nothing to train on!')
       log(f'Skipping training for {ticker}')
       return False, None, None, None, None

    train_buy_signals, test_buy_signals = \
            get_signals(X_train, y_train, X_test, BUY_THRESHOLD)
    # test_buy_signals += len(train_buy_signals)
    secs.append(time() - start_time)

    try:
        # get the sell signals
        start_time = time()
        step = "get sell signals"
        hist[target] = 0
        hist[target].iloc[max_ids] = 1
        y_train = hist[target].iloc[0:y_train_len].copy()
        train_sell_signals, test_sell_signals = \
            get_signals(X_train, y_train, X_test, SELL_THRESHOLD)
        # test_sell_signals += len(train_sell_signals)
        secs.append(time() - start_time)
        
        # merge the buy and sell signals
        start_time = time()
        step = "merge buy and sell signals"
        train_buy_signals_ids =np.where(train_buy_signals)
        train_sell_signals_ids=np.where(train_sell_signals)
        train_buy_n_sell = merge_buy_n_sell_signals(train_buy_signals, train_sell_signals)
        test_buy_n_sell  = merge_buy_n_sell_signals(test_buy_signals, test_sell_signals)
        secs.append(time() - start_time)
        
        # extract trades
        start_time = time()
        step = "extract trades"
        train_ticker_df,   _   = extract_trades(hist, train_buy_n_sell, 0, ticker, verbose)
        test_ticker_df, buy_df = extract_trades(hist, test_buy_n_sell, len(train_buy_n_sell), ticker, verbose)
        secs.append(time() - start_time)

        log(f'')
        log(f'len(hist)                 ={len(hist)}') 
        log(f'np.sum(train_buy_signals) ={np.sum(train_buy_signals)}')
        log(f'np.sum(train_sell_signals)={np.sum(train_sell_signals)}')
        log(f'np.sum(train_buy_n_sell)  ={np.sum(train_buy_n_sell)//3}')
        log(f'len(train_ticker_df)      ={len(train_ticker_df)}') 
        log(f'')
        log(f'np.sum(test_buy_signals)  ={np.sum(test_buy_signals)}')
        log(f'np.sum(test_sell_signals) ={np.sum(test_sell_signals)}')
        log(f'np.sum(test_buy_n_sell)   ={np.sum(test_buy_n_sell)//3}')
        log(f'len(test_ticker_df)       ={len(test_ticker_df)}') 
        log(f'')
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
        #   break
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

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def stats_table(df, days_cutoff, verbose=False):
    pos_df = df[['gain_pct', 'trading_days']].loc[df.gain_pct > 0].groupby(by='trading_days').count()
    pos_df.columns = ['pos_counts']
    loss_df = df[['gain_pct', 'trading_days']].loc[df.gain_pct < 0].groupby(by='trading_days').count()
    loss_df.columns = ['neg_counts']
    zero_df = df[['gain_pct', 'trading_days']].loc[df.gain_pct == 0].groupby(by='trading_days').count()
    zero_df.columns = ['zero_counts']
    temp = pd.concat([loss_df, pos_df, zero_df], join="outer").reset_index()
    cols = [ 'pos_counts', 'neg_counts', 'zero_counts' ]

    stats_cnt = np.zeros((3, 4))
    totals_col = np.zeros((4,))
    total_lt_xd = 0
    for i, num in enumerate(temp[cols].loc[temp.trading_days < days_cutoff].sum()):
        stats_cnt[0,i] = num
        stats_cnt[2,i] += num
    stats_cnt[0,3] = np.sum(stats_cnt[0,0:3])
    
    for i, num in enumerate(temp[cols].loc[temp.trading_days >= days_cutoff].sum()):
        stats_cnt[1,i] = num
        stats_cnt[2,i] += num
    stats_cnt[1,3] = np.sum(stats_cnt[1,0:3])  
    stats_cnt[2,3] = stats_cnt[0,3] + stats_cnt[1,3]
    
    stats_pct = (stats_cnt.copy() / stats_cnt[2,3]) * 100
    if verbose == False:
        return stats_cnt, stats_pct
    
    print('         \t pos     neg    zero    total')
    print('         \t======  ======  =====   ======')
    print(f'< {days_cutoff} days\t', end='')
    for i in range(4):
        print(f'{stats_cnt[0, i]:.0f}', end='\t')
        
    print(f'\n>= {days_cutoff} days\t', end='')
    for i in range(4):
        print(f'{stats_cnt[1, i]:.0f}', end='\t') 
    
    print('\n         \t------\t------\t-----\t------')
    print('         \t', end='')
    for i in range(4):
        print(f'{stats_cnt[2, i]:.0f}', end='\t')

    print('\n\n\n')
    print('         \t pos     neg    zero    total')
    print('         \t======  ======  =====   ======')
    print(f'< {days_cutoff} days\t', end='')
    for i in range(4):
        print(f'{stats_pct[0, i]:.2f}', end='\t')
        
    print(f'\n>= {days_cutoff} days\t', end='')
    for i in range(4):
        print(f'{stats_pct[1, i]:.2f}', end='\t') 
    
    print('\n         \t------\t------\t-----\t------')
    print('         \t', end='')
    for i in range(4):
        print(f'{stats_pct[2, i]:.2f}', end='\t')
    return stats_cnt, stats_pct

if __name__ == "__main__":

    log_fnm = "trade"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        run_type = 'single'
        if run_type == 'job':
           cols = ['MACD', 'PctDiff', 'StdDev', 'RSI', 'WPR', 'MFI', 'BBP', 'P/E Ratio']
           combs = []
           for p in powerset(cols):
               if len(p) > 2:
                  combs.append(p)
        else:
           combs = [ ('MACD', 'RSI', 'BBP') ]

        best_comb1, best_comb2, best_comb3    = None, None, None
        best_score1, best_score2, best_score3 = None, None, None
        print('Starting for loop', len(combs))
        for run, comb in enumerate(combs):
            # if run > 3:
            #   break

            print('')
            msg = f'{run:03d}/{len(combs)}: Running trade_main() with combination {comb}:'
            print(f'{msg}')
            print(f'='*len(msg))
            print('\n')
            trade_main()
            test_df = pd.read_csv(TEST_TRADE_FNM)
            stats_cnt, stats_pct = stats_table(test_df, 55)

            score1 = stats_pct[0,0]
            score2 = 100 * (stats_pct[0,0] / stats_pct[0,3])
            score3 = stats_pct[2,0]

            if best_score1 is None or score1 > best_score1:
               best_comb1  = comb
               best_score1 = score1

            if best_score2 is None or score2 > best_score2:
               best_comb2  = comb
               best_score2 = score2

            if best_score3 is None or score3 > best_score3:
               best_comb3  = comb
               best_score3 = score3
            
            print(f'best_comb1={best_comb1} best_score1={best_score1:.2f}')
            print(f'best_comb2={best_comb2} best_score2={best_score2:.2f}')
            print(f'best_comb3={best_comb3} best_score3={best_score3:.2f}')

        print(f'Best combination1 of features is : {best_comb1}')
        print(f'Best score1 is                   : {best_score1:.2f}') 

        print(f'Best combination2 of features is : {best_comb2}')
        print(f'Best score2 is                   : {best_score2:.2f}') 

        print(f'Best combination3 of features is : {best_comb3}')
        print(f'Best score3 is                   : {best_score3:.2f}') 

    print('')
    print('Done.')
    print('')
