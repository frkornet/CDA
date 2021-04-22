import pandas as pd
import numpy  as np
import gc; gc.collect()

from   pathlib    import Path
from   symbols    import NAME_MAP, YFLOAD_PATH, TRADE_PERIOD

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
        print(f'Successfully retrieved smoothed price data for {ticker} '+
            f'(len(hist)={len(hist)}, success={success})')
    except:
        hist = None
        success = False
        print(f'Failed to find {ticker}.csv in {YFLOAD_PATH}!')

    return success, hist


def get_pe(name, ticker):

    sub = pd.read_parquet(data_path / '2020_4' / 'parquet' / 'sub.parquet')
    corp = sub[sub.name == name].T.dropna().squeeze()

    # Get details on the forms
    corp_subs = pd.DataFrame()
    cik = corp.T.cik

    if type(cik) is int:
        cik_l = [ cik ]
    else:
        cik_l = list(corp.T.cik.unique())

    for sub in data_path.glob('**/sub.parquet'):
        sub = pd.read_parquet(sub)
        idx = (sub.cik.isin(cik_l)) & (sub.form.isin(['10-Q', '10-K']))
        corp_sub = sub.loc[idx]
        corp_subs = pd.concat([corp_subs, corp_sub])

    if len(corp_subs) == 0:
        print(f"No 10-Q or 10-K data found in SUB for {name} {ticker}!")
        return None
    
        # Get numeric details on the forms
    nums_file = data_path / 'PE_nums' / f'{ticker}_nums.parquet'
    if nums_file.exists():
        corp_nums = pd.read_parquet(nums_file)
    else:
        print("File not available, will process the individual num files instead")

        corp_nums = pd.DataFrame()
        for num in data_path.glob('**/num.parquet'):
            num = pd.read_parquet(num).drop('dimh', axis=1)
            corp_num = num[num.adsh.isin(corp_subs.adsh)]
            corp_nums = pd.concat([corp_nums, corp_num])

        assert len(corp_nums) != 0, "No rows in corp_nums!"
        corp_nums.ddate = pd.to_datetime(corp_nums.ddate, format='%Y%m%d')
        corp_nums.to_parquet(nums_file)

    # Retrieve the Earnings per share diluted
    # It only keeps the latest reported earnings.
    # Needs be refined over time as it is leaking information here.
    idx = (corp_nums.tag == 'EarningsPerShareDiluted') & (corp_nums.qtrs == 1)
    eps = corp_nums.loc[idx].drop('tag', axis=1)
    if len(eps) == 0:
        print("No quarterly reports with 'EarningsPerShareDiluted' available!")
        return None

    eps = eps[['adsh', 'ddate', 'value']].groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))
    eps.index = eps.ddate.values
    eps = eps[['ddate', 'value']]
    eps = eps.sort_values(by='ddate', ascending=True)

    # Get stock data and process splits and adjust reported earnings
    # corp_stock =yf.Ticker(ticker).history(start=str(eps.index.min())[:10], end=str(eps.index.max())[:10])
    success, corp_stock = get_stock_n_smooth(ticker, TRADE_PERIOD)
    if success == False or len(corp_stock) == 0:
        print(f"get_stock_n_smooth() did not return any rows!")
        return None
    
    splits = corp_stock[['Stock Splits']].loc[corp_stock['Stock Splits'] > 0]
    splits['Split_Date'] = splits.index

    for index, row in splits.sort_values(by='Split_Date', ascending=False).iterrows():
        split_date  = str(index)[:10]
        stock_split = row['Stock Splits']
        # print(split_date, stock_split)
        eps.loc[eps.ddate < split_date,'value'] = eps.loc[eps.ddate < split_date, 'value'].div(stock_split)

    eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
    eps = eps.rolling(window=4).sum()#.dropna()
    eps[eps == 0.0] = 0.00001

    # Calculate p/e ratio
    pe = corp_stock.Close.to_frame('price').join(eps.to_frame('eps'))
    pe = pe.fillna(method='ffill')#.dropna()
    pe['P/E Ratio'] = pe.price.div(pe.eps)
    return pe

name_map = pd.read_csv(NAME_MAP)
name_map = name_map[name_map.data == 1]
print(f'YFLOAD_PATH={YFLOAD_PATH}')

data_path = Path('/Users')
data_path = data_path / 'frkornet' / 'CDA' / 'Project' / 'fund_indicators' / 'data'

error_l = []
for ticker, subs_name in zip(name_map.ticker.tolist(), name_map.subs_name.tolist()):

    print(ticker, subs_name)
    success, hist = get_stock_n_smooth(ticker, '10y')
    assert success == True, 'Failed to retrieve stock data!'

    # Make sure P/E Ratio is not already there
    if 'P/E Ratio' in hist.columns:
       continue

    pe = get_pe(subs_name, ticker)

    if len(pe) > len(hist):
       print('len(pe) > len(hist):', end='')
       pe['Date'] = pe.index
       pe = pe.groupby(by='Date').max()#.reset_index()
       print(f' len(pe)={len(pe)}')
       print(pe)

    hist_new = hist.join(pe[['P/E Ratio']], how='inner')
    hist_new['Date'] = hist_new.index
    hist_new = hist_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'smooth', 'P/E Ratio']]
    print(f'len(hist_new)={len(hist_new)} len(hist)={len(hist)}')
    if len(hist_new) != len(hist):
       error_l.append(ticker)

    fnm = f'{YFLOAD_PATH}{ticker}.csv'
    print(f'Saving {ticker} -> {fnm} ...')
    print(hist_new)
    hist_new.to_csv(fnm, header=True, index=False)
    print('')

print('')
print(f'Error list (tickers): {error_l} {len(error_l)}')
print('')

print('Done.')
