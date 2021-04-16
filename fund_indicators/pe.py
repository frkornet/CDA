from   pathlib  import Path
from   datetime import date
import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import sys

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
    corp_stock =yf.Ticker(ticker).history(start=str(eps.index.min())[:10], end=str(eps.index.max())[:10])
    if len(corp_stock) == 0:
        print(f"Yfinance did not return any rows!")
        return None

    splits = corp_stock[['Stock Splits']].loc[corp_stock['Stock Splits'] > 0]
    splits['Split_Date'] = splits.index

    for index, row in splits.sort_values(by='Split_Date', ascending=False).iterrows():
        split_date  = str(index)[:10]
        stock_split = row['Stock Splits']
        print(split_date, stock_split)
        eps.loc[eps.ddate < split_date,'value'] = eps.loc[eps.ddate < split_date, 'value'].div(stock_split)

    eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
    eps = eps.rolling(window=4).sum()#.dropna()
    
    # Calculate p/e ratio
    pe = corp_stock.Close.to_frame('price').join(eps.to_frame('eps'))
    pe = pe.fillna(method='ffill')#.dropna()
    pe['P/E Ratio'] = pe.price.div(pe.eps)
    return pe

if __name__ == '__main__':
   if len(sys.argv) != 2:
      print('pe <ticker>\n')
      print('Where ticker is the stock symbol that you want to download the P/E ratio for')
      sys.exit() 

   data_path = Path('data') 
   ticker = sys.argv[1]

   try:
      name_map = pd.read_csv('name_map.csv')
   except:
      print('Unable to open name_map.csv!')
      sys.exit()

   idx = name_map.ticker == ticker
   if len(name_map.loc[idx]) == 0:
      print(f'Unable to find {ticker} in name_map.csv!')
      sys.exit()

   corp_name = (name_map['subs_name'].values)[0]
   data_avail = (name_map['data'].values)[0]
   if data_avail != 1:
      print('Data not available SUB, NUM, or Yfinance!')
      sys.exit()

   pe = get_pe(corp_name, ticker)
   if pe is None:
      print(f'- Unable to retrieve P/E ratio data for {ticker}!')
   else:
      print('')
      print('P/E ration for {ticker}:')
      print('')
      print(pe)

   
