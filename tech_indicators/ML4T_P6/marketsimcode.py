""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  

from datetime import timedelta, datetime, date

def add_days_to_date(ds, days, fmt='%Y-%m-%d'):
    ds_date = datetime.strptime(str(ds)[:10], '%Y-%m-%d')
    ds_date = ds_date + timedelta(days=int(days))
    return ds_date.__format__(fmt)

def author():
    return "fkornet3"

def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		     		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		     		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
    """		  	   		     		  		  		    	 		 		   		 		  

    # Read in orders file if orders_file is a string. Otherwise assume it is a dataframe
    # with the correct columns. Once read, determine start and end date
    odf = pd.read_csv(orders_file) if type(orders_file) == type("") else orders_file
    syms = list(set(odf.Symbol))
    odf = odf.sort_values(by='Date', ascending=True)
    start_date = odf.iloc[0].Date
    end_date   = odf.iloc[-1].Date

    # Create prices data frame
    pdf = get_data(syms, pd.date_range(start_date, end_date))
    pdf = pdf[syms]
    pdf['Cash'] = 1
    pdf.fillna(method='ffill')
    pdf.fillna(method='bfill')

    # Create initial trades data frame
    tdf = pdf.copy()
    tdf[syms]   = 0
    tdf['Cash'] = 0

    # Add orders to trades data frame
    for i, o in odf.iterrows():
        dat, sym = o.Date, o.Symbol
        assert o.Order.upper() in ['BUY', 'SELL'], "unknown order type"
        shares = o.Shares if o.Order.upper() == 'BUY' else -o.Shares
        if dat not in pdf.index or sym not in pdf.columns:
            continue
        cash = pdf.loc[dat, sym] * shares * -1.0
        transaction_cost = commission + impact * abs(cash)
        cash -= transaction_cost
        tdf.loc[dat, sym]    += shares
        tdf.loc[dat, 'Cash'] += cash

    # Create holdings data frame
    hdf = tdf.copy()
    s_minus1 = add_days_to_date(start_date, -1)
    first_row = {}
    for s in syms:
        first_row[s] = [ 0 ]
    first_row['Cash'] = [ start_val ]
    fr_df = pd.DataFrame.from_dict(first_row)
    fr_df.index = [s_minus1]

    hdf = pd.concat([fr_df, hdf])
    hdf = hdf.cumsum()

    # Create values data frame
    vdf = pdf * hdf
    port_vals = vdf.sum(axis=1)
    index = port_vals.index == s_minus1
    port_vals = port_vals.loc[~index]  
    return port_vals

def test_code(of, sv):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		     		  		  		    	 		 		   		 		  
    """  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Process orders  		  	   		     		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		     		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		     		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		     		  		  		    	 		 		   		 		  
    else:  		  	   		     		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		     		  		  		    	 		 		   		 		  

    odf = pd.read_csv(of)
    syms = list(set(odf.Symbol))
    start_date = odf.iloc[0].Date
    end_date   = odf.iloc[-1].Date
  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Date Range            : {start_date} to {end_date}")
    print(f"Symbols traded in     : {syms}")
    print(f"Number of trading days: {len(portvals)}")
    print(f"Start Portfolio Value : {portvals[0]}")
    print(f"Final Portfolio Value : {portvals[-1]}")

	  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  
    order_files = [ f'./orders/orders-{i:02d}.csv' for i in range(1,13) ]
    lev_files   = [ f'./orders/orders-leverage-{i:01d}.csv' for i in range(1,4) ]
    test_files  = [  './orders/orders-short.csv',
                     './orders/orders-short-fk.csv',
                     './orders/orders.csv',
                     './orders/orders2.csv'
                    ]
    files_to_process = order_files + lev_files + test_files
    for f in files_to_process:
        print(f"Processing file: {f}:")
        test_code(f, 1000000)
        print('')
    # test_code('./orders/orders-short-fk.csv', 1000000)  