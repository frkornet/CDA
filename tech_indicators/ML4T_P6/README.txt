Author: fkornet3
Date:   10/15/2020

Project 6 covers the implementation of five technical analysis indicators 
that will be used in the final ML4T project.

The README file explains what files are included in the submission:

- marketsimcode.py: 

  is the updated version of marketsim.py from project 5. Function 
  compute_portvals now accepts both a filename string and an orders 
  dataframe (parameter orders_file). The dataframe is expected to have the 
  same layout as the data stored in the order file CSV files. The expected 
  columns are: Date, Symbol, Order, and Shares. The program is not expected
  to be run by itself, and is called by TheoreticallyOptimalStrategy.py and 
  indicators.py.

- TheoreticallyOptimalStrategy.py: 

  determines the optimal trading strategy by peeking into the future and 
  making the optimal trade for each day. For each trading day, the allowed 
  trading positions are to 1) own 1000 shares, 2) short 1000 shares, or 
  3) not take a trading position (i.e. holding no shares in the company). 
  The program generates a PNG file ("optimal.png") that is included in 
  report.pdf. 
  
  Assuming you are in the directory where the files are located, you can 
  then run the program using the command:

  PYTHONPATH=../. python TheoreticallyOptimalStrategy.py

- indicators.py: 

  implements five technical analysis indicators. The following indicators 
  are implemented: 1) moving average convergence/divergence (MACD), 
  2) Bollinger Band position (BBP), 3) relative strength index (RSI), 4) money
  flow index (MFI), and 5) Williams %R index. The program generates a set of
  PNG files that are included in report.pdf to explain how the five indicators 
  work (BBP.png, MACD.png, MFI.png, RSI.png, SPY_JPM.png, and WilliamsR.png). 

  Assuming you are in the directory where the files are located, you can 
  then run the program using the command:

  PYTHONPATH=../. python indicators.py

- testproject.py:

  A wrapper test program that exercises indicators.py and 
  TheoreticallyOptimalStrategy.py. It generates all the figures used
  in report.pdf. It can be called from the command line as follows
  (assuming you are in the directory where all the files are located):

  PYTHONPATH=../. python testproject.py  

