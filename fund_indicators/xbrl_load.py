import warnings
warnings.filterwarnings('ignore')

from   pathlib import Path
from   datetime import date
import json
from   io import BytesIO
from   zipfile import ZipFile, BadZipFile
import requests

import pandas_datareader.data as web
import pandas as pd

from   pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import yfinance as yf
import sys

SEC_URL = 'https://www.sec.gov/'
FSN_PATH = 'files/dera/data/financial-statement-and-notes-data-sets/'

today = pd.Timestamp(date.today())
this_year = today.year
this_quarter = today.quarter

if len(sys.argv) < 2 or len(sys.argv) > 3:
   print('xbrl_load <from_year> [<to_year>]')
   print('')
   print('Default for <to_year> is current year')
elif len(sys.argv) == 2:
   from_year, to_year = int(sys.argv[1]), this_year
else:
   from_year, to_year = int(sys.argv[1]), int(sys.argv[2])

past_years = range(from_year, to_year+1) 

filing_periods = [(y, q) for y in past_years for q in range(1, 5)
                  if y != this_year or (y==this_year and q <= this_quarter)
]


####################################################
#####       Download the XRBL tsv file         #####
####################################################

data_path = Path('data')
if not data_path.exists():
    data_path.mkdir()

print(f'Downloading tsv files for {len(filing_periods)}')
for i, (yr, qtr) in enumerate(filing_periods, 1):
    print(f'- {i} {yr}-{qtr}: ', end=' ', flush=True)
    
    filing = f'{yr}q{qtr}_notes.zip'
    path = data_path / f'{yr}_{qtr}' / 'source'
    print(path)
    
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    
    url = SEC_URL + FSN_PATH + filing
    response = requests.get(url).content
    
    try:
        with ZipFile(BytesIO(response)) as zip_file:
            for file in zip_file.namelist():
                local_file = path / file
                if local_file.exists():
                    continue
                with local_file.open('wb') as output:
                    for line in zip_file.open(file).readlines():
                        output.write(line)
    except BadZipFile:
        'https://www.sec.gov/files/node/add/data_distribution/2020q1_notes.zip'
        print('got bad zip file')
        continue

print('')
print('The following files have been down loaded from Edgar:')
print('')

for f in sorted(data_path.glob('**/*.tsv')):
    print(f'- {f}')


###########################################################
#####       Converting tsv files to parquet files     #####
###########################################################

print('')
print('Converting the down loaded tsv files to parquet files to save space:')
print('')

# A bit of twisted logic going on here. Needs to be fixed over time.
for f in sorted(data_path.glob('**/*.tsv')):
    file_name = f.stem  + '.parquet'
    path = Path(f.parents[1]) / 'parquet'
    if (path / file_name).exists():
        continue
    if not path.exists():
        path.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False)
    except:
        print(f)
    df.to_parquet(path / file_name, engine='pyarrow' )

print('')
print('The following files have been converted to parquet:')
print('')

for f in sorted(data_path.glob('**/*.parquet')):
    print(f'- {f}')

print('')
print('Done.')
