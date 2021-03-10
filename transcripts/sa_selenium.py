#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Hacked the original version around to make it work with the latest
# version of the web site of seekingalpha.com
#
# Inspired by 02_earning_call_transcripts from Stefan Jansen
#
# Author: Frank Kornet
#

import re
from pathlib import Path
from random import random
from time import sleep
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from furl import furl
from selenium import webdriver

transcript_path = Path('transcripts')
print("transcript_path=", transcript_path)


def store_result(meta, participants, content):
    """Save parse content to csv"""
    path = transcript_path / 'parsed' / meta['symbol']
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(content, columns=['speaker', 'content']).to_csv(path / 'content.csv', header=True, index=False)
    pd.DataFrame(participants, columns=['type', 'name']).to_csv(path / 'participants.csv', header=-True, index=False)

    for k in meta.keys():
        meta[k] = [ meta[k] ]
    pd.DataFrame.from_dict(meta).to_csv(path / 'earnings.csv', index=False, header=True)


def parse_html(html):
    """Main html parser function"""
    quarter_pattern = re.compile(r'(\bQ\d\b) (\d{4})')
    soup = BeautifulSoup(html, 'lxml')

    meta, participants, content = {}, [], []
    h1 = soup.find('h1', attrs={"data-test-id":"post-title"})
    if h1 is None:
        return
    h1 = h1.text
    # print("h1=", h1)
    match = quarter_pattern.search(h1)
    if match:
        meta['quarter'] = str(match.group(0))
    
    meta['company'] = h1[:h1.find('(')].strip()
    meta['symbol'] = h1[h1.find('(') + 1:h1.find(')')]

    title = soup.find('span', attrs={"data-test-id":"post-date"}) #, class_='title')
    print("title 1:", title)
    if title is None:
        return
    dstr = title.text
    m, d, y = dstr[:3], int(dstr[5:7]), int(dstr[9:13])
    print(f"m={m} d={d} y={y}")
    meta['month'] = m
    meta['day'] = d
    meta['year'] = y

    speaker_types = ['Executives', 'Analysts']
    for i, header in enumerate([p.parent for p in soup.find_all('strong')]):
        text = header.text.strip()
        if text.lower().startswith('copyright'):
            continue
        elif text.lower().startswith('question-and'):
            continue
        elif i in [0, 1]:
           for participant in header.find_next_siblings('p'):
               if participant.find('strong'):
                   break
               else:
                   participants.append([header.text, participant.text])
        else:
            p = []
            for participant in header.find_next_siblings('p'):
                if participant.find('strong'):
                    break
                else:
                    p.append(participant.text)
            content.append([header.text, '\n'.join(p)])

    return meta, participants, content


SA_URL = 'https://seekingalpha.com/'
TRANSCRIPT = re.compile('Earnings Call Transcript')

next_page = True
page = 1
driver = webdriver.Firefox()
while next_page:
    print(f'Page: {page}')
    url = f'{SA_URL}/earnings/earnings-call-transcripts/{page}'
    driver.get(urljoin(SA_URL, url))
    sleep(8 + (random() - .5) * 2)
    response = driver.page_source
    page += 1
    soup = BeautifulSoup(response, 'lxml')
    links = soup.find_all(name='a', string=TRANSCRIPT)
    print(f"len(links)={len(links)}")
    if len(links) == 0:
        next_page = False
    else:
        for link in links:
            transcript_url = link.attrs.get('href')
            article_url = furl(urljoin(SA_URL, transcript_url)).add({'part': 'single'})
            driver.get(article_url.url)
            html = driver.page_source
            
            result = parse_html(html)
            if result is not None:
                meta, participants, content = result
                meta['link'] = link

                print('')                
                print(meta)
                print('')
                print(participants)
                print('')
                print('content:', type(content), len(content))
                print('')

                store_result(meta, participants, content)
                sleep(8 + (random() - .5) * 2)

driver.close()

print('')
print('Done.')
print('')
