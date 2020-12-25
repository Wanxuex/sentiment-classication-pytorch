#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import pandas as pd
from bs4 import BeautifulSoup
url = 'https://www.bilibili.com/v/popular/rank/all'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')


all_products = []
products = soup.select('li.rank-item')
for product in products:
    rank = product.select('div.num')[0].text
    name = product.select('div.info > a')[0].text.strip()
    play = product.select('span.data-box')[0].text
    comment = product.select('span.data-box')[1].text
    up = product.select('span.data-box')[2].text
    url = product.select('div.info > a')[0].attrs['href']
    all_products.append({
        "视频排名":rank,
        "视频名": name,
        "播放量": play,
        "弹幕量": comment,
        "up主": up,
        "视频链接": url
    })


keys = all_products[0].keys()

pd.DataFrame(all_products,columns=keys).to_csv('B站视频热榜TOP100.csv', encoding='utf-8')
