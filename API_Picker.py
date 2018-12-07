#!/usr/bin/env python3

from coinbase.wallet.client import Client
import requests
import json
import sys

"""
This code is part of a UZH course project named: 
"Introduction to systematic risk premia strategies traded at hedge funds"
Our goal was to collect data from 2 exchanges and 1 web site place
them into a database and retrieve them to create an arbitrage index graph.
Our ultimate goal is to make this program to trade bud unfortunately we 
were not able to establish an account in Bithumb (Korea) which was a necessary
assumption to replicate the paper that we presented. 
In the future we will further develop this program to make it fully functional.
"""



flag = 1
threshold = 0.02
while (flag):

    # Documentation
    # https://github.com/coinbase/coinbase-python


    #fixer api - get korean won (KRW) and USD exchange rate
    f_api_url = "http://data.fixer.io/api/latest?access_key=dec14747f2f1dc199e927db91e52c060"
    r = requests.get(f_api_url)
    json_data = json.loads(str(r.content, encoding='utf-8'))
    USD = float(json_data['rates']['USD'])
    KRW = float(json_data['rates']['KRW'])
    USDKRW = KRW/USD


    #print(r.__dict__.keys())
    #print(json.load(r.content))
    json_data = json.loads(str(r.content, encoding='utf-8'))

    #bithumb data extraction
    api_url_private = "https://api.bithumb.com/info/account"
    api_url = "https://api.bithumb.com/public/ticker/BTC"
    api_key_B = 'place your api key for bithumb'
    api_secret_B = 'place your api secret for bithumb'


    r = requests.get(api_url)

    #print(r.__dict__.keys())
    #print(json.load(r.content))
    json_data = json.loads(str(r.content, encoding='utf-8'))
    buy_price_B = json_data['data']['buy_price']
    sell_price_B= json_data['data']['sell_price']
    buy_BTCUSD_KRW = float(buy_price_B) / float(USDKRW)
    sell_BTCUSD_KRW = float(sell_price_B) / float(USDKRW)



    #coinbase data extraction
    api_key_C = 'place you api key for coinbase'
    api_secret_C = 'place your	api secret for coinbase'

    client = Client(api_key_C, api_secret_C)
    buy_BTCUSD = float(client.get_buy_price(currency_pair = 'BTC-USD'))
    sell_BTCUSD = float(client.get_sell_price(currency_pair = 'BTC-USD'))

    buy_price = float(buy_price['amount'])
    sell_price = float(sell_price['amount'])
    date_time_C = client.get_time()
    date_time_C = date_time_C ["iso"]

    abs_diff = abs(buy_BTCUSD_KRW, buy_BTCUSD)
    abs_dif_per = abs_diff / max(buy_BTCUSD_KRW, buy_BTCUSD)

    data  = [date_time_C, USDKRW, buy_BTCUSD_KRW, sell_BTCUSD_KRW, buy_BTCUSD, sell_BTCUSD, abs_diff, abs_dif_per]

    if "12:00" in date_time_C:
        #insert hourly data into the database
        Database.Database.WriteHourlyData(data)

        #I can trade only to coinbase because I cannot open a ba
        if (abs_dif_per > threshold):
            if (buy_BTCUSD < buy_BTCUSD_KRW):
                #Buy bitcoins on Coinbase
                client.buy(account_id, amount='1', currency='BTC')

                #Commit buy
                #You only need to do this if the initial buy was explictly uncommitted
                buy = account.buy(amount='1', currency='BTC', commit=False)
                client.commit_buy(account_id, buy.id)

                #Sell bitcoins on Bithumb
                #documenation not clear ...
            else:
                #Sell bitcoins on Coinbase
                client.sell(account_id, amount='1', currency='BTC')
                # Commit buy
                # You only need to do this if the initial buy was explictly uncommitted
                sell = account.sell(amount='1', currency='BTC', commit=False)
                client.commit_sell(account_id, sell.id)

                #Buy bitcoins on Bithumb
                #documentation not clear ...
            Database.Database.WriteOrders()

        flag = 0


