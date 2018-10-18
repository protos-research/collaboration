#!/usr/local/bin/python3

"""
This script fetches data from the CoinMarketCap API and populates a MySQL database.

It is intended to replace the Google Sheet (https://docs.google.com/spreadsheets/d/1IEOI96L6P7CLyf9TO0_MUcCPFyWOmmw8wlXnj8xvo48/edit) that was previously used.
It is run on a cronjob on a server.
"""


"""
Previous architecture:

- gsupdates/updateCoinmarketcap.py: Fetch data from CoinMarketCap API and populate 'Protos Strategies' GSheet
- rdbpipeline/[various files].py: Fetch data from GSheet and insert rows into MySQL
    -- indices.py: indices sheet, indices table
    -- nvx.py: Creates indexes on tables, does various (re)formatting of data in tables
    -- protosterminal.py: Fetches data from GSheet 'Dashboard Data' and puts into MySQL
    -- txvol_actadr.py: Fetches data from CoinMarketCap API (about transaction volumes) and puts into MySQL
    -- update_coinmarket_cap.py: ditto
    -- coinmarketcap.py: ditto



New architecture design decisions:
- I have deliberately tried to implement this new script/architecture with minimal disruption to the old one.  This is because the scope of the work is not a full rewrite and it would be dangerous to change more than necessary.
- The architecture in this script provides a blueprint for when a full rewrite is to be done.  At that point all scripts should be rearchitected along the lines of this file
"""

'''
Sheets whose equivalent data this script needs to put into MySQL
BLACK
- price (table name = high)
- volume (√)
- cap (√)
- tx_volume (√)
- active_adr (there is a table named active_address but it does not match the data in the sheet)
BLUE
- return (calculated from above data)
- vol (calculated from above data)
- ann vol (calculated from above data)
- access top50 (calculated from above data)


Suggested improvements:
    - Put code into source control (git/GitHub)
    - Replace crontab with something less ad-hoc, for example, timed triggers on Heroku
    - Use industry standard virtualenv and requirements.txt to manage dependencies
    - Scripts that exist purely to format data already in the db (eg, nvx.py) should have their functionality placed in the script that populates the DB
    - Remove database addresses and passwords from source code and put into secure environment variables

Minor improvements:
    - In cronjobs: execute python scripts directly, remove superfluous bash script
    - Remove hard coded dates
    - There is much repeated functionality which should be encapsulated into funtional components for reuse

Nice to haves:
    - Setup automatic deployment connected to source control (and continous integration)
    - Create tests
'''


import logging
import pickle
import math
import os
import itertools
import datetime
from functools import lru_cache

import sqlalchemy as sql
import pandas as pd
import numpy as np

from get_data_from_api import get_data_from_api_legacy
from get_txvol_data_from_api import get_data_txvol_legacy


# TODO: This should be in an environment variable - when updating server setup make this change
DEV = True

LOGGER_LEVEL = logging.INFO if not DEV else logging.DEBUG

logging.basicConfig(level=LOGGER_LEVEL)
logger = logging.getLogger(__name__)

if DEV:
    logger.debug('In DEV mode - NOT WRITING TO PRODUCTION DATABASE')
else:
    logger.info('In PRODUCTION mode - WRITING TO PRODUCTION DATABASE')


PRODUCTION_DB_URL = 'mysql+pymysql://protos:Theshowmustgoon!@google-sheet-data-eu-west-1b.cfyqhzfdz93r.eu-west-1.rds.amazonaws.com:3306/newarchitecture'
DEV_DB_URL = 'mysql+pymysql://root:password@localhost:3306/internal'
DB_URL = DEV_DB_URL if DEV else PRODUCTION_DB_URL

engine = sql.create_engine(DB_URL)

'''
These are the variances used in the GSheet - useful for testing consistency with GSheets
Note: For consistent testing these need to be updated daily
'''
VARIANCES_FOR_TESTING = {'0x': 0.0095841076, 'aelf': 0.0099395279, 'aeternity': 0.0195811843, 'aion': 0.0190375303, 'ardor': 0.0084259257, 'ark': 0.0124218183, 'augur': 0.0069846246, 'aurora': 0.0180360041, 'bancor': 0.0057064635, 'basic-attention-token': 0.0088429915, 'binance-coin': 0.0126076397, 'bitcoin': 0.0022385676, 'bitcoin-cash': 0.0095989341, 'bitcoin-diamond': 0.0643285748, 'bitcoin-gold': 0.0144287356, 'bitsend': 0.0259449468, 'bitshares': 0.0100998457, 'blocknet': 0.0143253061, 'bulwark': 0.0128915752, 'bytecoin-bcn': 0.0623462683, 'bytom': 0.0119003586, 'cardano': 0.016177607, 'chainlink': 0.0098694995, 'crown': 0.0184695561, 'crypto-com': 0.0226824639, 'cryptonex': 0.009222643, 'cybermiles': 0.0100837545, 'dai': 0.0001494021, 'dash': 0.0054334299, 'decentraland': 0.0184147565, 'decred': 0.0095290124, 'dentacoin': 0.0388717917, 'diamond': 0.0095926789, 'digibyte': 0.0200614211, 'digitex-futures': 0.012645447, 'digixdao': 0.0075097645, 'dogecoin': 0.0077870745, 'dropil': 0.0065254045, 'elastos': 0.0076674682, 'electroneum': 0.0159020441, 'eos': 0.0154907053, 'eternal-token': 0.0883035046, 'ethereum': 0.0044284032, 'ethereum-classic': 0.0062065232, 'exclusivecoin': 0.0280431031, 'funfair': 0.0422207146, 'gas': 0.0177888081, 'golem-network-tokens': 0.0104989741, 'gxchain': 0.0110490502, 'hexx': 0.0852765415, 'holo': 0.0116858135, 'huobi-token': 0.0048555574, 'hypercash': 0.014459153, 'icon': 0.0131700557, 'ion': 0.0383974335, 'iostoken': 0.029647691, 'iota': 0.009089495, 'komodo': 0.063213188, 'korecoin': 0.0294102363, 'kucoin-shares': 0.0132433345, 'lisk': 0.007943842, 'litecoin': 0.0057649211, 'loom-network': 0.0110243797, 'loopring': 0.0133619309, 'maidsafecoin': 0.0051114011, 'maker': 0.3560708855, 'metaverse': 0.0686298179, 'mithril': 0.0356532292, 'mixin': 0.0093891797, 'moac': 0.0065086026, 'monacoin': 0.0139780421, 'monero': 0.0054358402, 'monetaryunit': 0.0143211433, 'nano': 0.0197876255, 'nebulas-token': 0.0134889245, 'nem': 0.0123106007, 'neo': 0.0136786027, 'neoscoin': 0.0160255979, 'noah-coin': 0.0178401542, 'nxt': 0.0093574241, 'omisego': 0.0106758194, 'ontology': 0.0101986176, 'pivx': 0.0121470317, 'populous': 0.0124338689, 'power-ledger': 0.0143710834, 'pundi-x': 0.0182111414, 'qash': 0.0094029946, 'qtum': 0.0116088462, 'rchain': 0.0162935781, 'reddcoin': 0.0411488757, 'ripple': 0.0141393729, 'siacoin': 0.0120971053, 'sibcoin': 0.0072657211, 'status': 0.0144325463, 'steem': 0.0111645792, 'stellar': 0.0136443039, 'stratis': 0.0096565459, 'tenx': 0.0149162888, 'tether': 0.0000632384, 'tezos': 0.008891534, 'theta-token': 0.0105088845, 'transfercoin': 0.0282597101, 'tron': 0.0237252628, 'trueusd': 0.0001129721, 'vechain': 0.0099191272, 'verge': 0.039620529, 'waltonchain': 0.0204756172, 'wanchain': 0.0071673718, 'waves': 0.0060821917, 'wax': 0.0190379917, 'zcash': 0.0054625996, 'zcoin': 0.0107989088, 'zencash': 0.0166101779, 'zilliqa': 0.0072406397 }


# trend sheet
TREND_ASSETS = set([
    'bitcoin',
    'bitcoin-cash',
    'ethereum',
    'litecoin',
    'ripple',
    'neo',
    'stellar',
    'eos',
])


### Helper functions

def get_today_date(df, today_index):
    return df['Date'][today_index]

def convert_number(n):
    if not n or n == '-':
        return 0.0

    if isinstance(n, float):
        return n
    else:
        return float(n.replace(',', ''))


def get_number_from_df(df, column_name, i):
    return convert_number(df[column_name][i])

def format_df(df):
    df.sort_values(by='Date', ascending=True, inplace=True)

def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """
    cachefile = f'{cachefile}.pickle'

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):   # define a wrapper that will finally call "fn" with all arguments
            if DEV:
                # if cache exists -> load it and return its content
                if os.path.exists(cachefile):
                        with open(cachefile, 'rb') as cachehandle:
                            logger.debug("using cached result from '%s'" % cachefile)
                            return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            if DEV:
                # write to cache file
                with open(cachefile, 'wb') as cachehandle:
                    logger.debug("saving result to cache '%s'" % cachefile)
                    pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator   # return this "customized" decorator that uses "cachefile"


@cached('data_from_api_part_one')
def cached_get_data_from_api_legacy():
    return get_data_from_api_legacy()

@cached('data_from_api_all')
def get_data_from_api():

    data = cached_get_data_from_api_legacy()

    data_2 = get_data_txvol_legacy()

    logger.debug('Got data from API')

    # data_to_return = data
    data_to_return = data + data_2

    return data_to_return



def create_return_variance_query_text(list_of_currencies):
    TABLE_NAME = 'return'

    select_clause = ", ".join(
        f"VARIANCE(`{currency}`) as `{currency}`" for currency in list_of_currencies)

    query_text = f"SELECT {select_clause} FROM `{TABLE_NAME}`;"

    return query_text


def calculate_data_wrapper(df_to_iterate_over, create_row, reversed=False, offset=0):
    range_to_iterate_over = range(len(df_to_iterate_over) - offset)
    range_to_iterate_over = reversed(range_to_iterate_over) if reversed else range_to_iterate_over

    rows = [create_row(i) for i in range_to_iterate_over]
    df = pd.DataFrame(rows)
    format_df(df)

    return df


def write_df_to_db(df, table_name, engine):
    if DEV:
        logger.debug(f'DEV mode - not writing {table_name} to db')
    else:
        logger.info(f'writing {table_name} to db')
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

#############

#### Formula functions

def calculate_return_price(today_max, yesterday_max):
    '''=iferror(IF((price!F4/price!F3-1)=-1,"",price!F4/price!F3-1),"")'''
    return today_max / yesterday_max - 1


def calculate_vol(yesterday_vol, todays_return, historical_variance_for_currency):
    '''
    F$2 = variance_of_all_return_prices

    calc_without_variance = 0.94*yesterday_vol+0.06*today_return^2
    calc_with_variance = 0.94*variance_of_all_return_prices+0.06*today_return^2

    x = yesterday_vol if not today_return else calc_without_variance # if(today_return="",yesterday_vol,calc_without_variance)
    y = calc_with_variance

    return None if x==yesterday_vol else (y if not yesterday_vol else x)
    '''
    volatility = lambda z: 0.94*(z if isinstance(z, float) else 0.0) + 0.06*math.pow(todays_return, 2.0)

    calc_without_variance = volatility(yesterday_vol)
    calc_with_historical_volatility_variance = volatility(historical_variance_for_currency)

    if not yesterday_vol or math.isnan(yesterday_vol):
        return calc_with_historical_volatility_variance
    else:
        return calc_without_variance


def calculate_access_top50(today_mcap, currency, top_50_0x_mcap):
    '''=iferror(if(Rank(cap!F2,cap!$B2:2,0)<=50,1,0)*index(dictionary!$A:$K,match(F$1,dictionary!$A:$A,0),match(dictionary!$J$1,dictionary!$A$1:$K$1,0)),"")'''
    in_todays_top50_mcap = today_mcap >= top_50_0x_mcap

    if in_todays_top50_mcap:
        return get_total_access(currency)
    else:
        return False


def calculate_ann_vol(todays_vol, todays_access_top50):
    '''=iferror(if(sqrt(vol!F4)*sqrt(365)*'access top50'!F4=0,"",sqrt(vol!F4)*sqrt(365)*'access top50'!F4),"")'''

    ann_vol = math.sqrt(todays_vol) * math.sqrt(365)

    if not ann_vol or ann_vol == 0.0 or math.isnan(ann_vol) or not todays_access_top50:
        return None

    return ann_vol

##########################


def get_total_access(currency_name):
    '''This is a duplicate of the dictionary sheet'''

    total_access_dict = {'0x': True, 'aelf': True, 'aeternity': True, 'aion': True, 'ardor': True, 'ark': True, 'augur': True, 'aurora': False, 'bancor': True, 'basic-attention-token': False, 'binance-coin': True, 'bitcoin': True, 'bitcoin-cash': True, 'bitcoin-diamond': True, 'bitcoin-gold': True, 'bitsend': True, 'bitshares': True, 'blocknet': True, 'bulwark': True, 'bytecoin-bcn': True, 'bytom': False, 'cardano': True, 'chainlink': True, 'crown': True, 'crypto-com': False, 'cryptonex': False, 'cybermiles': True, 'dai': False, 'dash': True, 'decentraland': True, 'decred': True, 'dentacoin': False, 'diamond': True, 'digibyte': True, 'digitex-futures': False, 'digixdao': True, 'dogecoin': True, 'dropil': False, 'elastos': False, 'electroneum': False, 'eos': True, 'eternal-token': False, 'ethereum': True, 'ethereum-classic': True, 'exclusivecoin': True, 'funfair': True, 'gas': True, 'golem-network-tokens': True, 'gxchain': True, 'hexx': True, 'holo': False, 'huobi-token': False, 'hypercash': False, 'icon': True, 'ion': True, 'iostoken': True, 'iota': True, 'komodo': True, 'korecoin': True, 'kucoin-shares': False, 'lisk': True, 'litecoin': True, 'loom-network': True, 'loopring': True, 'maidsafecoin': False, 'maker': False, 'metaverse': True, 'mithril': False, 'mixin': False, 'moac': False, 'monacoin': True, 'monero': True, 'monetaryunit': True, 'nano': True, 'nebulas-token': False, 'nem': True, 'neo': True, 'neoscoin': True, 'noah-coin': False, 'nxt': True, 'omisego': True, 'ontology': True, 'pivx': True, 'populous': True, 'power-ledger': True, 'pundi-x': False, 'qash': True, 'qtum': True, 'rchain': False, 'reddcoin': True, 'ripple': True, 'siacoin': True, 'sibcoin': True, 'status': True, 'steem': True, 'stellar': True, 'stratis': True, 'tenx': True, 'tether': False, 'tezos': False, 'theta-token': False, 'transfercoin': True, 'tron': True, 'trueusd': False, 'vechain': True, 'verge': True, 'waltonchain': True, 'wanchain': True, 'waves': True, 'wax': True, 'zcash': True, 'zcoin': True, 'zencash': True, 'zilliqa': True, }

    return total_access_dict.get(currency_name, False)


def get_historical_return_variances(engine, list_of_currencies):
    if DEV:
        return VARIANCES_FOR_TESTING

    variance_query_text = create_return_variance_query_text(list_of_currencies)

    result = engine.execute(variance_query_text)

    row = result.fetchone()

    historical_return_variances = dict(zip(row.keys(), row))

    return historical_return_variances


def get_return_price(currency, price_df, today_index, yesterday_index):
    currency_maxes = price_df[currency]

    today_max = float(currency_maxes[today_index])
    yesterday_max = float(currency_maxes[yesterday_index])

    return calculate_return_price(today_max, yesterday_max)


@cached('return_df')
def calculate_return_data(list_of_currencies, price_df):
    return_rows = []

    for i in range(len(price_df)):
        today_index = i
        yesterday_index = i + 1

        if yesterday_index >= len(price_df):
            continue

        today_date = price_df['Date'][today_index]

        row = {currency: get_return_price(currency, price_df, today_index, yesterday_index) for currency in list_of_currencies}
        row['Date'] = today_date

        return_rows += [row]

    return_df = pd.DataFrame(return_rows)

    format_df(return_df)
    return return_df


def get_vol(currency, volume_df, return_df, today_index, yesterday_index, historical_return_variances, yesterday_vol, today_date):
    try:
        # Some of the data is malformed
        todays_return = float(return_df[currency][today_index])
    except KeyError:
        return None

    historical_variance_for_currency = historical_return_variances.get(currency, 0.0)

    vol = calculate_vol(yesterday_vol, todays_return, historical_variance_for_currency)

    return vol


@cached('vol_df')
def calculate_vol_data(list_of_currencies, volume_df, return_df, historical_return_variances):
    return_rows = []

    yesterdays_vols = {}
    for i in reversed(range(len(volume_df))):
        row = {}

        today_index = i
        yesterday_index = i - 1
        today_date = volume_df['Date'][today_index]

        for currency in list_of_currencies:
            yesterday_vol = yesterdays_vols.get(currency, None)
            today_vol = get_vol(currency, volume_df, return_df, today_index, yesterday_index, historical_return_variances, yesterday_vol, today_date)

            row[currency] = today_vol

        row['Date'] = today_date

        yesterdays_vols = row

        return_rows += [row]

    vol_df = pd.DataFrame(list(reversed(return_rows)))

    format_df(vol_df)
    return vol_df


@cached('ann_vol_df')
def calculate_ann_vol_data(vol_df, access_top50_df, list_of_currencies):
    return_rows = []

    for i in range(len(vol_df)):
        today_index = i
        today_date = vol_df['Date'][today_index]

        row = {}
        for currency in list_of_currencies:
            todays_vol = convert_number(vol_df[currency][today_index])
            todays_access_top_50 = access_top50_df[currency][today_index]

            row[currency] = calculate_ann_vol(todays_vol, todays_access_top_50)

        row['Date'] = today_date

        return_rows += [row]

    return_df = pd.DataFrame(return_rows)

    format_df(return_df)
    return return_df


@cached('access_top50_df')
def calculate_access_top50_data(mcap_df, list_of_currencies):
    return_rows = []

    for i in range(len(mcap_df)):
        today_index = i
        today_date = mcap_df['Date'][today_index]

        all_mcaps_today = mcap_df.loc[mcap_df['Date'] == today_date]
        all_mcaps_today_list_with_header = all_mcaps_today.values.tolist()[0]
        all_mcaps_today_list = all_mcaps_today_list_with_header[1:]
        all_mcaps_today_as_number_list = [convert_number(n) for n in all_mcaps_today_list]

        all_mcaps_today_as_number_list_sorted = sorted(all_mcaps_today_as_number_list, reverse=True)
        top_50_mcaps_today = all_mcaps_today_as_number_list_sorted[:50]
        top_50th_mcap = top_50_mcaps_today[-1]

        row = {}
        for currency in list_of_currencies:
            mcap_raw = mcap_df[currency][today_index]
            today_mcap = convert_number(mcap_raw)
            access_top50 = calculate_access_top50(today_mcap, currency, top_50th_mcap)

            row[currency] = access_top50
        row['Date'] = today_date

        return_rows += [row]

    access_top50_df = pd.DataFrame(return_rows)

    format_df(access_top50_df)
    return access_top50_df


def calculate_trend(today_price, thiry_days_ago_price, sixty_days_ago_price, ninety_days_ago_price, today_access_top50):
    '''=(if(price!B482-price!B392>0,1/3,-1/3)+if(price!B482-price!B422>0,1/3,-1/3)+if(price!B482-price!B452>0,1/3,-1/3))*'access top50'!B482'''

    trend = lambda price: 1/3 if today_price - price > 0 else -1/3

    if not today_access_top50:
        return None

    complete_trend = (trend(ninety_days_ago_price) + trend(sixty_days_ago_price) + trend(thiry_days_ago_price))

    return complete_trend


@cached('trend')
def calculate_trend_data(price_df, access_top50_df, list_of_currencies):
    def create_row(today_index):
        def calculate_trend_for_currency(currency):
            def today_price_offset(d=0): return convert_number(price_df[currency][today_index + d])

            today_access_top50 = access_top50_df[currency][today_index]

            return calculate_trend(today_price_offset(), today_price_offset(30), today_price_offset(60), today_price_offset(90), today_access_top50)


        row = {c: calculate_trend_for_currency(c) for c in list_of_currencies}
        row['Date'] = get_today_date(price_df, today_index)

        return row

    # Since we use an offset of 90 days to calculate the trend we need to take this into account
    return_rows = [create_row(i) for i in range(len(price_df) - 90)]


    trend_df = pd.DataFrame(return_rows)
    format_df(trend_df)

    return trend_df


def calculate_trend_ann_vol(currency, any_access_top50_today, ann_vol_today_list, today_ann_vol):
    '''
    =iferror((sumif('access top50'!$B349:349,"1",'ann vol'!$B349:349)/'ann vol'!M349)*if(match(O$1,dictionary!$N$1:$N$16,0)>0,1,0),"")
    '''

    if currency not in TREND_ASSETS:
        return None

    return np.nansum(ann_vol_today_list) / today_ann_vol


@cached('trend_ann_vol')
def calculate_trend_ann_vol_data(ann_vol_df, access_top50_df, list_of_currencies):
    def create_row(today_index):
        today_date = get_today_date(ann_vol_df, today_index)

        any_access_top50_today = any(access_top50_df.loc[today_index])

        ann_vol_today_list = [convert_number(val) for name, val in ann_vol_df.loc[today_index].iteritems() if not name == 'Date']

        def calculate_trend_ann_vol_for_currency(currency):
            today_ann_vol = ann_vol_df[currency][today_index]

            return calculate_trend_ann_vol(currency, any_access_top50_today, ann_vol_today_list, today_ann_vol)


        row = {c: calculate_trend_ann_vol_for_currency(c) for c in list_of_currencies}
        row['SUM'] = sum(v for v in row.values() if v)
        row['Date'] = today_date

        return row

    rows = [create_row(i) for i in range(len(ann_vol_df))]
    df = pd.DataFrame(rows)
    format_df(df)

    return df


def calculate_w_trend(today_access_top50, today_trend, trend_ann_vol_today, trend_ann_vol_sum_today):
    '''
    =iferror('trend/annvol'!O2/'trend/annvol'!$B2*'access top50'!M349*trend!M2,"")
    '''

    if not today_access_top50 or not today_trend or not trend_ann_vol_today:
        return None

    return trend_ann_vol_today / trend_ann_vol_sum_today * today_access_top50 * today_trend


def create_row_helper(calculation_func, list_of_currencies, today_date, add_sum_column=False):
    row = {c: calculation_func(c) for c in list_of_currencies}
    row['Date'] = today_date
    if add_sum_column:
        row['SUM'] = sum(v for v in row.values() if v and isinstance(v, float))

    return row


@cached('w_trend')
def calculate_w_trend_data(access_top50_df, trend_ann_vol_df, trend_df, list_of_currencies):
    def create_row(today_index):
        today_date = get_today_date(access_top50_df, today_index)

        def calculate_w_trend_for_currency(currency):
            today_access_top50 = access_top50_df[currency][today_index]
            today_trend = trend_df[currency][today_index]
            trend_ann_vol_today = trend_ann_vol_df[currency][today_index]
            trend_ann_vol_sum_today = trend_ann_vol_df['SUM'][today_index]

            return calculate_w_trend(today_access_top50, today_trend, trend_ann_vol_today, trend_ann_vol_sum_today)

        return create_row_helper(calculate_w_trend_for_currency, list_of_currencies, today_date)

    # Note: Trend has the smallest number of values because it requires a 90 day lead to calculate
    return calculate_data_wrapper(trend_df, create_row)


def calculate_dominance(mcap_today, mcap_all_currencies_today):
    '''
    The dominance table will include the dates and the name of the coins. For every day, and for every coin, will be calculating: Market Cap of Coin / Sum of Market caps for that day
    eg, Dominance of BTC for today = Marketcap BTC for today / sum(market caps for today )
    '''

    return mcap_today / mcap_all_currencies_today


@cached('dominance')
def calculate_dominance_data(mcap_df, list_of_currencies):
    @lru_cache()
    def get_all_mcaps_for_date(today_date):
        all_mcaps_today = mcap_df.loc[mcap_df['Date'] == today_date]
        all_mcaps_today_list_with_header = all_mcaps_today.values.tolist()[0]
        all_mcaps_today_list = all_mcaps_today_list_with_header[1:]
        all_mcaps_today_as_number_list = [convert_number(n) for n in all_mcaps_today_list]

        return all_mcaps_today_as_number_list


    def create_row(today_index):
        today_date = get_today_date(mcap_df, today_index)

        mcap_all_currencies_today = np.nansum(get_all_mcaps_for_date(today_date))

        def calculate_dominance_for_currency(currency):
            mcap_today = get_number_from_df(mcap_df, currency, today_index)

            return calculate_dominance(mcap_today, mcap_all_currencies_today)

        return create_row_helper(calculate_dominance_for_currency, list_of_currencies, today_date)

    return calculate_data_wrapper(mcap_df, create_row)


@cached('trend_dominance')
def calculate_trend_dominance_data(dominance_df, access_top50_df, list_of_currencies):
    def create_row(today_index):
        today_date = get_today_date(dominance_df, today_index)

        def calculate_trend_dominance_for_currency(currency):
            def today_dominance_offset(d=0): return get_number_from_df(dominance_df, currency, today_index + d)

            access_top50_today = access_top50_df[currency][today_index]

            # trend_dominance is the trend function with dominance instead of price as its variable
            return calculate_trend(today_dominance_offset(), today_dominance_offset(30), today_dominance_offset(60), today_dominance_offset(90), access_top50_today)

        return create_row_helper(calculate_trend_dominance_for_currency, list_of_currencies, today_date)

    return calculate_data_wrapper(dominance_df, create_row, offset=90)


@cached('w_trend_dominance')
def calculate_w_trend_dominance_data(access_top50_df, trend_ann_vol_df, trend_dominance_df, list_of_currencies):
    def create_row(today_index):
        today_date = get_today_date(access_top50_df, today_index)

        def calculate_w_trend_dominance_for_currency(currency):
            today_access_top50 = access_top50_df[currency][today_index]
            today_trend_dominance = trend_dominance_df[currency][today_index]
            trend_ann_vol_today = trend_ann_vol_df[currency][today_index]
            trend_ann_vol_sum_today = trend_ann_vol_df['SUM'][today_index]

            return calculate_w_trend(today_access_top50, today_trend_dominance, trend_ann_vol_today, trend_ann_vol_sum_today)

        return create_row_helper(calculate_w_trend_dominance_for_currency, list_of_currencies, today_date)

    # Note: Trend dominance has the smallest number of values because it requires a 90 day lead to calculate
    return calculate_data_wrapper(trend_dominance_df, create_row)


## DB Logic

def put_into_db(dfs):
    '''
    NOTE: These tables were named incorrectly in the previous script.
    '''
    (
        max_df,
        price_df,
        volume_df,
        mcap_df,
        *_,
    ) = dfs

    if not DEV:
        max_df.to_sql(name="max", con=engine, if_exists='replace', index=False)
        price_df.to_sql(name="price", con=engine, if_exists='replace', index=False)
        volume_df.to_sql(name="volume", con=engine, if_exists='replace', index=False)
        mcap_df.to_sql(name="cap", con=engine, if_exists='replace', index=False)
    else:
        logger.debug('In DEV mode - not writing original data into DB')


def populate_db(data_from_api):
    put_into_db(data_from_api)

    (
        _,
        price_df,
        volume_df,
        mcap_df,
        # *_
        txvolume_df,
        df_actadr,
        df_txcount,
        df_fees,
        df_difficulty,
    ) = data_from_api

    return get_calculated_data_and_populate_db(price_df, volume_df, mcap_df, txvolume_df)


#####

def get_calculated_data_and_populate_db(price_df, volume_df, mcap_df, txvolume_df):
    list_of_currencies = list(price_df)
    list_of_currencies.remove('Date')

    return_df = calculate_return_data(list_of_currencies, price_df)
    write_df_to_db(return_df, 'return', engine)

    historical_return_variances = get_historical_return_variances(engine, list_of_currencies)

    vol_df = calculate_vol_data(list_of_currencies, volume_df, return_df, historical_return_variances)
    write_df_to_db(vol_df, 'vol', engine)

    access_top50_df = calculate_access_top50_data(mcap_df, list_of_currencies)
    write_df_to_db(access_top50_df, 'access_top50', engine)

    ann_vol_df = calculate_ann_vol_data(vol_df, access_top50_df, list_of_currencies)
    write_df_to_db(ann_vol_df, 'ann_vol', engine)

    trend_df = calculate_trend_data(price_df, access_top50_df, list_of_currencies)
    write_df_to_db(trend_df, 'trend', engine)

    trend_ann_vol_df = calculate_trend_ann_vol_data(ann_vol_df, access_top50_df, list_of_currencies)
    write_df_to_db(trend_ann_vol_df, 'trend_ann_vol', engine)

    dominance_df = calculate_dominance_data(mcap_df, list_of_currencies)
    write_df_to_db(dominance_df, 'dominance', engine)

    trend_dominance_df = calculate_trend_dominance_data(dominance_df, access_top50_df, list_of_currencies)
    write_df_to_db(trend_dominance_df, 'trend_dominance', engine)

    w_trend_dominance_df = calculate_w_trend_dominance_data(access_top50_df, trend_ann_vol_df, trend_dominance_df, list_of_currencies)
    write_df_to_db(w_trend_dominance_df, 'w_trend_dominance', engine)

    #### Green

    w_trend_df = calculate_w_trend_data(access_top50_df, trend_ann_vol_df, trend_df, list_of_currencies)
    write_df_to_db(w_trend_df, 'w_trend', engine)


    return (
        # New tables
        dominance_df,
        trend_dominance_df,
        w_trend_dominance_df,

        # Blue sheets
        return_df,
        vol_df,
        access_top50_df,
        ann_vol_df,

        # Red sheets
        trend_df,
        trend_ann_vol_df,
        trend_ann_vol_full_df,
        mcapxvolume_total_df,
        mcapxvolume_access_df,
        mcap_txvolume_df,

        # Green sheets
        w_trend_df,
    )


def main():
    logger.info('Starting API -> DB script...')

    data_from_api = get_data_from_api()

    populate_db(data_from_api)


if __name__ == '__main__':
    main()
