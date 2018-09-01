"""
Requirements & Limitations:
- module takes csv files of time series
- name of csv file represents a name of a parameter
- csv file should present data for different cryptocurrencies
- first column should be a timestamp
- data should be provide in day-by-day basis with timestamps at 00:00
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import skew, kurtosis


def fin_to_float(val):
    if val != np.nan:
        if not isinstance(val, pd.DatetimeIndex):
            if isinstance(val, str):
                if val == '-':
                    return np.nan
                return float(val.replace(",", ""))
    return val


class Storer(object):
    def __init__(self, data_folder='data', reload=False):
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        self.data_folder = os.path.join(file_dir, data_folder)
        self.params = self.__get_params()
        self.coins = None
        self.times = None
        self.mf = None

        self.additional_params = [
            'days_exist',  # number of days cryptocurrency existing
            'rate_btc',  # price of crypto in BTC
            'mcap_ratio',  # ratio of crypto market cap of all market cap
            'trad_to_trans_vol',  # trades volume / transactions volume
            'trans_per_address',  # transactions volume / active_address
            'tx_per_address',  # tx_count / active_adress
            'mdiff_to_volatility',  # mining_difficulty / volatility
            'volatility_to_mdiff'  # volatility / mining_difficulty
        ]

        if reload:
            self.__all_data = self.__load_data()
            self.coins = self.get_all_coins()
            self.times = self.get_timestamps()
            self.mf = self.get_mainframe()
            self.save_mainframe(self.mf)
        else:
            self.load_mainframe()

    def __get_params(self):
        base_params = [name.replace('.csv', '') for name in os.listdir(self.data_folder) if
                       os.path.isfile(os.path.join(self.data_folder, name))]
        return base_params

    def __load_data(self):
        all_data = dict()
        for param in self.params:
            filename = os.path.join(self.data_folder, param + '.csv')
            filename = os.path.abspath(os.path.realpath(filename))
            df = pd.read_csv(filename)
            df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            df = df.set_index(pd.DatetimeIndex(df['timestamp']), drop=True)
            del df['timestamp']
            df = df.applymap(fin_to_float)
            all_data[param] = df
        return all_data

    def __validate_data(self):
        """
        To be implemented if necessary
        :return:
        """
        pass

    def get_all_coins(self):
        all_coins = set()
        for param in self.params:
            df = self.__all_data[param]
            all_coins.update(list(df.columns.values))
        return all_coins

    def get_timestamps(self):
        start = self.__all_data[self.params[0]].index[0].to_pydatetime()
        finish = self.__all_data[self.params[0]].index[0].to_pydatetime()
        for param, data in self.__all_data.items():
            if data.index[0].to_pydatetime() < start:
                start = data.index[0].to_pydatetime()
            if data.index[-1].to_pydatetime() > finish:
                finish = data.index[-1].to_pydatetime()

        days = pd.date_range(start, finish, freq='D')
        return days

    def get_mainframe(self, include_additional_params=True):
        if include_additional_params:
            all_params = self.params + self.additional_params
        else:
            all_params = self.params

        # generating dataframe filled with NAs
        midx = pd.MultiIndex.from_product([self.coins, all_params])
        mf_nan = np.empty((len(midx), 1))
        mf_nan[:] = np.nan
        mf = pd.DataFrame(np.nan, self.times, midx)

        # fill dataframe with known values
        # TODO: should be optimized with using numpy vector operations OR pandas builtins
        for param in self.params:
            param_df = self.__all_data[param]
            for index, row in param_df.iterrows():
                for co in row.index:
                    mf.at[index, (co, param)] = row[co]

        # calculate total market capitalization for each day
        tot_mcap = mf['bitcoin', 'mcap']
        for index, row in mf.iterrows():
            mcap = np.nansum([mf.at[index, (co, 'mcap')] for co in self.coins])
            tot_mcap.at[index] = mcap

        # fill additional parameters
        for param in self.additional_params:
            for co in self.coins:
                # captain mode on: each additional parameter has it's own formula
                if param == 'rate_btc':
                    mf[co, param] = (mf[co, 'high'] + mf[co, 'low']) / (mf['bitcoin', 'high'] + mf['bitcoin', 'low'])
                elif param == 'trad_to_trans_vol':
                    mf[co, param] = mf[co, 'volume'] / mf[co, 'txvolume']
                elif param == 'trans_per_address':
                    mf[co, param] = mf[co, 'txvolume'] / mf[co, 'active_address']
                elif param == 'tx_per_address':
                    mf[co, param] = mf[co, 'txcount'] / mf[co, 'active_address']
                elif param == 'mdiff_to_volatility':
                    mf[co, param] = mf[co, 'mining_difficulty'] / mf[co, 'volatility']
                elif param == 'volatility_to_mdiff':
                    mf[co, param] = mf[co, 'volatility'] / mf[co, 'mining_difficulty']
                elif param == 'days_exist':
                    exists = False
                    days_exists = 0
                    for index, row in mf[co].iterrows():
                        if exists:
                            days_exists += 1
                            mf.at[index, (co, param)] = days_exists
                        else:
                            if not all(np.isnan(row.values)):
                                exists = True
                                days_exists = 1
                                mf.at[index, (co, param)] = days_exists
                elif param == 'mcap_ratio':
                    for index, row in mf[co].iterrows():
                        mf[co, param] = mf[co, 'mcap'] / tot_mcap[index]

        mf.index.name = 'timestamp'
        return mf

    def prep_existance_plot(self):
        tot_cryptos = len(self.coins)
        unit = 100.0 / tot_cryptos

        # create blank of existance frame
        ex_df = self.mf['bitcoin', 'mcap']

        for timestamp in self.times:
            ex_df.at[timestamp] = np.count_nonzero(
                ~np.isnan(self.mf.loc[timestamp, (slice(None), 'days_exist')])) * unit
        return ex_df

    @staticmethod
    def save_mainframe(mf: pd.DataFrame, name="mainframe.csv"):
        mf.to_csv(name)

    def load_mainframe(self, name="mainframe.csv"):
        self.mf = pd.read_csv(name, header=[0, 1], index_col=[0], parse_dates=[0])
        self.times = self.mf.index
        self.coins = set([self.mf.columns[i][0] for i in range(len(self.mf.columns))])

    def applicability(self):
        valid_params = [p for p in self.params + self.additional_params if p != 'days_exist']

        # calculate applicability of parameters to coins
        applicability = pd.DataFrame(index=self.coins, columns=self.params)
        for coin in self.coins:
            for param in valid_params:
                applicability.at[coin, param] = not all(np.isnan(self.mf[coin, param].values))
        return applicability

    def data_completeness(self):
        """
        Method for defining ratio of data precense for each date for each parameter.
        Number of existing coins for each date is taken into account
        :return: table with ratio of precense with dates as index and parameters as columns
        """

        # create empty df
        compl_df = pd.DataFrame(index=self.times, columns=self.params)
        valid_params = [p for p in self.params + self.additional_params if p != 'days_exist']

        # calculate applicability of parameters to coins
        applicability = self.applicability()

        for ts in self.times:
            coins_exist_df = ~np.isnan(self.mf.loc[ts, (slice(None), 'days_exist')])
            coins_exist = coins_exist_df[coins_exist_df == True].index.get_level_values(0)

            for param in valid_params:
                coin_df = ~np.isnan(self.mf.loc[ts, (slice(None), param)])
                filled_coins = coin_df[coin_df == True].index.get_level_values(0)
                max_filled = [co for co in coins_exist if applicability[param][co]]
                compl_ratio = len(filled_coins) / len(max_filled)

                # compl_num = np.count_nonzero(~np.isnan(self.mf.loc[ts, (slice(None), param)]))
                # compl_ratio = compl_num / coins_exist_num
                compl_df.at[ts, param] = compl_ratio

        return compl_df


class Extractor(object):
    """
    Class for dealing with data relevant to single coin.
    """
    features_num = 5  # don't forget to change it in case of adding new features

    def __init__(self, coin_data: pd.DataFrame, cut_start=None):
        self.data = self.__crop_since_exist_start(coin_data, cut_start)
        self.normalized = self.__norm()
        self.cleared = self.fix_na()
        self.features = self.__features()

    @staticmethod
    def __crop_since_exist_start(coin_data, cut_start):
        if cut_start:
            return coin_data[coin_data['days_exist'] >= 1].loc[cut_start:]
        return coin_data[coin_data['days_exist'] >= 1]

    def na_distr(self, silent=True, basic_only=False):
        nas = dict()
        tot = float(self.data.shape[0])

        if basic_only:
            params = ['open', 'high', 'low', 'close', 'volume', 'mcap']
        else:
            params = self.data.columns

        for param in params:
            nas[param] = self.data[param].isnull().sum(axis=0) / tot

        if not silent:
            for k, v in nas.items():
                print("{:<20}: {}".format(k, round(v, 2)))
        return nas

    def bull_bear_periods(self, base_t_len_days=60, thresh=0.2) -> np.array:
        """
        Function for splitting coin timeline for bullish/bearish/stable periods
        :param base_t_len_days: length of time window for searching
        :param thresh: minimal price difference for start and end of period to consider bullish/bearish
        :return: pd.Series where -1 corresponds to bear, +1 bull, 0 stable
        """

        # Initialize the empty DataFrame with axis:
        # X = date (start of window)
        # Y = position of day in window (min: 0, max: base_t_len_days-1

        scoring_mtrx = pd.DataFrame(index=range(base_t_len_days), columns=self.data.index)
        one_line = np.zeros(len(self.data.index), dtype=int)

        for start_idx in range(len(self.data.index) - base_t_len_days):
            window_start = self.data.index[start_idx]
            window_end = self.data.index[start_idx + base_t_len_days - 1]
            price_start = 0.5 * (self.data['high'][window_start] + self.data['low'][window_start])
            price_end = 0.5 * (self.data['high'][window_end] + self.data['low'][window_end])
            price_ratio = (price_end - price_start) / price_start

            if price_ratio > thresh:
                one_line[start_idx] = 1
            elif price_ratio < -thresh:
                one_line[start_idx] = -1

        states = np.empty(len(self.data.index))
        states[:] = np.nan
        state_cand = None
        len_consecutive = 0
        for i in range(len(self.data.index) - 1):
            if state_cand is None:
                state_cand = one_line[i]
                continue

            if one_line[i] == state_cand:
                len_consecutive += 1

            else:
                if len_consecutive >= base_t_len_days:
                    interval = np.r_[i - len_consecutive:i]
                    states[interval] = state_cand

                len_consecutive = 0
                state_cand = one_line[i]

        return states

    def fix_na(self, method='linear'):
        cleared = self.data.interpolate(method=method)
        return cleared

    def __features(self):
        """
        Method for extracting basic features:
        Means, Medians, Standard deviations, Skewness, Kurtosis for each parameter
        :return: dictionary(param: dict_of_features(feature: value))
        """
        params_dict = {}
        params = self.cleared.columns
        # we should go by each param
        for param in params:
            t_serie = self.cleared[param].dropna()
            if t_serie.shape[0] == 0:
                params_dict[param] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'stddev': np.nan,
                    'skewns': np.nan,
                    'kurtos': np.nan
                }
            else:
                params_dict[param] = {
                    'mean': np.mean(t_serie),
                    'median': np.median(t_serie),
                    'stddev': np.std(t_serie),
                    'skewns': skew(t_serie),
                    'kurtos': kurtosis(t_serie)
                }
        return params_dict

    def __norm(self):
        """
        Function to normalize the data
        :return:
        """
        cleared = self.data.interpolate(method='akima')
        return cleared
        # min_max_scaler = preprocessing.MinMaxScaler()
        # norm = min_max_scaler.fit_transform(cleared)
        # return pd.DataFrame(norm, columns=self.data.columns, index=self.data.index)


def extract_features(st: Storer, start_date, coins_set=None):
    """
    Function to extract features of all coins inside cluster
    :param coins_set: list with coin names of particular cluster
    :param start_date: moment of timeline start, f.e. '2014-01-31'
    :param st: Storer object
    :return: table with coins as index and features as columns
    """
    if coins_set is None:
        coins_set = st.coins
    # obtain full features names to init empty table
    f_names = []
    btc = Extractor(st.mf['bitcoin'], cut_start=start_date)
    coin_fe = btc.features
    for param, fe_dict in coin_fe.items():
        prefix = param
        for feature, value in fe_dict.items():
            f_names.append(prefix + "_" + feature)

    # Init empty table of coins features
    ef_table = pd.DataFrame(index=coins_set, columns=f_names)

    for coin in coins_set:
        coin_ex = Extractor(st.mf[coin])
        coin_fe = coin_ex.features
        for param, fe_dict in coin_fe.items():
            prefix = param
            for feature, value in fe_dict.items():
                ef_table[prefix + "_" + feature][coin] = value

    return ef_table


def get_params():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(file_dir, 'data')
    return [name.replace('.csv', '') for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))]


def load_data(param):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(file_dir, 'data/' + param + '.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    df = pd.read_csv(filename)
    return df


def test():
    for param in get_params():
        data = load_data(param)
        print("{:>17}:{:4} x {}".format(param, data.shape[0], data.shape[1]))

    # nulls = data.isnull().any(axis=1).tolist()
    # last_null = 0
    # for i in range(len(nulls)):
    #     if nulls[i]:
    #         last_null = i
    #
    # print("last N/A observed: {}/{}".format(last_null, len(nulls)))
