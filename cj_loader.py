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


def fin_to_float(val):
    if val != np.nan:
        if not isinstance(val, pd.DatetimeIndex):
            if isinstance(val, str):
                if val == '-':
                    return np.nan
                return float(val.replace(",", ""))
    return val


class Storer(object):
    def __init__(self, data_folder='data'):
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        self.data_folder = os.path.join(file_dir, data_folder)
        self.params = self.__get_params()
        self.all_data = self.__load_data()
        self.coins = self.get_all_coins()
        self.times = self.get_timestamps()

    def __get_params(self):
        return [name.replace('.csv', '') for name in os.listdir(self.data_folder) if
                os.path.isfile(os.path.join(self.data_folder, name))]

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
            df = self.all_data[param]
            all_coins.update(list(df.columns.values))
        return all_coins

    def get_timestamps(self):
        start = self.all_data[self.params[0]].index[0].to_pydatetime()
        finish = self.all_data[self.params[0]].index[0].to_pydatetime()
        for param, data in self.all_data.items():
            if data.index[0].to_pydatetime() < start:
                start = data.index[0].to_pydatetime()
            if data.index[-1].to_pydatetime() > finish:
                finish = data.index[-1].to_pydatetime()

        days = pd.date_range(start, finish, freq='D')
        return days

    def get_mainframe(self):
        # generating dataframe filled with NAs
        midx = pd.MultiIndex.from_product([self.coins, self.params])
        mf_nan = np.empty((len(midx), 1))
        mf_nan[:] = np.nan
        mf = pd.DataFrame(np.nan, self.times, midx)

        # fill dataframe with known values
        # TODO: should be optimized with using numpy vector operations OR pandas builtins
        for param in self.params:
            param_df = self.all_data[param]
            for index, row in param_df.iterrows():
                for co in row.index:
                    mf.at[index, (co, param)] = row[co]

        return mf


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
