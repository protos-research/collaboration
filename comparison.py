# Import of necessary libs and our classes
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from cj_loader import Storer, Extractor, extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features as tsf_ef
import warnings
warnings.filterwarnings("ignore")


def comparison():
    # Init storer object with given data and calculate precense of data
    storer = Storer()

    # Extract info for BTC
    btc = Extractor(storer.mf['bitcoin'])

    # Define parameters to extract BTC bullish, bearish and stable periods. (parameters can be adjusted)
    time_window = 10
    threshold = 0.1

    # Obtain periods of interest
    periods = btc.bull_bear_periods(base_t_len_days=time_window, thresh=threshold)
    stable = periods == 0
    bullish = periods == 1
    bearish = periods == -1

    precense = storer.applicability()

    # Calculation of data completeness matrix and weights
    data_compl = storer.data_completeness()
    weights = data_compl.mean().sort_values()

    # transform boolean matrix to numeric
    weighted = precense.copy()
    for index, row in precense.iterrows():
        weighted.loc[index][:] = row * weights

    # calculate distances between pairs of coins
    distances = pd.DataFrame(squareform(pdist(weighted)), index=weighted.index, columns=weighted.index)

    # Top-level clustering with DBSCAN algorithm
    clustering = DBSCAN(eps=0.3, min_samples=3).fit(distances)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Clusters aggregation
    weighted['label'] = labels
    clusters = {}
    for label in np.unique(labels):
        cl = weighted[weighted['label'] == label]
        clusters[label] = cl

    # Extracting features
    cl_coin_features_all = {}
    cl_coin_features_stable = {}
    cl_coin_features_bullish = {}
    cl_coin_features_bearish = {}
    top_level_outliers = set()
    for label, cluster in clusters.items():
        if label != -1:
            cl_coin_features_all[label] = extract_features(storer, '2014-04-01', coins_set=cluster.index)
            cl_coin_features_stable[label] = extract_features(storer, periods_of_interest=stable, coins_set=cluster.index)
            cl_coin_features_bullish[label] = extract_features(storer, periods_of_interest=bullish, coins_set=cluster.index)
            cl_coin_features_bearish[label] = extract_features(storer, periods_of_interest=bearish, coins_set=cluster.index)
        else:
            top_level_outliers.update(cluster.index.values.tolist())

    top_level_clusters = {
        'all': cl_coin_features_all,
        'bearish': cl_coin_features_bearish,
        'bullish': cl_coin_features_bullish,
        'stable': cl_coin_features_stable
    }

    ts_df = storer.tsfresh_format
    extracted_features = tsf_ef(ts_df, column_id="id", column_sort="time", column_kind="kind", column_value="value")
    extracted_features = impute(extracted_features)
    ef_norm = normalize(extracted_features)

    return n_clusters_, top_level_clusters, top_level_outliers, ef_norm


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    result = result.dropna(axis='columns', how='any')
    return result
