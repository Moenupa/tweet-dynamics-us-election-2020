import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import logging
import pickle as pkl


STANCES = ['stance_biden', 'stance_trump']
PREFIXES = ['sentiment', 'emotion', 'language'] + STANCES

CANDIDATES = ['biden', 'trump']
DATA_ROOT = 'data'

def load_data(candidate: str, cache: bool = True) -> pd.DataFrame:
    """
    Load data for a given candidate, structured as follows:\n
    `data/{df_name}/{candidate}.csv`\n
    e.g. `data/src/hashtag_donaldtrump.csv` and `data/lang/trump.csv`
    and merge them column-wise into a single dataframe.
    
    Args:
        `candidate` (str): Candidate name, either `biden` or `trump`
        
    Returns:
        `pd.DataFrame`: Merged dataframe of all the files for the given candidate
    """
    if candidate not in CANDIDATES:
        raise ValueError('Candidate must be either biden or trump')
    
    # if cache enabled, return binary cache file if exists
    pickle_path = f'{DATA_ROOT}/cache/{candidate}.pkl'
    if cache and os.path.exists(pickle_path):
        return pkl.load(open(pickle_path, 'rb'))

    # find all csv files for the given candidate
    file_paths = [
        path
        for path in glob(f'{DATA_ROOT}/*/*.csv')
        if f'{candidate}.csv' in path
    ]
    assert len(file_paths) > 0, 'No data found for candidate'
    logging.info(f'{candidate}: {len(file_paths)} files: {file_paths}')
    
    # find the source file from kaggle, only which contains `hashtag` in its name
    main_path = [path for path in file_paths if 'hashtag' in path]
    assert main_path, 'source not found, plz download dataset from kaggle'
    main_path = main_path[0]
    file_paths.remove(main_path)
    logging.info(f'source: {main_path}')

    df = pd.read_csv(main_path, lineterminator='\n')
    assert len(df.columns) == 21, \
        f'Unexpected no. of columns in source file: {df.shape}'
        
    for col_name in ['created_at', 'collected_at', 'user_join_date']:
        assert col_name in df.columns, \
            f'Column {col_name} not found in source file'
        df[col_name] = pd.to_datetime(df[col_name], errors='raise', exact=False)
    
    # df['weight'] = 1
    def get_weight(x):
        # [1, ratio]
        ratio = 1000
        w1 = min(ratio, x['likes'] + 1)
        w2 = min(ratio, x['retweet_count'] + 1)
        return (w1 + w2) / 2
    df['weight'] = df.apply(get_weight, axis=1)

    # adding columns to the main dataframe from other files
    # these files are same-indexed csv files for prediction results
    for path in file_paths:
        partial_data = pd.read_csv(path, index_col=False)
        
        if partial_data.shape[1] == 1:
            # if the file only contains 1 column, it is a prediction result
            # and we need to convert it to one-hot encoding
            dummies = pd.get_dummies(partial_data)
            partial_data = pd.concat([partial_data, dummies], axis=1)

        else:
            # else, it is a dataframe with multiple columns
            # and we need an argmax to get the predicted class
            path_name = os.path.basename(os.path.dirname(path))
            assert path_name, f'{path} -> {path_name}'
            preds = pd.DataFrame({path_name: partial_data.idxmax(axis=1)})
            partial_data.rename(columns=lambda x: f'{path_name}_{x}', inplace=True)
            partial_data = pd.concat([partial_data, preds], axis=1)
            
        # now, our partial_data will contain:
        # dummies like sentiment_positive: 1, sentiment_negative: 0, sentiment_neutral: 0
        # predictions like sentiment: 'positive'

        # check if the no. of rows are consistent
        assert df.shape[0] == partial_data.shape[0], \
            f'Inconsistent no. of rows: {df.shape} vs {partial_data.shape} ' \
            f'when reading {path}'

        df = pd.concat([df, partial_data], axis=1)
        
    os.makedirs(f'{DATA_ROOT}/cache', exist_ok=True)
    pkl.dump(df, open(pickle_path, 'wb'))

    return df


def get_cols_by_prefix(data: pd.DataFrame, prefix: str) -> list[str]:
    """
    Return all column names that has the given prefix
    
    Args:
    - `data` (pd.DataFrame): Dataframe to search
    - `prefix` (str): Column prefix to search for
    
    Example:
    ```
    >>> df = pd.DataFrame({'sentiment_positive': [1], 'sentiment_neutral': [2], 'emotion_joy': [3]})
    >>> get_cols_by_prefix(df, 'sentiment')
    ['sentiment_positive', 'sentiment_neutral']
    """
    target_col = sorted(list(filter(
        lambda x: x.startswith(f'{prefix}_'),
        data.columns
    )))
    assert target_col, f'no column found with prefix `{prefix}_`'
    return target_col


def get_cols_by_suffix(data: pd.DataFrame, suffix: str) -> list[str]:
    """
    Return all column names that has the given prefix
    
    Args:
    - `data` (pd.DataFrame): Dataframe to search
    - `prefix` (str): Column prefix to search for
    
    Example:
    ```
    >>> df = pd.DataFrame({'sentiment_positive': [1], 'sentiment_neutral': [2], 'emotion_joy': [3]})
    >>> get_cols_by_prefix(df, 'sentiment')
    ['sentiment_positive', 'sentiment_neutral']
    """
    target_col = sorted(list(filter(
        lambda x: x.endswith(f'_{suffix}'),
        data.columns
    )))
    assert target_col, f'no column found with suffix `_{suffix}`'
    return target_col


def merge_data(cache: bool = True):
    pickle_path = f'{DATA_ROOT}/cache/merged.pkl'
    if cache and os.path.exists(pickle_path):
        return pkl.load(open(pickle_path, 'rb'))
    
    assert len(CANDIDATES) == 2, 'only 2 candidates supported'
    cand1 = load_data(CANDIDATES[0], cache=cache)
    cand2 = load_data(CANDIDATES[1], cache=cache)
    cand1['hashtag'] = CANDIDATES[0]
    cand2['hashtag'] = CANDIDATES[1]
    
    df = pd.concat([cand1, cand2], axis=0)
    if cache:
        pkl.dump(df, open(pickle_path, 'wb'))
    
    return df


def dist(df: pd.DataFrame, grouped_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    """
    Get distribution of target columns, grouped by grouped columns such that
    `sum(target_cols) = 1` and `distribution = target_cols*weight`
    
    Args:
    - `df` (pd.DataFrame): Dataframe to group by
    - `grouped_cols` (list[str]): Columns to group by
    - `target_cols` (list[str]): Columns to sum up
    
    e.g.
    ```
    >>> groupby(load_data('biden'), ['created_at'], get_cols_by_prefix(df, 'sentiment'))
                sentiment_negative  sentiment_neutral  sentiment_positive   weight
    created_at                                                                     
    2020-10-15            0.542378           0.326894            0.130728  46541.0 
    2020-10-16            0.474969           0.324787            0.200244  58563.0 
    2020-10-17            0.495558           0.337216            0.167225  34063.5 
    2020-10-18            0.506811           0.329069            0.164120  31878.0 
    2020-10-19            0.493768           0.346814            0.159418  32154.5 
    ```
    """
    assert set(grouped_cols) <= set(df.columns), \
        f'grouped columns not found {grouped_cols}'
    assert set(target_cols) <= set(df.columns), \
        f'target columns not found {target_cols}'
    assert 'weight' in df.columns, f'weight column not found'

    # apply weight to target_cols
    all_cols = grouped_cols + target_cols + ['weight']
    sum_cols = target_cols + ['weight']
    
    # copy related dataframe columns only
    ret = df[all_cols].copy()
    for col in target_cols:
        ret[col] = ret[col] * ret['weight']
    # perform groupby
    ret = ret.groupby(grouped_cols, sort=True)[sum_cols].sum()
    
    # normalize by weight
    for col in target_cols:
        ret[col] = ret[col] / ret['weight']
    
    return ret


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    
    df = load_data('biden')
    df['created_at'] = df['created_at'].dt.floor('D')
    print(dist(df, ['created_at'], get_cols_by_prefix(df, 'sentiment')).head())
