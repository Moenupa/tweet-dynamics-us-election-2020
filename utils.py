import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import logging
import pickle as pkl


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

    # adding columns to the main dataframe from other files
    # these files are same-indexed csv files for prediction results
    for path in file_paths:
        partial_data = pd.read_csv(path, index_col=False)
        if partial_data.shape[1] > 1:
            path_name = os.path.basename(os.path.dirname(path))
            assert path_name, f'{path} -> {path_name}'
            partial_data.rename(columns=lambda x: f'{path_name}_{x}', inplace=True)
        partial_data = pd.get_dummies(partial_data)
        
        # check if the no. of rows are consistent
        assert df.shape[0] == partial_data.shape[0], \
            f'Inconsistent no. of rows: {df.shape} vs {partial_data.shape} ' \
            f'when reading {path}'

        df = pd.concat([df, partial_data], axis=1)
        
    os.makedirs(f'{DATA_ROOT}/cache', exist_ok=True)
    pkl.dump(df, open(pickle_path, 'wb'))

    return df

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    df = load_data('biden', cache=False)
    print(df.head())
    print(type(df['created_at'][0]), df['created_at'][0])
    
    # col_to_inspect = 'created_at'
    # dates = df[col_to_inspect].apply(lambda x: len(x))
    # print(df.loc[dates == 19, col_to_inspect][:5])