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
        if candidate in path
    ]
    assert len(file_paths) > 0, 'No data found for candidate'
    logging.info(f'{candidate}: {len(file_paths)} files: {file_paths}')
    
    # find the source file from kaggle, only which contains `hashtag` in its name
    main_path = [path for path in file_paths if 'hashtag' in path][0]
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
    # these files are single-column csv files for prediction results
    for path in file_paths:
        column_data = pd.read_csv(path, index_col=False).iloc[:, 0]
        
        # check if the no. of rows are consistent
        assert df.shape[0] == column_data.shape[0], \
            f'Inconsistent no. of rows: {df.shape} vs {column_data.shape}'

        # to be safe, we need to do `df[column_data.name] = column_data.values`
        # https://stackoverflow.com/questions/12555323/how-to-add-a-new-column-to-an-existing-dataframe
        # but actually, index are never changed in the two files
        # so assigning values is enough
        df[column_data.name] = column_data
        
    assert len(df.columns) == 21 + len(file_paths), \
        f'Unexpected no. of columns after merging: {df.shape}'
        
    if cache:
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