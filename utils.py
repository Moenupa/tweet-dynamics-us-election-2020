import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import logging
from collections import Counter

DATA_ROOT = 'data'

def load_data(candidate: str) -> pd.DataFrame:
    """
    Load data for a given candidate, structured as follows:\n
    `data/{df_name}/{candidate}.csv`\n
    e.g. `data/src/hashtag_donaldtrump.csv` and `data/lang/trump.csv`
    
    Args:
        `candidate` (str): Candidate name, either `biden` or `trump`
        
    Returns:
        `pd.DataFrame`: Merged dataframe of all the files for the given candidate
    """
    if candidate not in ['biden', 'trump']:
        raise ValueError('Candidate must be either biden or trump')

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

    # adding columns to the main dataframe from other files
    # these files are single-column csv files for prediction results
    for path in file_paths:
        column_data = pd.read_csv(path, index_col=False).iloc[:, 0]
        # to be safe, we need to do this
        # df[column_data.name] = column_data.values
        # https://stackoverflow.com/questions/12555323/how-to-add-a-new-column-to-an-existing-dataframe
        # but actually, index are never changed in the two files
        # so assigning values is enough
        df[column_data.name] = column_data
        
    assert len(df.columns) == 21 + len(file_paths), \
        f'Unexpected no. of columns after merging: {df.shape}'

    # we are safe to return this merged dataframe
    return df

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    df = load_data('biden')
    print(df.head())