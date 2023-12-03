import logging
from utils import load_data, CANDIDATES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from tqdm import tqdm
from collections import defaultdict


def plot_candidate(candidate_name: str, target_col: str = 'sent'):
    par_data = load_data(candidate_name)
    if target_col not in par_data.columns:
        raise ValueError(f'Column {target_col} not found in data')
    
    # counter by value of the column with struct like:
    # { column_value: { date: count } }
    # e.g. { 'positive': { datetime(2020, 10, 1): 10 } }
    # means there are 10 tweets with positive sentiment on 2020-10-1
    counter_by_value = {
        # value: counter
        value: defaultdict(int)
        for value in par_data[target_col].unique()
    }
    counter_by_date = defaultdict(int)
    
    for idx, (dt_obj, col_val) in tqdm(par_data[['created_at', target_col]].iterrows(), total=par_data.shape[0], desc=candidate_name):
        # logging.info(i, tweet_datetime, column_value)
        if not isinstance(dt_obj, pd.Timestamp):
            logging.error(f"err! At {idx}th type={type(dt_obj)}, val={dt_obj}")
            continue
    
        # truncate by hour, by setting min, s, ms to 0
        # this avoids plotting too many xticks
        # by minute is too sharp, by day is too coarse
        dt_obj = dt_obj.replace(minute=0, second=0, microsecond=0)
        
        counter_by_date[dt_obj] += 1
        counter_by_value[col_val][dt_obj] += 1

    x = list(sorted(counter_by_date.keys()))
    for each_date in x:
        total = counter_by_date[each_date]
        for each_value in counter_by_value.keys():
            counter_by_value[each_value][each_date] *= 100 / total

    sorted_counter_by_value = {
        value: [pct_daily for _, pct_daily in sorted(counter.items())]
        for value, counter in sorted(counter_by_value.items())
    }
    # print(sorted_counter_by_value['neutral'])

    plt.figure(figsize=(10, 5))
    plt.stackplot(x, 
                  *tuple(sorted_counter_by_value.values()), 
                  labels=list(sorted_counter_by_value.keys()), 
                  colors=sns.color_palette(
                      "Spectral", 
                      n_colors=len(sorted_counter_by_value))
                  )
    # plt.plot(x, yneg, color='#E6645C', linestyle='-')
    # plt.plot(x, np.array(yneg) + np.array(yneu), color='#5CE687', linestyle='-')
    plt.xlabel('Time (MM-DD)')
    plt.ylabel('Percentage (%)')
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 100)
    plt.title(f'Tweets\' Sentiment Percentage with #{candidate_name.title()}')
    plt.legend(loc='lower right', ncol=3)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f"figures/{candidate_name}_{target_col}.png", dpi=300)
    plt.clf()


if __name__ == '__main__':
    for par_name in CANDIDATES:
        plot_candidate(par_name, 'emotion')
