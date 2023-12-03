import logging
from utils import load_data, CANDIDATES

import numpy as np
import pandas as pd
from datetime import datetime
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
    
    for _, (tweet_datetime, column_value) in par_data[['created_at', target_col]].iterrows():
        # logging.info(i, tweet_datetime, column_value)
        if not isinstance(tweet_datetime, str):
            # print("err! {2:}th dateStr is not string, type = {0:}, val = {1:}".format(type(dateStr), dateStr, i))
            continue
        if '.' in tweet_datetime:
            pre, post = tweet_datetime.split('.')
            tweet_datetime = pre + '.' + post[:6]
            date = datetime.strptime(tweet_datetime, '%Y-%m-%d %H:%M:%S.%f')
        else:
            date = datetime.strptime(tweet_datetime, '%Y-%m-%d %H:%M:%S')
        date_obj = datetime(date.year, date.month, date.day, date.hour)
        
        counter_by_date[date_obj] += 1
        counter_by_value[column_value][date_obj] += 1

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


if __name__ == '__main__':
    for par_name in CANDIDATES:
        plot_candidate(par_name, 'emotion')
