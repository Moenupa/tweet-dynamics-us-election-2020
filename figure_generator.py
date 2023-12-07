import os
import logging

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils import load_data, CANDIDATES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns


def save_stackplot(name: str, target: str, ylabel: str, x, **kwargs) -> None:
    """
    Save a stackplot to `figures/<ylabel>` or `figures/quantity` folder
    
    Args:
    - `name`: candidate name
    - `target`: target column name (used in title and filename for saving). e.g. `<emotion> stackplot with #biden tweets`
    - `ylabel`: if 'percentage', y axis will be in %, else in quantity (i.e. no. of tweets)
    - `x`: x axis, usually time
    - `kwargs`: key-value pairs of column name and its value, usually the value is a list of y axis
    """
    os.makedirs(f"figures/{ylabel}/{name}", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # declare type, this helps intellicode and pylint
    fig: Figure = fig
    ax: Axes = ax

    # plot stackplot
    ax.stackplot(x,
                 *list(kwargs.values()),
                 labels=list(k.split("_")[-1] for k in kwargs.keys()),
                 colors=sns.color_palette("Spectral", n_colors=len(kwargs))
                 )
    ax.set_xlabel('Time (MM-DD)')
    ax.set_xlim(x[0], x[-1])

    if ylabel == 'percentage':
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 100)
    elif ylabel == 'quantity':
        ax.set_ylabel('No. of Tweets')
        ax.set_ylim(0)
    else:
        ax.set_ylabel(ylabel)
        ax.set_ylim(0)
    ax.set_title(
        f'{target.replace("_", " ").title()} Stackplot'
        f' with #{name.title()} Tweets')

    fig.legend(loc='outside lower center', ncols=min(10, len(kwargs)))
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    fig.autofmt_xdate()

    fig.savefig(f"figures/{ylabel}/{name}/{target}.png")
    fig.clear()


def plot_candidate_multiclass(candidate_name: str, target_prefix: str = 'stance_biden'):
    """
    Plot the stackplot of each class of target_prefix
    """
    # read and set 'created_at' precision to hour, which helps plotting
    # simply put, plot with xtick by each hour
    par_data = load_data(candidate_name)
    par_data['created_at'] = par_data['created_at'].dt.floor('H')

    # get all columns with target_prefix
    target_col = sorted(list(filter(
        lambda x: x.startswith(target_prefix),
        par_data.columns
    )))
    assert target_col, f'no column found with prefix `{target_prefix}`'

    # group by hour and sum up the count
    stats = par_data.groupby(['created_at'], sort=True)[target_col].sum()
    stats['hr_sum'] = stats.sum(axis=1).round()

    # run a percentage plot
    save_stackplot(candidate_name, target_prefix, 'percentage',
                   x=stats.index, **{
                       col: stats[col] / stats['hr_sum'] * 100
                       for col in target_col
                   })
    # run a quantity plot
    save_stackplot(candidate_name, target_prefix, 'quantity',
                   x=stats.index, **{
                       col: stats[col]
                       for col in target_col
                   })


if __name__ == '__main__':
    target_cols = ['emotion', 'language',
                   'sentiment', 'stance_biden', 'stance_trump']
    # target_cols = ['language']
    for col in target_cols:
        for par_name in CANDIDATES:
            plot_candidate_multiclass(par_name, col)
