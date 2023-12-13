import os
import logging

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils import load_data, CANDIDATES

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter, DayLocator
import seaborn as sns


PLOT_KW_STACK = {'figsize':(8, 6), 'dpi':200}
PLOT_KW = {'figsize':(8, 4.5), 'dpi':200}


def fmt(string: str) -> str:
    return string.replace("_", " ").title()


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
    fig, ax = plt.subplots(**PLOT_KW_STACK)
    
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
    ax.xaxis.set_major_locator(DayLocator(interval=2))
    ax.xaxis.set_minor_locator(DayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    fig.autofmt_xdate()

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
        f'{fmt(target)} Stackplot with #{name.title()} Tweets')

    fig.legend(loc='outside lower center', ncols=min(10, len(kwargs)))

    os.makedirs(f"figures/{ylabel}/{name}", exist_ok=True)
    fig.savefig(f"figures/{ylabel}/{name}/{target}.png")
    plt.close(fig)


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


def cal_score(df: pd.DataFrame, prefix: str, score_name: str) -> pd.DataFrame:
    # using sentiment because its easy, only to compute no. of tweets
    cols = [f'sentiment_{s}' for s in ['negative', 'positive', 'neutral']]
    s = 'No. of Tweets'
    df[s] = df[cols].sum(axis=1)
    
    neg = f'{prefix}_negative'
    pos = f'{prefix}_positive'
    assert neg in df.columns and pos in df.columns, f'no neg and pos cols'
    df[score_name] = df.apply(lambda x: (x[pos] - x[neg]) / x[s], axis=1)
    # df[score_name] = (df[pos]-df[neg]) / df[s]
    # print(df[pos].value_counts().sort_index())
    # exit(0)
    return df
    

def plot_candidate_geo(candidate_name: str, target_prefix: str = 'stance_biden'):
    # download from US Census
    states = [ 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    usa = gpd.read_file('shp/cb_2022_us_state_5m.shp')
    usa = usa[usa['STUSPS'].isin(states)]
    
    par_data = load_data(candidate_name)
    # get all columns with target_prefix
    target_col = sorted(list(filter(
        lambda x: x.startswith(target_prefix),
        par_data.columns
    )))
    assert target_col, f'no column found with prefix `{target_prefix}`'
    assert len(target_col) == 3, f'column is not neg-neu-pos form {target_col}'
    
    # drop tweets that has no geo location info
    par_data = par_data.dropna(subset=['lat', 'long'])
    
    # filter lat and long to get only us mainland tweets
    lat = par_data['lat']
    long = par_data['long']
    par_data = par_data[
        (lat > 19) & (lat < 50) & 
        (long > -128) & (long < -50) & 
        (par_data['country'] == 'United States of America')
    ]
    score_name = f'{fmt(target_prefix)} Score'
    
    # draw graph according to lat and long onto a map
    scatter_data = par_data.groupby(['lat', 'long']).sum(numeric_only=True)
    scatter_data = cal_score(scatter_data, target_prefix, score_name)
    fig, ax = plt.subplots(**PLOT_KW)
    usa.boundary.plot(ax=ax, linewidth=1, alpha=0.1, color='grey')
    # sns.relplot(data=scatter_data, x="total_bill", y="tip", col="time", hue="day", style="day", kind="scatter")
    sns.scatterplot(scatter_data, 
                    x='long', y='lat', alpha=0.2, 
                    hue=score_name, palette='Spectral',
                    size='No. of Tweets', sizes=(20, 200),
                    ax=ax, legend=False)
    # leg = plt.legend()
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1)
    ax.set_title(f'{fmt(target_prefix)} Score HeatMap with #{candidate_name.title()} Tweets')
    os.makedirs(f"figures/national/{candidate_name}", exist_ok=True)
    fig.savefig(f"figures/national/{candidate_name}/{target_prefix}.png")
    plt.close(fig)
    
    # draw graph for each state, categorized, then display on map
    state_data = par_data.groupby(['state_code']).sum(numeric_only=True)
    state_data = cal_score(state_data, target_prefix, score_name)
    state_data = usa.merge(state_data, left_on='STUSPS', right_index=True, how='left')
    
    cmap = sns.color_palette('Spectral', as_cmap=True)
    norm = mcolors.Normalize(vmin=-1, vmax=1, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(**PLOT_KW)
    ax: Axes = ax
    ax.set_xlabel('long')
    ax.set_ylabel('lat')
    ax.set_title(f'{fmt(target_prefix)} Score State HeatMap with #{candidate_name.title()} Tweets')
    state_data.plot(ax=ax, linewidth=1, alpha=0.1)

    for row in state_data.itertuples():
        vf = state_data[state_data.STUSPS==row.STUSPS]
        c = mcolors.to_hex(mapper.to_rgba(vf[score_name]))
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.9')

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(mapper, cax=cax)
    os.makedirs(f"figures/states/{candidate_name}", exist_ok=True)
    fig.savefig(f"figures/states/{candidate_name}/{target_prefix}.png")
    plt.close(fig)


if __name__ == '__main__':
    targets = ['emotion', 'language',
               'sentiment', 'stance_biden', 'stance_trump']
    targets = []
    for col in targets:
        for par_name in CANDIDATES:
            plot_candidate_multiclass(par_name, col)
    
    targets = ['sentiment', 'stance_biden', 'stance_trump']
    for col in targets:
        for par_name in CANDIDATES:
            plot_candidate_geo(par_name, col)
