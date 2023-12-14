import os

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils import (
    load_data, 
    merge_data,
    get_cols_by_prefix, 
    get_cols_by_suffix, 
    dist,
    CANDIDATES,
    PREFIXES
)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter, DayLocator
import seaborn as sns
from itertools import product
from tqdm import tqdm


PLOT_KW_STACK = {'figsize':(8, 6), 'dpi':200}
PLOT_KW = {'figsize':(8, 4.5), 'dpi':200}

TIME_SCALES = ['H', '6H', '12H', 'D']

GOP = '#d6342d'
DNC = '#1941c3'

ELECTION_PALETTE = [DNC, GOP]
ELECTION_ORDER = ['biden', 'trump']


def fmt(string: str) -> str:
    return string.replace("_", " ").title()


def save_stackplot(dirname: str, candidate: str, task: str, percentage: bool, x, **kwargs) -> None:
    """
    Save a stackplot to `figures/<dirname>/<candidate>/<task>.png`.
    If `percentage=True`, cap y axis to 100 otherwise no y upper limit
    
    Args:
    - `candidate`: candidate name
    - `target`: target column name (used in title and filename for saving). e.g. `<emotion> stackplot with #biden tweets`
    - `percentage`: whether to plot in percentage or quantity
    - `task`: task name, ONLY used in saving folder name
    - `x`: x axis, usually time
    - `annotation`: annotation to add to the plot, used in title and filename only
    - `kwargs`: key-value pairs of column name and its value, usually the value is a list of y axis
    """
    fig, ax = plt.subplots(**PLOT_KW_STACK)
    # declare type, this helps intellicode and pylint
    fig: Figure = fig; ax: Axes = ax

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

    if percentage:
        ax.set_ylabel('Percentage (%)');    ax.set_ylim(0, 100)
    else:
        ax.set_ylabel('No. of Tweets');     ax.set_ylim(0)
    ax.set_title(f'{fmt(task)} Stackplot with #{candidate.title()} Tweets')

    fig.legend(loc='outside lower center', ncols=min(10, len(kwargs)))

    os.makedirs(f"figures/{dirname}/{candidate}", exist_ok=True)
    fig.savefig(f"figures/{dirname}/{candidate}/{task}.png")
    plt.close(fig)


def plot_candidate_multiclass(candidate_name: str, prefix: str = 'stance_biden', time_scale: str = None):
    """
    Plot the stackplot of each class of target_prefix
    """
    # read and set 'created_at' precision to hour, which helps plotting
    # simply put, plot with xtick by each hour
    par_data = load_data(candidate_name)
    if time_scale != None:
        par_data['created_at'] = par_data['created_at'].dt.floor(time_scale)
    else:
        par_data['created_at'] = par_data['created_at'].dt.floor('D')

    # get all columns with target_prefix
    # group by time_scale and sum up the count
    target_col = get_cols_by_prefix(par_data, prefix)
    stats = dist(par_data, ['created_at'], target_col)

    if time_scale == None:
        # do everything normally
        # run a percentage plot, percentage=True
        save_stackplot('percentage', candidate_name, prefix, True,
                    x=stats.index, **{
                        col: stats[col] * 100
                        for col in target_col
                    })
        # run a quantity plot, percentage=False
        save_stackplot('quantity', candidate_name, prefix, False,
                    x=stats.index, **{
                        col: stats[col] * stats['weight']
                        for col in target_col
                    })
    else:
        # only run percentage plot
        save_stackplot(time_scale, candidate_name, prefix, True,
                       x=stats.index, **{
                           col: stats[col] * 100
                           for col in target_col
                       })


def cal_score(df: pd.DataFrame, prefix: str, score_name: str) -> pd.DataFrame:
    # score = (pos - neg), and because sum(pos, neg, neu) = 1, score in [-1, 1]
    neg = f'{prefix}_negative'
    pos = f'{prefix}_positive'
    df[score_name] = df.apply(lambda x: (x[pos] - x[neg]), axis=1)
    df.rename(columns={'weight': 'No. of Tweets'}, inplace=True)
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
    target_col = get_cols_by_prefix(par_data, target_prefix)
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
    scatter_data = dist(par_data, ['lat', 'long'], target_col)
    scatter_data = cal_score(scatter_data, target_prefix, score_name)
    fig, ax = plt.subplots(**PLOT_KW)
    usa.boundary.plot(ax=ax, linewidth=1, alpha=0.1, color='grey')
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
    state_data = dist(par_data, ['state_code'], target_col)
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


def plot_candidate_correlation(candidate_name: str, time_scale: str = 'H', aim: tuple = ('negative', 'anger')):
    """
    Plot the scatter between sentiment and emotion of each class
    """
    # read and set 'created_at' precision to hour, which helps plotting
    # simply put, plot with xtick by each hour
    par_data = load_data(candidate_name)
    par_data['created_at'] = par_data['created_at'].dt.floor(time_scale)

    # get all columns with emotion and sentiment seperately
    full_col_x = get_cols_by_prefix(par_data, 'sentiment')
    full_col_y = get_cols_by_prefix(par_data, 'emotion')
    
    target_col_x = sorted(list(filter(
        lambda x: x.endswith(aim[0]),
        full_col_x
    )))
    assert target_col_x, f'no column found with suffix `{aim[0]}`'

    target_col_y = sorted(list(filter(
        lambda x: x.endswith(aim[1]),
        full_col_y
    )))
    assert target_col_y, f'no column found with suffix `{aim[0]}`'

    # group by time_scale and sum up the count
    stats_x = par_data.groupby(['created_at'], sort=True)[target_col_x].sum()
    sum_x = par_data.groupby(['created_at'], sort=True)[full_col_x].sum().sum(axis=1).round()

    stats_y = par_data.groupby(['created_at'], sort=True)[target_col_y].sum()
    sum_y = par_data.groupby(['created_at'], sort=True)[full_col_y].sum().sum(axis=1).round()

    row = [stats_x[i] / sum_x * 100 for i in target_col_x]
    col = [stats_y[i] / sum_y * 100 for i in target_col_y]
    
    # save the scatter plot
    fig, ax = plt.subplots(**PLOT_KW_STACK)
    
    # declare type, this helps intellicode and pylint
    fig: Figure = fig
    ax: Axes = ax

    # plot scatter
    color = (0, 66/255., 202 / 255.) if candidate_name == 'biden' \
        else (233/255., 20/255., 30/255.)
    sns.regplot(x=row[0], y=col[0], ax=ax, color=color, 
                truncate=False, line_kws={'color': (0.5, 0.5, 0.5)}, scatter_kws={'s': 15})
    ax.set_xlabel(f'Percentage of Sentiment {aim[0]} (%)')
    ax.set_ylabel(f'Percentage of Emotion {aim[1]} (%)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
 
    ax.set_title(
        f'Scatter of {aim[0]} and {aim[1]} with #{candidate_name.title()} Tweets Grouped by {time_scale}')

    os.makedirs(f"figures/correlation/{candidate_name}", exist_ok=True)
    fig.savefig(f"figures/correlation/{candidate_name}/{aim[0]}_{aim[1]}_{time_scale}.png")
    plt.close(fig)


def plot_indictor_correlation() -> None:
    """
    Plot the KDE plot of combinations of indicator values in
    `['sentiment', 'stance_biden', 'stance_trump']`
    """
    def parser(l: list) -> str:
        def cats(s: str) -> str:
            if s.endswith("negative"):
                return "n"
            elif s.endswith("positive"):
                return "p"
            else:
                return "u"
        return "".join(cats(i) for i in l[:3])

    os.makedirs(f"figures/correlation/indicators", exist_ok=True)
    data = merge_data()
    all_cols = [get_cols_by_prefix(data, prefix) 
                for prefix in ['sentiment', 'stance_biden', 'stance_trump']]
    for cols in tqdm(product(*all_cols), total=27):
        cols = list(cols) + ['hashtag']
        g = sns.pairplot(
            data=data.sample(n=10_000)[cols],
            palette=ELECTION_PALETTE,
            hue="hashtag",
            hue_order=ELECTION_ORDER,
            kind="kde",
            plot_kws={"fill": True, "alpha": 0.3}
            # thresh=.1,
        )
        g.savefig(
            f"figures/correlation/indicators/{parser(cols)}.png", dpi=200)


def plot_overall_distribution() -> None:
    os.makedirs(f"figures/overall", exist_ok=True)
    data = merge_data()
    for prefix in PREFIXES:
        fig, ax = plt.subplots(**PLOT_KW)
        sns.countplot(data, ax=ax, 
                      x=prefix,
                      stat="percent",
                      palette=ELECTION_PALETTE,
                      hue="hashtag", 
                      hue_order=ELECTION_ORDER)
        
        fig.savefig(f'figures/overall/{prefix}.png')
        plt.close(fig)


if __name__ == '__main__':
    plot_indictor_correlation()
    exit(0)
    
    targets = ['emotion', 'sentiment', 'stance_biden', 'stance_trump']
    for ts in TIME_SCALES:
        for prefix in PREFIXES:
            for par_name in CANDIDATES:
                plot_candidate_multiclass(par_name, prefix, ts)
    
    for prefix in PREFIXES:
        for par_name in CANDIDATES:
            plot_candidate_multiclass(par_name, prefix)
    
    for col in ['sentiment', 'stance_biden', 'stance_trump']:
        for par_name in CANDIDATES:
            plot_candidate_geo(par_name, col)

    aims = [('negative', 'anger'), ('positive', 'joy')]
    for ts in TIME_SCALES:
        for par_name in CANDIDATES:
            for aim in aims:
                plot_candidate_correlation(par_name, ts, aim)