import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from palettable.tableau import (Tableau_20, Tableau_10, TableauLight_10,
        TableauMedium_10,PurpleGray_6, PurpleGray_12, ColorBlind_10, Gray_5)
from palettable.colorbrewer.sequential import Greens_5, Greens_9

import trend_analysis as ta
from trend_analysis import StrainSalesDF # converts single strain data to df
from trend_analysis import StrainTrendsDF # transforms single strain data
from trend_analysis import StrainStatsDF # compiles stats for multiple strains
from trend_analysis import CompTrendsDF # compares strains by ts data
from trend_analysis import RankStrains # returns ranked results


"""Graphic design adapted from: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""

def rescale_RGB(RGB_palette):
    """Input list of RGB tuples; return list of tuples rescaled (0, 1)"""
    scaled_palette = []
    for i in range(len(RGB_palette)):
        r, g, b = RGB_palette[i]
        scaled_palette.append((r / 255., g / 255., b / 255.))
    return scaled_palette


def range_vals(df):
    """Input CompTrendsDF object; return range, min and max values (tuple) among
    all products
    """
    stats_df = df.describe()
    minimum = stats_df.loc['min',].min()
    maximum = stats_df.loc['max',].max()
    return (maximum - minimum, minimum, maximum)


def select_step(val_range, low, high, max_N_ticks=10):
    # Set y_axis range on which to calculate step
    digits = lambda x: len(str(int(x)))
    range_digits = digits(val_range)
    low_digits = digits(low)
    high_digits = digits(high)
    if low >= 0:
        if range_digits == high_digits:
            yaxis_range = high
        if range_digits < high_digits:
            yaxis_range = high - low
    if low < 0:
        yaxis_range = high - low

    steps = [1, 10, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]
    for step in steps:
        if yaxis_range / step <= max_N_ticks:
            return step


def round_to_step(val_range, low, high, max_N_ticks=10):
    """Round max or abs(min) ytick value up to next step"""
    step = select_step(val_range, low, high, max_N_ticks)
    multiple = (int(high) / step) + 1
    return step * multiple


def round_down_to_step(val_range, low, high, max_N_ticks=10):
    """Round positive low ytick value down to next step"""
    step = select_step(val_range, low, high, max_N_ticks)
    multiple = (int(low) / step)
    return step * multiple


def round_up_to_pwr(high_val, range_val):
    """Round up to nearest power of 10 of range_val; e.g. 9340 >> 10000"""
    power = lambda x: len(str(int(x))) - 1 # get power of 10
    r_pwr = power(range_val)

    return int(np.ceil(high_val / 10**(r_pwr))*10**(r_pwr))


def round_down_to_pwr(low_val, range_val):
    """Round down to nearest power of 10 of range_val; e.g. 9340 >> 9000"""
    power = lambda x: len(str(int(x))) - 1 # get power of 10
    r_pwr = digits(range_val)

    return int(np.floor(low_val / 10**(r_pwr))*10**(r_pwr))


def ylims(val_range, low, high, max_N_ticks=10):
    """Return y_limits tuple (y_low, y_high) based on df value range"""
    digits = lambda x: len(str(int(x)))
    range_digits = digits(val_range)
    high_digits = digits(high)
    low_digits = digits(low)

    if low >= 0:
        if range_digits == high_digits:
            y_low = 0
            y_high = round_to_step(val_range, low, high, max_N_ticks)

        elif range_digits < high_digits:
            y_low = round_down_to_step(val_range, low, high, max_N_ticks)
            y_high = round_to_step(val_range, low, high, max_N_ticks)

    if low < 0:
        y_low = -1 * round_to_step(abs(low), val_range, max_N_ticks)
        y_high = round_to_step(val_range, low, high, max_N_ticks)

    return y_low, y_high


def space_yticks(y_low, y_high, step, trunc_yticks=False):
    "Return list of y-coordinates for yticks"
    if y_low >= 0:
        if trunc_yticks:
            return range(y_low, y_high + step, step)
        else:
            return range(0, y_high + step, step)
    else:
        yticks = range(0, y_low - step, -step) # start w/ negative ticks
        yticks.reverse()
        yticks.remove(0)
        pos_ticks = range(0, y_high + step, step)
        yticks.extend(pos_ticks)
        return yticks


def y_to_str(y): # format y-labels for graphs of dollar values
    if y >= 0:
        return '$' + str(y)
    else:
        return '-$' + str(abs(y))


def parse_title(str):
    A, B = str.split(' over ')[0], str.split(' over ')[1]
    if 'shift' in str.lower():
        A = 'Change in ' + str.split(' over ')[0]
    return (A, B)


def subtitle_y(fig_height):
    if fig_height >=14: return 0.825
    if fig_height >= 10: return 0.82
    elif fig_height <= 5: return 0.79
    elif fig_height <= 7: return 0.80
    else: return 0.81


def write_file_name(df):
    A = '{}wk_MA'.format(df.name.split('-')[0])
    B = '_{}'.format(df.name.split(' ')[5])
    path = '../img/{}.{}'.format(A+B, file_format)
    return path


def PlotCompTrends(df, fig_height=12, palette=Greens_9, reverse_palette=True,
                   max_yticks=10, trunc_yticks=False, legend=False,
                   write_path=None):
    """Plot time series in CompTrendsDF object
    ARGUMENTS:
     -- df: CompTrendsDF object (pandas DataFrame)
     -- fig_height: (int, default=12) recommended values 7 <= x <= 14
     -- palette: palettable object (default=Greens_9; possible
         values: Tableau_20, Tableau_10, TableauLight_10,
         TableauMedium_10, PurpleGray_6, PurpleGray_12, ColorBlind_10, Greens_5,
         Greens_9, Greys_5, Greys_9, Purples_5, Purples_9
     -- reverse_palette: (bool, default=True)
     -- max_yticks: (int, default=10)
     -- trunc_yticks: (bool, default=False) If True, remove yticks between zero
          and step below lowest data value and set that step as x-axis
     -- legend: (book, default=False) default places product labels at end of lines
     -- write_path: (str, default=None) write graph to jpeg or png at path provided
    """
    plt.figure(figsize=(12,fig_height))
    ax = plt.axes([.1,.1,.8,.65])
    colors = rescale_RGB(palette.colors)
    if reverse_palette:
        colors.reverse()
    # remove plot frame lines
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # set y_axis limits and format tick labels and grid lines
    if 'scale' in df.name.lower(): # i.e., if data rescaled / normed
        plt.ylim(-2.075, 2)
        tick_arr = [-1, 0, 1]
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on')
        plt.yticks(tick_arr, [y for y in tick_arr], fontsize=16)
        ax.yaxis.grid(True, ls='--', lw=0.75, color='black', alpha=0.5)

    else:
        val_range, data_min, data_max = range_vals(df)
        y_low, y_high = ylims(val_range, data_min, data_max, max_N_ticks=max_yticks) # custom function
        step = select_step(val_range, data_min, data_max, max_N_ticks=max_yticks)
        bottom_buffer = val_range * 0.05
        plt.ylim(y_low - bottom_buffer, y_high)

        # space and format y_ticks
        tick_arr = space_yticks(y_low, y_high, step, trunc_yticks)
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on')
        if 'unit' in df.name.lower():
            plt.yticks(tick_arr, [y for y in tick_arr], fontsize=14)
        else:
            plt.yticks(tick_arr, [y_to_str(y) for y in tick_arr], fontsize=14)
        ax.yaxis.grid(True, ls='--', lw=0.75, color='black', alpha=0.5)

    # set x_axis limits; add one day to upper limit for last minor tick mark
    idx = df.index
    ax.set_xlim(idx[0], idx[-1] + pd.DateOffset(1))

    # Format x_major ticks with month name at each beginning of month
    plt.tick_params(axis='x', which='major', bottom='off', top='off',
                    left='off', right='off', labelbottom='on', pad=10,
                    labelsize=14, rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # place a minor tick on x-axis at each week
    plt.tick_params(axis='x', which='minor', direction='out', length=10,
                    width=1.0, labelbottom='off', left='off', right='off')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(7)) # 7 days

    # plot data
    strains = list(df.columns)
    for i, strain in enumerate(strains):
        ci = i % len(colors) # revolving index for colors
        plt.plot(df[strain], lw=2.5, color=colors[ci])
        y_pos = df[strain].values[-1]
        if not legend: # strain labels at line ends
            plt.text(idx[-1] + pd.DateOffset(1), y_pos, strain, fontsize=16,
                    color=colors[ci], va='center')

    # format optional legend
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
        num_products = len(list(df.columns))
        legend_cols = num_products if num_products <= 5 else num_products / 2
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
          frameon=True, ncol=legend_cols, fontsize=12)

    # Title plot
    title_x = 0.50 if legend else 0.54
    if 'over' in df.name: # conditional on smoothed data
        sup, sub = parse_title(df.name)
        plt.figtext(title_x, .85, sup, fontsize=20, ha='center')
        plt.figtext(title_x, subtitle_y(fig_height), sub, fontsize=16, ha='center')

    else:
        plt.title(title_plot(df.name), x=title_x, fontsize=20)
        rcParams['axes.titlepad'] = 50

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25)

    plt.show()

"""
~~~~~~~~~~~~~~~~~~~~~~~
Graph Ranked Strains
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def title_subtitle_footnote(results_df):
    """Take in RankStrains.results object; return title, subtitle and
    footnote for graphs"""
    name_str = results_df.name
    stat_str = results_df.columns[-1]
    title, subtitle, footnote = None, None, None

    # fragments
    sales_bool = 'Sales' if 'sales' in name_str.lower() else 'Units Sold'
    MA = stat_str.split('wk')[0]
    period_str = name_str.split(' over ')[-1]
    shifted_str = 'Data Shifted (t0 = 0)'
    normed_str = 'Computed on Data Rescaled (-1, 1) then Shifted (T0 = 0)'

    if 'cumulative' in stat_str.lower():
        title = 'Cumulative Total ' + sales_bool + ' over ' + period_str

    if 'ma log' in stat_str.lower():
        title = 'Log-Areas under {}Wk-MA Trends in Daily {}'.format(
            MA, sales_bool)
        subtitle = period_str

    if 'shifted log' in stat_str.lower():
        title = 'Log-Areas under {}Wk-MA Gain/Loss in Daily {}'.format(
            MA, sales_bool)
        subtitle = period_str
        footnote = 'NOTE: Computed on Data Shifted to t0=0.'

    if 'gain' in stat_str.lower():
        title = 'Average Rates of Weekly Gain/Loss* in {}Wk-MA Trends in Daily {}'.format(
            MA, sales_bool)
        subtitle = period_str
        footnote = ('*NOTE: Rates computed by redistributing trend AUCs\
 under straight lines then taking the slopes.')

    if 'normd auc' in stat_str.lower():
        title = 'Areas under Normalized* {}Wk-MA Trends in Daily {}'\
            .format(MA, sales_bool)
        subtitle = period_str
        footnote = '*NOTE: Computed on Sales Data Rescaled (-1, 1) then Shifted (t0=0).'

    if 'normd slope' in stat_str.lower():
        title = 'Normalized Slopes* of {}Wk-MA Trends in Daily {}'\
            .format(MA, sales_bool)
        subtitle = period_str
        footnote = """*NOTE: Computed on rescaled (-1, 1) and shifted (t0=0) data by redistributing
trend AUCs under straight lines then taking the slopes."""

    return title, subtitle, footnote

def PlotRankedStrains(results_df, fig_height=7):
    """
    Input: RankStrains.results object (pandas DataFrame)
    Output: Horizontal bar graph showing strains ranked by statistic
    """
    # prepare data series
    to_plot = pd.Series(results_df.iloc[:,-1].values,
                        index=results_df['strain_name'])
    to_plot = to_plot[::-1]
    y_pos = range(len(to_plot))
    x = to_plot.values
    y_labels = to_plot.index

    # format figure
    plt.figure(figsize=(7, fig_height))
    ax = plt.axes([.1,.1,.8,.65])
    ax.set_frame_on(False)

    # plot data
    bars = ax.barh(y_pos, width=x, height=.8, color='green', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    plt.tick_params(axis='y', which='both', bottom='off', top='off',
                    left='off', right='off', labelleft='off', labelright='off',
                    labelsize=18, labelcolor='green')
    plt.tick_params(axis='x', which='both', bottom='off', top='off',
                    left='off', right='off', labelbottom='off',
                    labelsize=18, labelcolor='green')

    # color bars with negative values gray
    for i, bar in enumerate(bars):
        if x[i] < 0:
            bar.set_color('gray')
            bar.set_alpha(0.7)

    # label bars
    xmin, xmax = ax.get_xlim()
    buffer = max(abs(xmin), abs(xmax)) * 0.05
    for i, label in enumerate(y_labels):
        if x[i] >= 0:
            plt.text(-buffer, y_pos[i], label, color='green',
                     fontsize=14, ha='right', va='center')
        else:
            plt.text(buffer, y_pos[i], label, color='gray',
                     fontsize=14, ha='left', va='center')

    # title, subtitle and footnote
    tt, sb, ft = title_subtitle_footnote(results_df)
    plt.figtext(0.5, 0.85, tt, ha='center', fontsize=16)
    plt.figtext(0.5, 0.80, sb, ha='center', fontsize=14)
    plt.figtext(0.5, 0.0, ft, ha='center', fontsize=12)


    plt.show()
