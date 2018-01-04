import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from palettable.tableau import (Tableau_20, Tableau_10, TableauLight_10,
        TableauMedium_10,PurpleGray_6, PurpleGray_12, ColorBlind_10, Gray_5)


import trend_analysis as ta
from trend_analysis import StrainSalesDF # converts single strain data to df
from trend_analysis import StrainTrendsDF # transforms single strain data
from trend_analysis import StrainStatsDF # compiles stats for multiple strains
from trend_analysis import CompTrendsDF # compares strains by ts data
from trend_analysis import RankStrains # returns ranked results


"""Graphic design adapted from: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""

def rescale_RGB(RGB_palate):
    """Input list of RGB tuples; return list of tuples rescaled (0, 1)"""
    scaled_palate = []
    for i in range(len(RGB_palate)):
        r, g, b = RGB_palate[i]
        scaled_palate.append((r / 255., g / 255., b / 255.))
    return scaled_palate


def range_vals(df):
    """Input CompTrendsDF object; return range, min and max values (tuple) among
    all products
    """
    stats_df = df.describe()
    minimum = stats_df.loc['min',].min()
    maximum = stats_df.loc['max',].max()
    return (maximum - minimum, minimum, maximum)


def select_step(data_range, max_N_ticks=10):
    steps = [1, 10, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]
    for step in steps:
        if data_range / step <= max_N_ticks:
            return step


def round_to_step(data_max, data_range, max_N_ticks=10):
    """Round max or abs(min) ytick value up to next step"""
    step = select_step(data_range, max_N_ticks)
    multiple = (int(data_max) / step) + 1
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
            y_high = round_to_step(high, val_range, max_N_ticks)

        elif range_digits < high_digits:
            y_low = -1 * round_to_step(abs(low), val_range, max_N_ticks)
            y_high = round_to_step(high, val_range, max_N_ticks)

    if low < 0:
        y_low = -1 * round_to_step(abs(low), val_range, max_N_ticks)
        y_high = round_to_step(high, val_range, max_N_ticks)

    return y_low, y_high


def space_yticks(y_low, y_high, step):
    if y_low >= 0:
        return range(0, y_high + step, step)
    else:
        neg_ticks = range(0, y_low - step, -step)
        neg_ticks.reverse()
        neg_ticks.remove(0)
        pos_ticks = range(0, y_high + step, step)
        neg_ticks.extend(pos_ticks)
        return neg_ticks


def y_to_str(y):
    if y >= 0:
        return '$' + str(y)
    else:
        return '-$' + str(abs(y))


def PlotCompTrends(df, fig_height=12, palate=Tableau_20, max_yticks=10,
                   write_to_file=False, file_format='jpeg'):
    """Plot time series in CompTrendsDF object
    ARGUMENTS:
     -- df: CompTrendsDF object (pandas DataFrame)
     -- fig_height: (int, default=14)
     -- palate: (list) colors in tuples of RGB values (default=tableau20)
     -- write_to_file: if True, writes plot to image file (default=False)
     -- file_format: (str) image file extension (default='jpeg')
    """
    plt.figure(figsize=(12,fig_height))
    ax = plt.subplot(111)
    colors = rescale_RGB(palate.colors)
    # remove plot frame lines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # set y_axis limits
    if ('norm' or 'scale') in df.name.lower():
        pass

    else:
        val_range, data_min, data_max = range_vals(df)
        y_low, y_high = ylims(val_range, data_min, data_max, max_N_ticks=max_yticks) # custom function
        step = select_step(val_range, max_N_ticks=max_yticks)
        plt.ylim(y_low, y_high)

        # space and format y_ticks
        tick_arr = space_yticks(y_low, y_high, step)
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on')
        plt.yticks(tick_arr, [y_to_str(y) for y in tick_arr], fontsize=14)
        ax.yaxis.grid(True, ls='--')

    # set x_axis limits; add one day to upper limit (for tick mark)
    idx = df.index
    ax.set_xlim(idx[0], idx[-1] + pd.DateOffset(1))

    # Format x_major ticks with month name at each beginning of month
    plt.tick_params(axis='x', which='major', bottom='off', top='off',
                    left='off', right='off', labelbottom='on', pad=10,
                    labelsize=14, rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # place a minor tick on x-axis at each week (7 days)
    plt.tick_params(axis='x', which='minor', direction='out', length=10,
                    labelbottom='off', left='off', right='off')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(7))

    # plot data
    strains = list(df.columns)
    for i, strain in enumerate(strains):
        ci = i % len(colors) # revolving index for colors
        plt.plot(df[strain], lw=2.5, color=colors[ci])
        y_pos = df[strain].values[-1]
        plt.text(idx[-1] + pd.DateOffset(1), y_pos, strain, fontsize=14,
                color=colors[ci], va='center')

    plt.title(df.name, fontsize=18)
    rcParams['axes.titlepad'] = 50
    plt.show()

    if write_to_file:
        A = '{}wk_MA'.format(MA_5wk.name.split('-')[0])
        B = '_{}'.format(MA_5wk.name.split(' ')[4])
        path = '../img/{}.{}'.format(A+B, file_format)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.25)