import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

import trend_analysis as ta
from trend_analysis import StrainSalesDF # converts single strain data to df
from trend_analysis import StrainTrendsDF # transforms single strain data
from trend_analysis import StrainStatsDF # compiles stats for multiple strains
from trend_analysis import CompTrendsDF # compares strains by ts data
from trend_analysis import RankStrains # returns ranked results


"""Graphic design adapted from: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Tableau Color Blind 10
tableauCB10 = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
               (95, 158, 209), (200, 82, 0), (137, 137, 137), (162, 200, 236),
               (255, 188, 121), (207, 207, 207)]

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


def round_up(high_val, range_val):
    """Round up to nearest power of 10 of range_val; e.g. 9340 >> 10000"""
    power = lambda x: len(str(int(x))) - 1 # get power of 10
    r_pwr = power(range_val)

    return int(np.ceil(high_val / 10**(r_pwr))*10**(r_pwr))


def round_down(low_val, range_val):
    """Round down to nearest power of 10 of range_val; e.g. 9340 >> 9000"""
    power = lambda x: len(str(int(x))) - 1 # get power of 10
    r_pwr = digits(range_val)

    return int(np.floor(low_val / 10**(r_pwr))*10**(r_pwr))


def ylims(df):
    """Return y_limits tuple (y_low, y_high) based on df value range"""
    val_range, low, high = range_vals(df)
    digits = lambda x: len(str(int(x)))
    range_digits = digits(val_range)
    high_digits = digits(high)
    low_digits = digits(low)

    if low >= 0:
        if range_digits == high_digits:
            y_low = 0
            y_high = round_up(high, val_range)

        elif range_digits < high_digits:
            y_low = round_down(low, val_range)
            y_high = round_up(high, val_range)

    if low < 0:
        y_low = -1 * round_up(-1*low, val_range)
        y_high = round_up(high, val_range)

    return y_low, y_high


def select_step(low, high, max_N_ticks=10):
    steps = [1, 10, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]
    tick_range = high - low
    for step in steps:
        if tick_range / step <= max_N_ticks:
            return step


def space_yticks(low, high, max_N_ticks=10):
    step = select_step(low, high, max_N_ticks)
    if low >= 0:
        return range(0, high + step, step)
    else:
        neg_ticks = range(0, low - step, -step)
        neg_ticks.reverse()
        neg_ticks.remove(0)
        pos_ticks = range(0, high + step, step)
        neg_ticks.extend(pos_ticks)
        return neg_ticks


def PlotCompTrends(df, normed=False, palate=tableau20, max_yticks=10,
                   write_to_file=False, file_format='jpeg'):
    """Plot time series in CompTrendsDF object
    ARGUMENTS:
     -- df: CompTrendsDF object (pandas DataFrame)
     -- normed: If True, df contains normed data; default, data in dollars or units
     -- palate: (list) colors in tuples of RGB values (default=tableau20)
     -- write_to_file: if True, writes plot to image file (default=False)
     -- file_format: (str) image file extension (default='jpeg')
    """
    # plt.figure(figsize=(12,14))
    plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    colors = rescale_RGB(palate)
    # remove plot frame lines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_patch_line(ls='--')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # set y_axis limits
    if not normed:
        ylimits = ylims(df) # custom function
        plt.ylim(ylimits[0], ylimits[1])

        # space and format y_ticks
        tick_arr = space_yticks(ylimits[0], ylimits[1], max_yticks)
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on')
        plt.yticks(tick_arr, ["$" + str(x) for x in tick_arr], fontsize=14)
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
        plt.plot(df[strain], lw=2.5, color=colors[i])
        y_pos = df[strain].values[-1]
        plt.text(idx[-1] + pd.DateOffset(1), y_pos, strain, fontsize=14,
                color=colors[i], va='center')
    plt.title(df.name, fontsize=18)
    rcParams['axes.titlepad'] = 50
    plt.show()

    if write_to_file:
        A = '{}wk_MA'.format(MA_5wk.name.split('-')[0])
        B = '_{}'.format(MA_5wk.name.split(' ')[4])
        path = '../img/{}.{}'.format(A+B, file_format)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.25)
