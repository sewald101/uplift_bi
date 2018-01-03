import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mdates as mdates

import trend_analysis as ta
from trend_analysis import StrainSalesDF # converts single strain data to df
from trend_analysis import StrainTrendsDF # transforms single strain data
from trend_analysis import StrainStatsDF # compiles stats for multiple strains
from trend_analysis import CompTrendsDF # compares strains by ts data
from trend_analysis import RankStrains # returns ranked results


"""Code adapted from: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
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

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
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
    digits = lambda x: len(str(int(x))) # get power of 10
    r_dig = digits(range_val)

    return int(np.ceil(high_val / 10**(r_dig-1))*10**(r_dig-1))

def round_down(low_val, range_val):
    """Round down to nearest power of 10 of range_val; e.g. 9340 >> 9000"""
    digits = lambda x: len(str(int(x))) # get power of 10
    r_dig = digits(range_val)

    return int(np.floor(low_val / 10**(r_dig-1))*10**(r_dig-1))

def ylims(df):
    """Set y_limits based on value range"""
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
