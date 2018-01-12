import datetime
from math import trunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from palettable.tableau import (Tableau_20, Tableau_10, TableauLight_10,
        TableauMedium_10,PurpleGray_6, PurpleGray_12, ColorBlind_10, Gray_5)
from palettable.colorbrewer.sequential import (Greens_5, Greens_9,
    Greys_5, Greys_9, Purples_5, Purples_9)

import trend_analysis as ta
from trend_analysis import ImportSalesData # converts single product data to df
from trend_analysis import ProductTrendsDF # transforms single product data
from trend_analysis import ProductStatsDF # compiles stats for multiple products
from trend_analysis import CompTrendsDF # compares products by ts data
from trend_analysis import RankProducts # returns ranked results


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


def title_subtitle_footnote(ranked_df):
    """Take in RankProducts.results object; return title, subtitle and
    footnote for graphs"""
    name_str = ranked_df.name
    stat_str = ranked_df.columns[2]
    title, subtitle, footnote = None, None, None

    # string elements
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
        title = 'Weekly Gain/Loss* in {}Wk-MA Trends in Daily {}'.format(
            MA, sales_bool)
        subtitle = period_str
        footnote = ('*NOTE: Computed by redistributing trend AUCs\
 under straight lines then taking the slopes.')

    if 'normd auc' in stat_str.lower():
        title = 'Areas under Normalized* {}Wk-MA Trends in Daily {}'\
            .format(MA, sales_bool)
        subtitle = period_str
        footnote = '*NOTE: Computed on sales data rescaled (-50, 50) then shifted (t0=0).'

    if 'normd growth' in stat_str.lower():
        title = 'Normalized Growth Rates* of {}Wk-MA Trends in Daily {}'\
            .format(MA, sales_bool)
        subtitle = period_str
        footnote = """*NOTE: Rates of daily growth normalized for sales volumes
Data rescaled to (-50, 50) then shifted (t0=0)."""

    return title, subtitle, footnote


def currency_bool(stat_str):
    """Return True if values to be formatted in currency"""
    if 'cumulative' in stat_str.lower():
        return True
    if 'ma log' in stat_str.lower():
        return False
    if 'shifted log' in stat_str.lower():
        return False
    if 'gain' in stat_str.lower():
        return True
    if 'normd auc' in stat_str.lower():
        return False
    if 'normd growth' in stat_str.lower():
        return True


def format_currency(x, dollars=False, millions=False):
    if x >= 0:
        if millions:
            return '${:.2f}M'.format(x * 1e-6)
        if dollars:
            return '${:,}'.format(int(round(x, 0)))
        else:
            return '${:.2f}'.format(x)
    else:
        if millions:
            return '-${:.2f}M'.format(abs(x * 1e-6))
        if dollars:
            return '-${:,}'.format(int(round(abs(x), 0)))
        else:
            return '-${:.2f}'.format(abs(x))


def format_units(x, round_to_int=False, millions=False, decimals=2):
    if millions:
        return '{:.1f}M'.format(x * 1e-6)
    if round_to_int:
        return '{:,}'.format(int(round(x, 0)))
    else:
        return '{:.decimalsf}'.format(x)


def data_pos(data, xmin, xmax, buffer=5, in_bar=False):
    """Return list of positions for data labels in bar graph"""
    buff = max(abs(xmin), abs(xmax)) * (buffer / 100.)
    data_pos = []
    if in_bar:
        for x in data:
            if x >= 0:
                data_pos.append(x - buff)
            else:
                data_pos.append(x + buff)
    else:
        for x in data:
            if x >= 0:
                data_pos.append(x + buff)
            else:
                data_pos.append(x - buff)

    return data_pos


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

"""
~~~~~~~~~~~~~~~~~~~~~~~
Line Plot of Products over Uniform Trend Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

"""

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
        plt.ylim(-110, 110)
        tick_arr = [-100, 0, 100]
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on')
        if 'unit' in df.name.lower():
            plt.yticks(tick_arr, [y for y in tick_arr], fontsize=14)
        else:
            plt.yticks(tick_arr, [y_to_str(y) for y in tick_arr], fontsize=14)
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
    products = list(df.columns)
    for i, product in enumerate(products):
        ci = i % len(colors) # revolving index for colors
        plt.plot(df[product], lw=2.5, color=colors[ci])
        y_pos = df[product].values[-1]
        if not legend: # product labels at line ends
            plt.text(idx[-1] + pd.DateOffset(1), y_pos, product, fontsize=16,
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
Horizontal Bar Chart of Products Ranked by Statistic(s)
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def PlotRankedProducts(ranked_df, fig_height=7, round_to_int=False, millions=False,
        in_bar=False, data_buff=5, label_buff=40, write_path=None):
    """
    Input: RankProducts.ranked_df object (pandas DataFrame)
    Output: Horizontal bar graph showing products ranked by statistic
    """
    # prepare data series
    to_plot = pd.Series(ranked_df.iloc[:,2].values,
                        index=ranked_df['product_name'])
    to_plot = to_plot[::-1]
    y_pos = range(len(to_plot))
    x = to_plot.values
    y_labels = to_plot.index
    stat_str = ranked_df.columns[2]
    greys = rescale_RGB(Greys_9.colors)

    # format figure
    plt.figure(figsize=(7, fig_height), facecolor=greys[1])
#     plt.style.use('bmh')
    ax = plt.axes([.3,.1,.7,.65], facecolor=greys[1])
    ax2 = plt.axes([.0,.1,.3,.65], sharey=ax)
    ax.set_frame_on(True)
    ax2.set_frame_on(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # title, subtitle and footnote
    tt, sb, ft = title_subtitle_footnote(ranked_df)
    plt.figtext(-0.0, 0.85, tt, ha='left', fontsize=16, fontweight='bold')
    if sb: plt.figtext(-0.0, 0.80, sb, ha='left', fontsize=14)
    if ft: plt.figtext(-0.0, 0.0, ft, ha='left', fontsize=10)


    # ytick formatting
    ax.set_yticks(y_pos)
    ax2.set_yticklabels(y_labels, ha='right', va='center',
                        fontweight='ultralight', fontsize=12, color='black')
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                    left='off', right='off', labelleft='off',
                    labelright='off', labelbottom='off')
    ax2.tick_params(axis='both', which='both', bottom='off', top='off',
                left='off', right='off', labelleft='off',
                labelright='on', labelbottom='off', pad=-1*label_buff)

    #  labelsize=12, labelcolor=greys[7],

    # plot data
    bars = ax.barh(y_pos, width=x, height=.8, color='green', alpha=0.7)
    # color bars with negative values gray
    for i, bar in enumerate(bars):
        if x[i] < 0:
            bar.set_color(greys[5])
            bar.set_alpha(0.7)

            # xtick formating (must follow plot data)
        #     x_tix = ax.get_xticks()
        #     stat_str = ranked_df.columns[2]
        #     if currency_bool(stat_str):
        #         plt.xticks(x_tix, [format_currency(tick) for tick in x_tix])
        #     else:
        #         plt.xticks(x_tix, [tick for tick in x_tix])

        #     ax.tick_params(axis='x', which='both', bottom='off', top='off',
        #                     left='off', right='off', color='black',
        #                     labelbottom='off',
        #                     labelsize=12, labelcolor=greys[8])

    ax.axvline(0, color='black', linewidth=1)
    ax.axhline(y_pos[-1] + 0.5, ls='--', lw=0.5, color='black', alpha=0.2)
    for y in y_pos:
        ax.axhline(y - 0.5, ls='--', lw=0.5, color='black', alpha=0.2)

    # attach labels and values to bars
    xmin, xmax = ax.get_xlim()
    x_pos = data_pos(x, xmin, xmax, in_bar=in_bar, buffer=data_buff)
    data_color = 'black' if not in_bar else 'white'
    align = ('left', 'right') if not in_bar else ('right', 'left')
    # buffer = max(abs(xmin), abs(xmax)) * 0.05
    for i, label in enumerate(y_labels):
        if currency_bool(stat_str):
            if x[i] >= 0:
    #             plt.text(-buffer, y_pos[i], label, color='green',
    #                      fontsize=14, ha='right', va='center')
                ax.text(x_pos[i], y_pos[i],
                    format_currency(x[i],
                    dollars=round_to_int, millions=millions
                                   ),
                    color=data_color, fontsize=12, ha=align[0], va='center')

            else:
    #             plt.text(buffer, y_pos[i], label, color=greys[6],
    #                      fontsize=14, ha='left', va='center')
                ax.text(x_pos[i], y_pos[i],
                    format_currency(x[i],
                    dollars=round_to_int, millions=millions
                                   ),
                    color=data_color, fontsize=12, ha=align[1], va='center')
        else:
            if x[i] >= 0:
    #             plt.text(-buffer, y_pos[i], label, color='green',
    #                      fontsize=14, ha='right', va='center')
                ax.text(x_pos[i], y_pos[i],
                    format_units(x[i],
                    round_to_int=round_to_int, millions=millions
                                ),
                    color=data_color, fontsize=12, ha=align[0], va='center')

            else:
    #             plt.text(buffer, y_pos[i], label, color=greys[6],
    #                      fontsize=14, ha='left', va='center')
                ax.text(x_pos[i], y_pos[i],
                    format_units(x[i],
                    round_to_int=round_to_int, millions=millions
                                ),
                    color=data_color, fontsize=12, ha=align[1], va='center')

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25)

    plt.show()


def get_data(product_IDs, period_wks=10, end_date=None,
               rank_on_sales=True, MA=5,
               rank_by=['rate'], fixed_order=True):
    """Return a dataframe configured for custom plotting in HbarRanked function"""
    prod_stats = ProductStatsDF(product_IDs, period_wks, end_date,
                MA_params=[MA], compute_on_sales=rank_on_sales)

    if len(rank_by) < 2 or fixed_order: # just need the RankProducts.results object
        if len(rank_by) < 2:
            rank_1 = RankProducts(prod_stats)
            rank_1.main(rank_by[0])
            data = rank_1.results
            data.drop(['product_id'], axis=1, inplace=True)

        else:
            rank_1 = RankProducts(prod_stats)
            rank_1.main(rank_by[0])
            all_data = rank_1.ranked_df
            df_cols = all_data.columns
            cols = []
            for stat in rank_by:
                cols.append('product_name')
                cols.append(grab_column(df_cols, stat))

            data = all_data[cols]


    if len(rank_by) > 1 and not fixed_order:
            rank_1 = RankProducts(prod_stats)
            rank_1.main(rank_by[0])
            data = rank_1.results

            for i, stat in enumerate(rank_by[1:]):
                rank_next = RankProducts(prod_stats)
                rank_next.main(stat)
                next_ranked = rank_next.results
                data['Ranking By {}'.format(stat)] = next_ranked.iloc[:,0].values
                data[next_ranked.columns[-1]] = next_ranked.iloc[:,-1].values

            data.drop(['product_id'], axis=1, inplace=True)

    return data[::-1] # reverse row order for matplotlib bar graphing


def grab_column(df_cols, stat):
    """Get column title string from dataframe"""
    for c in df_cols:
        if stat in c:
            return c

def hide_spines(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def loc_format_value_label(x_arr, threshold=6, offset=0.02):
    """Return tuple (position, alignment, color) of value labels based on
    x values
    ARGUMENTS:
     -- x_arr: array-like, x-data
     -- threshold: (int or float, default=6) divisor of maximum abs x-value
          used to set threshold for whether value labels appear inside or
          outside of bars
     -- offset: (float, default=0.02) fraction of max, abs x-value by which
         to offset labels from bar caps

    """
    x_scale = max(abs(min(x_arr)), abs(max(x_arr)))
    gap = offset * x_scale
    pos, aligns, color = [], [], []
    for x in x_arr:
        if x > x_scale / float(threshold):
            pos.append(x - gap)
            aligns.append('right')
            color.append('white')
            continue
        if x > 0:
            pos.append(x + gap)
            aligns.append('left')
            color.append('0.3')
            continue
        if x > x_scale * -1 / float(threshold):
            pos.append(x - gap)
            aligns.append('right')
            color.append('0.3')
            continue
        else:
            pos.append(x + gap)
            aligns.append('left')
            color.append('white')

    return zip(pos, aligns, color)


def HbarRanked(product_IDs=None, period_wks=10, end_date=None,
           rank_on_sales=True, MA_param=5, rank_by=['rate'],
           fixed_order=True, fig_height=4, label_buffs=[4,5,5],
           x_buff=0.005, write_path=None):
    """
    ARGUMENTS:
     -- product_IDs: (list of ints) products for ranking
     -- period_wks: (int, default=10) sample period for time series in weeks
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date in dataset.
     -- rank_on_sales: (bool, default=True) ranks on sales data; if False,
          ranks on units sold data
     -- MA_param: (int) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average
     -- rank_by: (list of strings, default=['rate']) select statistic
          by which to rank products in the primary and optional secondary
          graphs in order of statistic. Values:
          * 'rate' = growth rate index for products with data
              normalized (rescaled -100, 100) for sales volumes
          * 'gain' = uniform weekly gain or loss over period
          * 'sales' = cumulative sales over period
     -- fixed_order: (bool, default=True) only rank products in the primary
          graph and maintain that rank-order in secondary graphs; if False,
          rank products in each graph
     -- write_path: (str, default=None) write graph to 'path/file'

    """

    # Construct dataframe for graph(s)
    df = get_data(product_IDs, period_wks, end_date=end_date,
               rank_on_sales=rank_on_sales, MA=MA_param,
               rank_by=rank_by, fixed_order=fixed_order)

    # Configure subplots
    fig, axs = plt.subplots(1, len(rank_by), figsize=(8*len(rank_by), fig_height), squeeze=False)
    plt.suptitle('Figure Title', x=0, y=0.98, fontsize=20, fontweight='bold', horizontalalignment='left')
    plt.figtext(0, 0.86, 'Figure Subtitle', fontsize=16, fontweight='normal', horizontalalignment='left')

    y_pos = range(len(df)) # positions for horizontal bars and product labels

    for i, ax in enumerate(axs.flatten()):

        # format plot title
        ax.set_title('Plot{}, {}'.format(i+1, rank_by[i].title()))

        # format y axis
        y_labels = df.iloc[:,(i*2)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, ha='right', va='center', fontsize=12)

        # get x data
        x = df.iloc[:,(i*2)+1]
        buff = x_buff * max(abs(min(x)), max(x))
        ax.set_xlim(min(x) - buff, max(x) + buff)
        xmin, xmax = ax.get_xlim()

        # Plot data
        bars = ax.barh(y_pos, width=x, height=.8, color='green', alpha=0.7)
        # color bars with negative values gray
        for j, bar in enumerate(bars):
            if x.values[j] < 0:
                bar.set_color('0.5')

    plt.tight_layout()
    plt.subplots_adjust(bottom=.1, top=.75)
    plt.show()
