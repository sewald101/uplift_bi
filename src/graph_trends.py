"""
PLOTTING FUNCTIONS

-- PlotCompTrends(df=None, products=None, period_wks=10, end_date=None,
                MA_param=5, shifted=False, normed=False, baseline='t_zero',
                compute_on_sales=True,
                fig_height=12, palette=Greens_9, fig_margins=(0.1, 0.75),
                reverse_palette=True, max_yticks=10, trunc_yticks=False,
                legend=False, txt=None, write_path=None):

-- HbarRanked(products=None, period_wks=10, end_date=None,
               rank_on_sales=True, MA_param=5, rank_by=['rate'], N_top=3,
               fixed_order=True, fig_height=4,
               x_buff=0.1, x_in_bar=6, manual_data_label_format=None,
               zero_gap=0.00, write_path=None)

-- PlotFilledTrends(products, period_wks=10, end_date=None,
                     compute_on_sales=True, MA_param=None, shifted=False,
                     normed=False, baseline='t_zero', max_yticks=10, fig_height=7,
                     trunc_yticks=False, write_path=None)

-- PlotBestSellers(df, labeler, N_top=None, footnote_pad=4.5, write_path=None)

"""


import datetime
from datetime import datetime
from math import trunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from palettable.tableau import (Tableau_20, Tableau_10, TableauLight_10,
        TableauMedium_10, PurpleGray_6, PurpleGray_12, ColorBlind_10, Gray_5)
from palettable.colorbrewer.sequential import (Greens_5, Greens_9,
    Greys_5, Greys_9, Purples_5, Purples_9)

import trend_analysis as ta
from trend_analysis import ImportSalesData # converts single product data to df
from trend_analysis import SalesTrendsDF # transforms single product data
from trend_analysis import SalesStatsDF # compiles stats for multiple products
from trend_analysis import CompTrendsDF # compares products by ts data
from trend_analysis import RankProductsPlaces # returns ranked results
from trend_analysis import HbarData

from id_dict import (strain_dict, names_formatted, locations_dict,
                     product_name_from_ID, locations_name_from_ID)


"""Graphic design adapted from: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""

"""
~~~~~~~~~~~~~~~~~~~~~~~
Line Plot of Raw Product Sales Data
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def PlotRawData(product=None, location=None, city=None, zipcode=None,
    plot_sales=True, upsample=True, fig_height=4):
    """
    Plot raw sales or units sold data over its complete time series.
    """

    # Get data
    stage_1 = ImportSalesData(product=product, location=location, city=city,
        zipcode=zipcode, upsample=upsample)
    stage_1.main()
    if plot_sales:
        ts = stage_1.sales
    else:
        ts = stage_1.units_sold

    # Plot data
    try:
        plt.figure(figsize=(16, fig_height))
        plt.plot(ts)
    except ValueError:
        return
    else:
        plt.grid(linewidth=0.5, color='0.7', linestyle='--')
        plt.title(ts.name)
        plt.show()



"""
~~~~~~~~~~~~~~~~~~~~~~~
Line Plot of Products over Uniform Trend Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def PlotCompTrends(df=None, period_wks=10, end_date=None, products=[None],
                locations=[None], cities=[None], zipcodes=[None],
                MA_param=5, shifted=False, normed=False, baseline='t_zero',
                compute_on_sales=True, NaN_filler=0.0,
                fig_height=12, palette=Greens_9, fig_margins=(0.1, 0.75),
                reverse_palette=True, max_yticks=10, trunc_yticks=False,
                legend=False, txt=None, write_path=None):
    """Plot time series in CompTrendsDF object
    DATA ARGUMENTS:
     -- df: CompTrendsDF object (pandas DataFrame)

    If df not provided:
     -- products: (list of ints or strings) product names and/or IDs for
          statistical comparison
     -- period_wks: (int) sampling period in weeks
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date.
     -- MA_param: (int) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average
     -- shifted: (bool, default=False) shift trend data to t0 = 0
     -- normed: (bool, default=False) rescale data to feature range (-1, 1)
          then shift data such that t0 = 0.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data to mean = 0
          * 'median' -- shift data to median = 0
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data


    GRAPHIC ARGUMENTS
     -- fig_height: (int, default=12) recommended values 7 <= x <= 14
     -- fig_margins: (tuple of floats, default=(0.2, 0.75)) variables to adjust
          figure footer and header via kwargs in plt.subplots_adjust,
          tuple: (bottom=, top=)
          See: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
     -- palette: palettable object (default=Greens_9; possible
         values: Tableau_20, Tableau_10, TableauLight_10,
         TableauMedium_10, PurpleGray_6, PurpleGray_12, ColorBlind_10, Greens_5,
         Greens_9, Greys_5, Greys_9, Purples_5, Purples_9
     -- reverse_palette: (bool, default=True)
     -- max_yticks: (int, default=10)
     -- trunc_yticks: (bool, default=False) If True, remove yticks between zero
          and step below lowest data value and set that step as x-axis
     -- legend: (book, default=False) default places product labels at end of lines
     -- txt: (list of tuples, default=None): manual specification of titles, subtitles
          and footnotes, each specified per the matplotlib.pyplot.text args and kwargs in tuple:
          (text(str), y-pos(float), fontsize(int), fontweight(str), color(str))
          See: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html
     -- write_path: (str, default=None) write graph to jpeg or png at path provided
    """

    if df is None:
        df = CompTrendsDF(period_wks=period_wks, end_date=end_date,
                          products=products, locations=locations,
                          cities=cities, zipcodes=zipcodes,
                          MA_param=MA_param, shifted=shifted, normed=normed,
                          baseline=baseline, compute_on_sales=compute_on_sales,
                          NaN_filler=NaN_filler
                          )

    fig, ax = plt.subplots(figsize=(12,fig_height))
    colors = rescale_RGB(palette.colors)
    if reverse_palette:
        colors.reverse()
    # remove plot frame lines
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('0.5')
    ax.spines['bottom'].set_lw(2.0)

    # set y_axis limits and format tick labels and grid lines
    if 'scale' in df.name.lower(): # i.e., if data rescaled / normed
        plt.ylim(-110, 110)
        tick_arr = [-100, 0, 100]
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on',
                        labelcolor='0.5')
        if 'unit' in df.name.lower():
            plt.yticks(tick_arr, [y for y in tick_arr], fontsize=14)
        else:
            plt.yticks(tick_arr, [y_to_str(y) for y in tick_arr], fontsize=14)
        ax.yaxis.grid(True, ls='--', lw=0.5, color='0.9')

    else:
        val_range, data_min, data_max = range_vals(df)
        y_low, y_high = ylims(val_range, data_min, data_max, max_N_ticks=max_yticks) # custom function
        step = select_step(val_range, data_min, data_max, max_N_ticks=max_yticks)
        bottom_buffer = val_range * 0.05
        plt.ylim(y_low - bottom_buffer, y_high)

        # space and format y_ticks
        tick_arr = space_yticks(y_low, y_high, step, trunc_yticks)
        plt.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='on',
                        labelcolor='0.3')
        if 'unit' in df.name.lower():
            plt.yticks(tick_arr, [y for y in tick_arr], fontsize=14)
        else:
            plt.yticks(tick_arr, [y_to_str(y) for y in tick_arr], fontsize=14)
        ax.yaxis.grid(True, ls='--', lw=0.5, color='0.8')

    # set x_axis limits; add one day to upper limit for last minor tick mark
    idx = df.index
    ax.set_xlim(idx[0], idx[-1] + pd.DateOffset(1))

    # Format x_major ticks with month name at each beginning of month
    plt.tick_params(axis='x', which='major', bottom='off', top='off',
                    left='off', right='off', labelbottom='on', pad=10,
                    labelsize=14, labelrotation=45, labelcolor='0.4')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # place a minor tick on x-axis at each week
    plt.tick_params(axis='x', which='minor', direction='out', length=10,
                    width=2.0, labelbottom='off', left='off', right='off',
                    color='0.5')
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
          frameon=False, ncol=legend_cols, fontsize=12)

    # Title, subtitle and notes
    title_x = 0.50 if legend else 0.54
    if txt: # manual title and note input
        for tup in txt:
            fig.text(x=0, y=tup[1], s=tup[0], fontsize=tup[2], fontweight=tup[3],
            color=tup[4], va='bottom', ha='left')

    else: # default automated titling
        if 'over' in df.name: # conditional on smoothed trend data
            sup, sub = parse_title(df.name)
            plt.figtext(title_x, .85, sup, fontsize=20, ha='center')
            plt.figtext(title_x, subtitle_y(fig_height), sub, fontsize=16, ha='center')

        else:
            plt.title(title_plot(df.name), x=title_x, fontsize=20)
            rcParams['axes.titlepad'] = 50

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25, dpi=300)

    plt.tight_layout()
    plt.subplots_adjust(bottom=fig_margins[0], top=fig_margins[1])

    plt.show()


"""
~~~~~~~~~~~~~~~~~~~~~~~
Horizontal Bar Chart of Products Ranked by Statistic(s)
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def HbarRanked(df=None, period_wks=None, end_date=None,
               products=[None], locations=[None],
               cities=[None], zipcodes=[None], MA_param=5,
               rank_on_sales=True, rank_by=['rate'], N_top=3,
               fixed_order=True, NaN_allowance=5, print_rejects=False,
               fig_height=10, fig_margins=(0.2, 0.8, None),
               x_buff=0.1, x_in_bar=6, manual_data_label_format=None,
               zero_gap=0.00, txt=None, write_path=None):
    """
    DATA ARGUMENTS:
     -- df: (None or Dataframe) HbarData object

    IF df=None, PROVIDE THE FOLLOWING DATA ARGUMENTS
     -- period_wks: (int, default=10) sample period for time series in weeks
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date in dataset.

    PROVIDE ONE OF THE BELOW OR A COMBINATION OF TWO ARGUMENTS with ONE OF THE
    TWO CONTAINING ONLY ONE VALUE IN ITS LIST
     -- products: (list of ints or strings) list of product names and/or IDs for
          filtering or statistical comparison
     -- locations: (list of ints or strings) list of retail store names and/or
          IDs for filtering or statistical comparison
     -- cities: (list of strings) list of cities for filtering or statistical
          comparison
     -- zipcodes: (list of 5-digit zipcodes as ints) list of zipcodes for filtering
          or statistical comparison

    ADDITIONAL DATA KWARGS
     -- rank_on_sales: (bool, default=True) ranks on sales data; if False,
          ranks on units sold data
     -- MA_param: (int or NoneType) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average; if None, computes
          on raw trend data.
     -- rank_by: (list of strings, default=['rate']) select statistic
          by which to rank products in the primary and optional secondary
          graphs in order of statistic. Values:
          * 'rate' = growth rate index for products with data
              normalized (rescaled -100, 100) for sales volumes
          * 'gain' = uniform weekly gain or loss over period
          * 'sales' = cumulative sales over period
     -- N_top: (int, default=3) highlight N-top results
     -- fixed_order: (bool, default=True) only rank products in the primary
          bar graph and maintain that rank-order in secondary graphs; if False,
          rank products in each bar graph.
     -- NaN_allowance: (int from 0 to 100, default=5) max allowable percentage of
          NaNs in product ts samples for statistical aggregation; products
          exceeding allowance are discarded from rankings.
     -- print_rejects: (bool, default=False) If True, print report of products
          rejected for excess null values in sample, with their corresponding
          percentage of nulls in sample.

   GRAPHICS ARGUMENTS
     -- fig_height: (int, default=4) y-dimension of plt.figure
     -- fig_margins: (tuple of floats, default=(0.2, 0.8, None)) variables to adjust figure margins
          via kwargs in plt.subplots_adjust, tuple: (bottom, top, wspace)
          See: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
     -- x_buff: (float, default=0.005) fraction of maximum absolute x-value
          by which to set left and right margins of plot around bars
     -- x_in_bar: (int or float, default=6) divisor of maximum abs x-value used to set
          threshold for whether each value label appears inside or outside of its bar.
          Lower number means label more likely to appear outside of bar.
     -- manual_data_label_format: (tuple, default=None) override default x_label
          formatting with the following ordered values in a tuple:
          * format as currency (bool)
          * format in millions (bool)
          * round_to_int (bool)
          * precision in decimals (int)
          example: (True, True, False, 3) or (1,1,0,3) formats -1234567 as -$1.234 million
     -- txt: (list of tuples, default=None): manual specification of titles, subtitles
          and footnotes, each specified per the matplotlib.pyplot.text args and kwargs in tuple:
          (text(str), y-pos(float), fontsize(int), fontweight(str), color(str))
          See: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html
     -- zero_gap: (float, default=0.00) fraction of max abs x_value as width of
          cosmetic whitespace between zero-line and bars; recommended value
          if used: 0.01
     -- write_path: (str, default=None) write graph to 'path/file'

    """

    # Construct dataframe for graph(s)
    if type(rank_by) == str: # In case user forgets to enclose arg in list
        rank_by = [rank_by]

    if df is None:
        df = HbarData(period_wks, end_date, products, locations, cities, zipcodes,
                   rank_on_sales=rank_on_sales, MA=MA_param,
                   rank_by=rank_by, fixed_order=fixed_order,
                   NaN_allowance=NaN_allowance, print_rejects=print_rejects)

    df_cols = df.columns

    # Configure subplots
    share_bool = 'all' if fixed_order else 'none'
    fig, axs = plt.subplots(1, len(rank_by), squeeze=False, sharey=share_bool,
                            figsize=(8*len(rank_by), fig_height),
                            )

    y_pos = range(len(df)) # positions for horizontal bars and product labels

    for i, ax in enumerate(axs.flatten()):

        hide_spines(ax)

        # format plot canvas(es)
        ax.axvline(0, color='0.2', linewidth=1)
        ax.tick_params(axis='y', which='both', bottom='off', top='off',
                left='off', right='off', labelleft='on',
                labelright='off', labelbottom='off')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                left='off', right='off', labelleft='off',
                labelright='off', labelbottom='off')

        # gridlines
        ax.axhline(y_pos[-1] + 0.5, ls='--', lw=0.5, color='0.8')
        for y in y_pos:
            ax.axhline(y - 0.5, ls='--', lw=0.5, color='0.8')

        # separator for multiple ranked graphs
        if not fixed_order and i != len(rank_by)-1:
            ax.spines['right'].set_visible('True')
            ax.spines['right'].set_color('0.8')
            ax.spines['right'].set_lw(0.5)
            ax.spines['right'].set_ls('-')

        # format y axis
        y_labels = df.iloc[:,(i*2)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, ha='right', va='center',
                           color='0.3', fontsize=12)

        # get x data and format data labels
        x = df.iloc[:,(i*2)+1]
        x_scale = max(abs(min(x)), abs(max(x))) # largest bar
        buff = x_buff * x_scale
        ax.set_xlim(min(x) - buff, max(x) + buff)
        xmin, xmax = ax.get_xlim()
        if manual_data_label_format:
            q = manual_data_label_format
            x_labels = manual_data_format(x, q[0], q[1], q[2], dec=q[3])
        else:
            x_labels = default_data_format(x, rank_on_sales, rank_by[i])

        # optional small gap (bar-zero-space, bzs) between bars and zero line
        gap = zero_gap * x_scale
        bzs = [gap if val > 0 else -1 * gap for val in x]

        # Plot data
        bars = ax.barh(y_pos, width=x-bzs, left=bzs, height=.8, color='0.5',
                       alpha=0.7)
        labels = ax.get_yticklabels()[::-1]

        # Format bars and labels for N-top results
        if not fixed_order or i==0:
            for j, bar in enumerate(bars[::-1]):
                if j < N_top:
                    bar.set_color('green')
                    labels[j]
                    labels[j].set_color('green')
                    labels[j].set_fontsize(16)
                    labels[j].set_fontweight('bold')

        # Add value labels to bars
        val_pos = loc_format_value_label(x, threshold=x_in_bar)
        for k, tup in enumerate(val_pos):
            ax.text(tup[0], y_pos[k], x_labels[k],
                   ha=tup[1], va='center', color=tup[2], fontsize=12)

        # format plot title and footnote
        if not txt:
            axtitle, footnote = axtitle_footnote(rank_by[i], rank_on_sales)
            if len(rank_by) > 1:
                ax.set_title(axtitle, loc='center', fontsize=16, fontweight='normal',
                            va='bottom', ha='center')
            else:
                title = u"Products Ranked by {}".format(axtitle)
                plt.suptitle(title, x=0, y=0.95, fontsize=20, fontweight='bold', ha='left',
                            va='top', color='k')

            if footnote:
                ax.annotate(footnote, xy=(0,-0.2), xycoords='axes fraction', ha='left',
                           fontsize=10, color='0.5')

    # Figure titles, subtitles and footnotes
    if txt: # manual input via txt kwarg
        for tup in txt:
            fig.text(x=0, y=tup[1], s=tup[0], fontsize=tup[2], fontweight=tup[3],
            color=tup[4], va='bottom', ha='left')

    else: # automated titles and notes
        dfname_parsed = df.name.split(' -- ')
        A = u'Product Performance over '
        B = dfname_parsed[0].split(' over ')[1]
        C = u"Computed on {}".format(dfname_parsed[1]) if MA_param else ''

        if len(rank_by) > 1:
            plt.suptitle(A + B + '\n', x=0.0, y=0.96, fontsize=20, fontweight='bold',
                     va='top', ha='left', color='k')
            fig.text(x=0.0, y=0.9, s=C, fontsize=16, fontweight='normal',
                     va='top', ha='left', color='0.4')

        else:
            fig.text(x=0.0, y=0.88, s=C, fontsize=16, fontweight='normal',
                     va='top', ha='left', color='0.4')

    plt.tight_layout()
    plt.subplots_adjust(bottom=fig_margins[0], top=fig_margins[1],
                        wspace=fig_margins[2])

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25,
                   dpi=300)
    plt.show()


def hide_spines(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def format_currency(x, dollars=False, millions=False, decimals=2):
    if x >= 0:
        if millions:
            return '${:.{prec}f} million'.format(x * 1e-6, prec=decimals)
        if dollars:
            return '${:,}'.format(int(round(x, 0)))
        else:
            return '${:.2f}'.format(x)
    else:
        if millions:
            return '-${:.{prec}f} million'.format(abs(x * 1e-6), prec=decimals)
        if dollars:
            return '-${:,}'.format(int(round(abs(x), 0)))
        else:
            return '-${:.2f}'.format(abs(x))


def format_units(x, round_to_int=False, millions=False, decimals=2):
    if millions:
        return '{:.{prec}f} million'.format(x * 1e-6, prec=decimals)
    if round_to_int:
        return '{:,}'.format(int(round(x, 0)))
    else:
        return '{:.{prec}f}'.format(x, prec=decimals)


def default_data_format(x_arr, curr_bool, rank_by):
    """Format data labels based on currency detection and max(abs(x))."""
    formatted = []
    max_x = max(max(x_arr), abs(min(x_arr)))
    if curr_bool: # if currency
        if max_x > 999999:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1: # full format for first data point
                    if rank_by == 'rate':
                        formatted.append(format_currency(x, millions=True) + '/wk')
                    elif rank_by == 'gain':
                        formatted.append(format_currency(x, millions=True) + '/wk')
                    else:
                        formatted.append(format_currency(x, millions=True) + '/wk')
                else:
                    formatted.append(format_currency(x, millions=True))
        elif max_x > 199:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1:
                    if rank_by == 'rate':
                        formatted.append(format_currency(x, dollars=True) + '/wk')
                    elif rank_by == 'gain':
                        formatted.append(format_currency(x, dollars=True) + '/wk')
                    else:
                        formatted.append(format_currency(x, dollars=True) + '/wk')
                else:
                    formatted.append(format_currency(x, dollars=True))
        else:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1:
                    if rank_by == 'rate':
                        formatted.append(format_currency(x, decimals=2) + '/wk')
                    elif rank_by == 'gain':
                        formatted.append(format_currency(x, decimals=2) + '/wk')
                    else:
                        formatted.append(format_currency(x, decimals=2) + '/wk')
                else:
                        formatted.append(format_currency(x, decimals=2))

    else: # if NOT currency...
        if max_x > 999999:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1: # full format for first data point
                    if rank_by == 'rate':
                        formatted.append(
                        format_units(x, millions=True) + ' units/wk'
                        )
                    elif rank_by == 'gain':
                        formatted.append(
                        format_units(x, millions=True) + ' units/wk'
                        )
                    else:
                        formatted.append(
                        format_units(x, millions=True) + ' units/wk'
                        )
                else:
                    formatted.append(format_units(x * 1e-6))

        if max_x > 50:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1:
                    if rank_by == 'rate':
                        formatted.append(
                        format_units(x, round_to_int=True) + ' units/wk'
                        )
                    elif rank_by == 'gain':
                        formatted.append(
                        format_units(x, round_to_int=True) + ' units/wk'
                        )
                    else:
                        formatted.append(
                        format_units(x, round_to_int=True) + ' units/wk'
                        )
                else:
                    formatted.append(format_units(x, round_to_int=True))

        else:
            for i, x in enumerate(x_arr):
                if i == len(x_arr)-1:
                    if rank_by == 'rate':
                        formatted.append(
                        format_units(x, decimals=2) + ' units/wk'
                        )
                    elif rank_by == 'gain':
                        formatted.append(
                        format_units(x, decimals=2) + ' units/wk'
                        )
                    else:
                        formatted.append(format_units(x) + ' units/wk')
                else:
                    formatted.append(format_units(x))

    return formatted


def manual_data_format(x_arr, curr_bool, millions_bool=None,
                        ints_bool=None, dec=3):
    """Format x_labels according to user specifications"""
    formatted = []
    if curr_bool:
        if millions_bool:
            for x in x_arr:
                formatted.append(format_currency(x, millions=True,
                                                    decimals=dec))
            return formatted
        elif ints_bool:
            for x in x_arr:
                formatted.append(format_currency(x, dollars=True))
            return formatted
        else:
            for x in x_arr:
                formatted.append(format_currency(x))
            return formatted
    else:
        if millions_bool:
            for x in x_arr:
                formatted.append(format_units(x, millions=True,
                                                decimals=dec))
            return formatted
        if ints_bool:
            for x in x_arr:
                formatted.append(format_units(x, round_to_int=True))
            return formatted
        else:
            for x in x_arr:
                formatted.append(format_units(x, decimals=dec))
            return formatted


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


def axtitle_footnote(rank_by, curr_bool):
    """Generate tuple (plot title, footnote) for rank-by statistic and currency boolean."""
    sym = '$' if curr_bool else ''
    if rank_by == 'sales':
        if curr_bool:
            title = u'Avg Weekly Sales\n'
            footnote = None
        else:
            title = u'Avg Weekly Units Sold\n'
            footnote = None
    if rank_by == 'rate':
        title = (u'Relative Growth Rate$^\u2020$\n')
        footnote = (
        u'\u2020 Daily sales data for each strain rescaled to (-{}50, {}50)'
        u' then shifted to t0 = {}0.00\n'
        u'   Rate then calculated from the slope of a straight '
        u'line containing area under trend curve.'.format(sym, sym, sym)
        )
    if rank_by == 'gain':
        title = u'Avg Weekly Gain/Loss$^*$\n'
        footnote = (
        u'* Slope of straight line containing '
        u'area under trend curve\n   Data shifted to t0 = {}0.00'.format(sym)
        )

    return title, footnote


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



"""
~~~~~~~~~~~~~~~~~~~~~~~
Separate, Filled Trend Plots for Products on Same Y_Scale
~~~~~~~~~~~~~~~~~~~~~~~~

"""

def PlotFilledTrends(df=None, products=None, period_wks=10, end_date=None,
                     compute_on_sales=True, MA_param=None, shifted=False,
                     normed=False, baseline='t_zero',fig_height=7,
                     fig_margins=(None, 0.85, 0.60), max_yticks=10, trunc_yticks=False,
                     fill_plot=True, txt=None, write_path=None):
    """Separately plot sales trends for products on identical y-scales and
    trend parameters.

    Default arguments plot on time series of raw sales data. Otherwise,
    assign value MA_param= for moving average. Optionally may assign True
    to either shifted= or normed= arguments (NOT BOTH).

    DATA ARGUMENTS:
     -- df: CompTrendsDF object (pandas DataFrame)

    If df not provided:
     -- products: (list of ints or strings) product names and/or IDs for
          statistical comparison
     -- period_wks: (int) sampling period in weeks
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date.
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
     -- MA_param: (int) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average
     -- shifted: (bool, default=False) shift trend data to t0 = 0
     -- normed: (bool, default=False) rescale data to feature range (-1, 1)
          then shift data such that t0 = 0.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data to mean = 0
          * 'median' -- shift data to median = 0

    GRAPHIC ARGUMENTS:
     -- fig_height: (int, default=7) factor for y-dimension of plt.figure
     -- fig_margins: (tuple of floats, default=(None, 0.85, 0.6)) variables to adjust figure margins
          via kwargs in plt.subplots_adjust, tuple: (bottom=, top=, hspace=)
          See: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
     -- max_yticks: (int, default=10)
     -- trunc_yticks: (bool, default=False) If True, remove yticks between zero
          and step below lowest data value and set that step as x-axis
     -- fill_plot: (bool, default=True) Fill areas between curve and baseline
     -- txt: (list of tuples, default=None): manual specification of titles, subtitles
          and footnotes, each specified per the matplotlib.pyplot.text args and kwargs in tuple:
          (text(str), y-pos(float), fontsize(int), fontweight(str), color(str))
          See: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html
     -- write_path: (str, default=None) write graph to 'path/file'
    """
    if df is None:
        df = CompTrendsDF(products, period_wks, end_date=end_date,
                        compute_on_sales=compute_on_sales, MA_param=MA_param,
                         shifted=shifted, normed=normed, baseline=baseline)

    products = df.columns
    fig, axs = plt.subplots(len(products), 1, squeeze=False, sharex='row',
                            figsize=(10, fig_height*len(products))
                           )

    # Title for figure
    if txt:
        for tup in txt:
            fig.text(x=0, y=tup[1], s=tup[0], fontsize=tup[2], fontweight=tup[3],
            color=tup[4], va='bottom', ha='left')
    else:
        if MA_param:
            title_parsed = df.name.split(', D')
            fig_t, fig_subt = title_parsed[0], title_parsed[-1]
            title = fig_t
        else:
            title = df.name
        plt.suptitle(title, x=0.5, y=0, fontsize=18,
                     fontweight='normal', va='bottom', ha='center')

    axes = axs.flatten()
    for i, ax in enumerate(axes):
        hide_spines(ax)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('0.6')
        series = df[products[i]]
        y_pos, y_neg = series.copy(), series.copy()
        ax.set_title('{}'.format(products[i]), loc='center', fontsize=16,
            fontweight='bold', ha='center', va='bottom', color='green')

        y_pos[y_pos < 0] = np.nan
        y_neg[y_neg >= 0] = np.nan
        ax.plot(y_pos, lw=2.5, color='green')
        if fill_plot:
            ax.fill_between(y_pos.index, y_pos, facecolor='green', alpha=0.5)
        ax.plot(y_neg, lw=2.5, color='0.3')
        if fill_plot:
            ax.fill_between(y_neg.index, y_neg, facecolor='0.3', alpha=0.5)
        ax.axhline(lw=1.5, color='0.6')


        # set x_axis limits; add one day to upper limit for last minor tick mark
        idx = df.index
        ax.set_xlim(idx[0], idx[-1] + pd.DateOffset(1))

        # Format x_major ticks with month name at each beginning of month
        ax.tick_params(axis='x', which='major', bottom='off', top='off',
                        left='off', right='off', labelbottom='on', pad=10,
                        labelsize=14, labelrotation=45, labelcolor='0.6')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # place a minor tick on x-axis at each week
        ax.tick_params(axis='x', which='minor', direction='out', length=10,
                        width=1.0, color='0.6',
                        labelbottom='off', left='off', right='off')
        ax.xaxis.set_minor_locator(plt.MultipleLocator(7)) # 7 days

        # set y_axis limits and format tick labels and grid lines
        if 'scale' in df.name.lower(): # i.e., if data rescaled / normed
            ax.set_ylim(-120, 120)
            tick_arr = [-100, 0, 100]
            ax.tick_params(axis='y', which='both', bottom='off', top='off',
                            left='off', right='off', labelleft='on',
                          labelsize=14, labelcolor='0.4')
            if 'unit' in df.name.lower():
                ax.set_yticks(tick_arr)
                ax.set_yticklabels([y for y in tick_arr])
            else:
                ax.set_yticks(tick_arr)
                ax.set_yticklabels([y_to_str(y) for y in tick_arr])
            ax.yaxis.grid(True, ls='--', lw=0.5, color='0.7')

        else:
            val_range, data_min, data_max = range_vals(df)
            y_low, y_high = ylims(val_range, data_min, data_max,
                    max_N_ticks=max_yticks) # custom function
            bottom_buffer = val_range * 0.05
            ax.set_ylim(y_low - bottom_buffer, y_high)

            # space and format y_ticks
            step = select_step(val_range, data_min, data_max,
                                max_N_ticks=max_yticks)
            tick_arr = space_yticks(y_low, y_high, step,
                                    trunc_yticks=trunc_yticks)
            ax.tick_params(axis='y', which='both', bottom='off', top='off',
                            left='off', right='off', labelleft='on',
                          labelsize=14, labelcolor='0.4')
            if 'unit' in df.name.lower():
                ax.set_yticks(tick_arr)
                ax.set_yticklabels([y for y in tick_arr])
                ax.set_ylabel('Units Sold', fontsize=14, labelpad=15)
            else:
                ax.set_yticks(tick_arr)
                ax.set_yticklabels([y_to_str(y) for y in tick_arr])
            ax.yaxis.grid(True, ls='--', lw=0.5, color='0.5')


    plt.tight_layout()
    ftr_dict = {1:0.30, 2:0.15, 3:0.10, 4:0.075, 5:0.06}
    footer = ftr_dict[len(products)] if not fig_margins[0] else fig_margins[0]
    fig.subplots_adjust(bottom=footer, top=fig_margins[1],
                        hspace=fig_margins[2])

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25,
                   dpi=300)

    plt.show()


"""
~~~~~~~~~~~~~~~~~~~~~~~
Plot BestSellerData
~~~~~~~~~~~~~~~~~~~~~~~~
"""

def PlotBestSellers(df, labeler, N_top=None, footnote_pad=4.5,
                    fig_margins=(0.3, 0.9), txt=None, write_path=None):
    """
    Plot BestSellerData object -- rankings over consecutive periods.

    ARGUMENTS:
    -- df: BestSellerData[1] object df_B dataframe
    -- labeler: BestSellerData[2] object, product label specifications
    -- N_top: (int, default=None) number of top-performing products
           to highlight
    -- footnote_pad: (float) factor to set padding for footnote beneath
       x-axis label
    -- fig_margins: (tuple of floats, default=(0.2, 0.75)) variables to adjust
          figure footer and header via kwargs in plt.subplots_adjust,
          tuple: (bottom=, top=)
          See: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    -- txt: (list of tuples, default=None): manual specification of titles, subtitles
          and footnotes, each specified per the matplotlib.pyplot.text args and kwargs in tuple:
          (text(str), y-pos(float), fontsize(int), fontweight(str), color(str))
          See: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html
    -- write_path: (str, default=None) write graph to 'path/file'

    """

    # Reverse sign of data to enable plotting with 1 at y_axis top
    df_revd = df * -1
    # Scale for setting labels and label offsets
    d_range = pd.Timedelta(df.index[-1] - df.index[0]).days

    # Format canvas, title and footnote
    fig, ax = plt.subplots(figsize=(10, 0.85 * len(df.columns)))
    hide_spines(ax)

    if txt: # manual input of title and footnote
        for tup in txt:
            fig.text(x=0, y=tup[1], s=tup[0], fontsize=tup[2], fontweight=tup[3],
            color=tup[4], va='bottom', ha='left')

    else: # default automated title and footnote
        title, footnote = df.name.split(' -- ')
        ax.set_title(title + '\n', loc='center', fontsize=20, va='bottom',
                     ha='center')
        # fn_ypos = -5 * len(df.columns)
        plt.annotate(footnote, (0.5, -footnote_pad/len(df.columns)),
                     xycoords='axes fraction', fontsize=14, va='bottom',
                     ha='center', color='0.5')


    # Format y_axis
    n_products = range(1, len(df.columns)+1)
    ytick_arr = np.array(n_products) * -1
    ax.set_ylim(top=-0.5, bottom=-1*len(df.columns)-0.5)
    ax.set_yticks(ytick_arr)
    ax.set_yticklabels(n_products, fontsize=16, color='0.5')
    ax.tick_params(axis='y', which='both', bottom='off', top='off',
                        left='off', right='off', labelleft='off')
    ax.set_ylabel('Rank', fontsize=16, labelpad=10)

    # Format x_axis
    ax.set_xticks(df.index)
    ax.tick_params(axis='x', which='both', bottom='off', top='off',
                   left='off', right='off', labelbottom='on',
                   pad=5, labelsize=14)

    x_labels = BestSeller_x_labels(df.index)
    ax.set_xticklabels(x_labels, rotation=90, color='0.5')
    ax.set_xlabel('Periods Ending', fontsize=16, labelpad=15)


    # Format colors and cosmetics, then plot data
    palate = rescale_RGB(ColorBlind_10.colors)
    colors_top = rescale_RGB(Greens_5.colors)
    colors_top.reverse()
    colors_bottom = ['0.1', '0.4', '0.7']
    label_offset = d_range / 20. if d_range > 14 else 1
    for i, col in enumerate(df_revd.columns):
        if N_top:
            ci = i % min(len(colors_top), len(colors_bottom))
            # revolving index for colors
            transp = 1.0 if i < N_top else 0.5
            lw = 5 if i < N_top else 1.5
            msize = 16 if i < N_top else 8
            label_wgt = 'bold' if i < N_top else 'normal'
            if i == 0:
                prod_color = 'g'
            elif i > 0 and i < N_top:
                prod_color = colors_top[ci]
            else:
                prod_color = colors_bottom[ci]
        else:
            ci = i % len(palate)
            lw, transp, msize = 1, 1, 14
            label_wgt = 'normal'
            prod_color = palate[ci]

        ax.plot(df_revd[col], linewidth=lw, color=prod_color, alpha=transp,
                marker='o', markersize=msize, zorder=len(df)-i)

        # Product labels
        ax.text(
                df.index[-1] + pd.DateOffset(label_offset),
                -labeler[names_formatted[df.columns[i]]],
                names_formatted[df.columns[i]],
                color=prod_color,alpha=transp, va='center', fontsize=16,
                fontweight=label_wgt
            )

    plt.tight_layout()
    plt.subplots_adjust(bottom=fig_margins[0], top=fig_margins[1])

    if write_path:
        plt.savefig(write_path, bbox_inches='tight', pad_inches=0.25,
                   dpi=300)

    plt.show()


def BestSeller_x_labels(df_index):
    """Construct x_axis labels for BestSellerData plots from source df
    DateIndex, formatting with year(s) included in various cases"""

    x_labels = []
    dt_form_A, dt_form_B = '%b %d', '%b %d\n(%Y)'
    years = [dt.year for dt in df_index]
    yr_idx = 0
    Jan_bool = True in [dt.month == 1 for dt in df_index]
        # True if DateIndex contains January
    multi_yr_bool = len(set(years)) > 1
        # True if DateIndex bridges multiple years
    multi_Jan_bool = [dt.month for dt in df_index].count(1) > 1
        # True if DateIndex contains multiple January dates
    counter = 0

    for i, dt in enumerate(df_index):
        # Add yr to first label if January not in DateIndex.
        if Jan_bool == False and i == 0:
            x_labels.append(datetime.strftime(dt, dt_form_B))
            counter += 1

        # Add yr to first label after a new year if not a January date
        elif (Jan_bool == False
              and dt.year != years[yr_idx] # detect new year
              and counter < len(set(years))
             ):
            x_labels.append(datetime.strftime(dt, dt_form_B))
            yr_idx += 1
            counter += 1

        # Add yr to each Jan label in an index that contains multiple
        # January anniversaries
        elif (dt.month == 1
              and multi_yr_bool
              and not multi_Jan_bool): # add year to each Jan x_label
            # in an index that contains multiple January anniversaries
            x_labels.append(datetime.strftime(dt, dt_form_B))

        # Add yr to first Jan label in an index that bridges a year beginning
        # contains multiple January dates in a single year
        elif (dt.month == 1
              and multi_yr_bool
              and multi_Jan_bool
              and counter == 0):
            x_labels.append(datetime.strftime(dt, dt_form_B))
            counter += 1

        # Add yr to first Jan label in an index that does NOT bridge a year
        # beginning but contains multiple Jan dates
        elif dt.month == 1 and not multi_yr_bool and counter == 0:
            x_labels.append(datetime.strftime(dt, dt_form_B))
            counter += 1

        else: # else, omit year from label
            x_labels.append(datetime.strftime(dt, dt_form_A))

    return x_labels



"""
~~~~~~~~~~~~~~~~~~~~~~~
Supporting, Multi-Use Functions
~~~~~~~~~~~~~~~~~~~~~~~~
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
        # yaxis_range = high
        if range_digits == high_digits:
            yaxis_range = high
        if range_digits < high_digits:
            yaxis_range = high - low
    if low < 0:
        yaxis_range = high - low

    steps = [1, 10, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000,
    25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]
    for step in steps:
        if yaxis_range / step <= max_N_ticks:
            return step


def round_to_step(val_range, low, high, max_N_ticks=10):
    """Round max or abs(min) ytick value down or up to next step"""
    step = select_step(val_range, low, high, max_N_ticks)
    multiple = (int(high) / step) + 1
    return step * multiple


def round_down_to_step(val_range, low, high, max_N_ticks=10):
    """Round low ytick value down to next step"""
    step = select_step(val_range, low, high, max_N_ticks)
    if low >= 0:
        multiple = (int(low) / step)
        return step * multiple
    else:
        multiple = abs(int(low) / step)
        return -1 * step * multiple


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
        y_low = round_down_to_step(val_range, low, high, max_N_ticks)
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
        A = 'Trends in ' + str.split(' over ')[0]
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
