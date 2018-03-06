"""
UTILITIES FOR SALES TREND ANALYSIS

CLASSES:
 -- ImportSalesData(product_id)
 -- ProductTrendsDF(ts, period_wks, end_date=None, MA_params=None,
                   exp_smooth_params=None, normed=True)
 -- RankProducts(product_stats_df, N_results=None)

MAJOR FUNCTIONS:
 -- ProductStatsDF(product_IDs, period_wks, end_date=None, MA_params=None,
                   exp_smooth_params=None, normed=True, compute_on_sales=True)
 -- CompTrendsDF(product_IDs, period_wks, end_date=None, MA_param=None,
                   exp_smooth_param=None, shifted=False, normed=False,
                   compute_on_sales=True)

A LA CARTE FUNCTIONS:
 -- compute_rolling_avg(df, window_wks, data_col='ttl_sales')
 -- slice_timeseries(data, period_wks, end_date=None)
 -- norm_Series(ts)
 -- trend_AUC(ts, normalize=False)
 -- add_rolling_avg_col(df, window_wks, data_col='ttl_sales')
"""

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from strain_dict import strain_dict, names_formatted, product_name_from_ID


class ImportSalesData(object):
    """
    Query sales data from postgres and import to pandas data objects. Initialize
    with product ID (int) or name (string) then run main() method to populate attributes

    ATTRIBUTES:
     -- product_df: pandas time series (DataFrame) with daily sales in dollars and units
     -- sales: pandas time series (Series) of total daily sales
     -- units_sold: pandas time series (Series) of total daily units sold
     -- product_id (int)
     -- product_name (string)
     -- ts_start, ts_end (Datetime) start and end dates for time series, to assist
           testing for continuity and synchronization among comparative products

    NOTE: DataFrame and Series title strings with product name and ID may be accessed
        via DataFrame.name and Series.name attributes
    """

    def __init__(self, product):
        self.product = product
        self.product_id = None
        self.product_name = None
        self._query = None
        self._connection_str = 'postgresql:///uplift'
        self._conn = None
        self.product_df = None
        self.sales = None
        self.units_sold = None
        self.ts_start, self.ts_end = None, None

    def main(self):
        self._retrieve_ID()
        self._query_product_sales()
        self._connect_to_postgres()
        self._SQL2pandasdf()

    def _retrieve_ID(self):
        if type(self.product) == str:
            key = self.product.lower()
            self.product_id = strain_dict[key]
        else:
            self.product_id = self.product

    def _query_product_sales(self):
        self._query = ("""
        SELECT CAST(DATE_TRUNC('day', date_of_sale) AS DATE) as date
         , strain_name as product_name
         , generic_strain_id as product_id
         , ROUND(SUM(retail_price)) as ttl_sales
         , ROUND(SUM(retail_units)) as ttl_units_sold
        FROM daily_sales
        WHERE generic_strain_id = {}
        GROUP BY date, strain_name, generic_strain_id
        ORDER BY date;
        """).format(self.product_id)

    def _connect_to_postgres(self):
        self._conn = create_engine(self._connection_str)

    def _SQL2pandasdf(self):
        stage_1 = pd.read_sql_query(self._query, self._conn)
        stage_2 = pd.DataFrame(stage_1[['ttl_sales', 'ttl_units_sold']])
        stage_2.index = pd.DatetimeIndex(stage_1['date'])

        # Construct continuous time series even if data is discontinuous
        self.ts_start, self.ts_end = stage_2.index[0], stage_2.index[-1]
        main_idx = pd.date_range(start=self.ts_start, end=self.ts_end)
        self.product_df = pd.DataFrame(index=main_idx)

        self.product_df['ttl_sales'] = stage_2['ttl_sales']
        self.product_df['ttl_units_sold'] = stage_2['ttl_units_sold']
        self.product_name = names_formatted[product_name_from_ID(self.product_id)]
        df_name = '{} (ID: {})'.format(self.product_name, self.product_id)
        self.product_df.name = df_name

        self.sales = self.product_df['ttl_sales']
        self.sales.name = df_name + ' -- Daily Sales'

        self.units_sold = self.product_df['ttl_units_sold']
        self.units_sold.name = df_name + ' -- Daily Units Sold'



class ProductTrendsDF(object):
    """Convert raw time series sales or unit-sales data for a single product into
    engineered trend data, including rolling averages and exponentially smoothed
    trends for both absolute and normalized (rescaled) values.

    INPUT:
     -- ts: ImportSalesData.sales or .units_sold object (pandas Series)
     -- period_wks: (int) date span of sampling period in weeks measured back
          from most recent datum or from user-supplied end_date
     -- end_date: (date string of form: '07/15/2016', default=None) alternative
          end-date for sampling period (default=None)
     -- MA_params: (list of ints, default=None) one or more rolling "boxcar"
          windows, in weeks, by which to generate distinct columns of moving-
          average data
     -- exp_smooth_params: (list of floats, default=None) one or more alpha
          smoothing factors (0 < alpha < 1) by which to generate distinct columns
          of exponentially smoothed data
     -- normed: (bool, default=True) add a column for each moving-average or
          exponentially smoothed column that computes on data rescaled (-1, 1)
          and then shifted per baseline parameter.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data by the mean
          * 'median' -- shift data by the median

    ATTRIBUTES:
     -- trendsDF: (pandas DataFrame)
     -- trend_stats: (OrderedDict) aggregate statistics, single record for insertion
          into comparison DF
     -- product_name: (str) extracted from ts.name
     -- product_ID: (int) extracted from ts.name
     -- sales_col_name: (str) either 'daily sales' or 'daily units sold', extracted
          from ts.name
     -- NaNs_ratio: (float) ratio of NaNs to total days in ts sample

    METHODS:
     -- main(): run after initialization to populate trendsDF
     -- aggregate_stats(): populates trend_stats containing record for product
          aggregated from trendsDF object
     -- norm_Series(col): rescales (-1, 1) and shifts selected data column
     -- trend_AUC(ts): computes area under curve for time series
     -- compute_aggr_slope(ts): returns slope of line describing avg growth rate
          over selected time series data
    """

    def __init__(self, ts, period_wks, end_date=None, MA_params=None,
                    exp_smooth_params=None, normed=True, baseline='t_zero'):
        self.ts = ts
        self.raw_df = None
        self.period_wks = period_wks
        self._period_days = period_wks * 7
        self.end_date = end_date
        self.MA_params = MA_params
        self.exp_smooth_params = exp_smooth_params
        self.normed = normed
        self.baseline = baseline
        self.product_name = self.ts.name.split('(')[0].strip()
        self.product_ID = int(self.ts.name.split(')')[0].split(' ')[-1])
        self.sales_col_name = self.ts.name.split(' -- Daily ')[-1]
        self.ts_sample = None
        self.trendsDF = None
        self.trend_stats = OrderedDict()
        self.NaNs_ratio = None


    def main(self):
        self._constuct_basic_trendsDF()
        if self.MA_params:
            self._compute_rolling_averages()
        # if self.exp_smooth_params:
        #     self._compute_exp_smoothed_trends()
        self.aggregate_stats()


    def _constuct_basic_trendsDF(self):
        """DF with sales over period"""
        self._slice_timeseries()
        self.trendsDF = pd.DataFrame(data=self.ts_sample.fillna(0.0).values,
                                    columns=[self.sales_col_name.lower()],
                                    index=self.ts_sample.index
                                    )
        self.NaNs_ratio = (
            self.ts_sample.isnull().sum() / float(len(self.ts_sample))
            )

        self.trendsDF.name = self._trendsDF_name()

        if self.baseline == 't_zero':
            self.trendsDF['SHIFTED to t0=0'] = \
                self.trendsDF.iloc[:,0] - self.trendsDF.iloc[:,-1][0]
        if self.baseline == 'mean':
            self.trendsDF['SHIFTED to mean=0'] = \
                self.trendsDF.iloc[:,0] - self.trendsDF.iloc[:-1].mean()
        if self.baseline == 'median':
            self.trendsDF['SHIFTED to median=0'] = \
                self.trendsDF.iloc[:,0] - np.median(self.trendsDF.iloc[:,-1])

        if self.normed:
            self.trendsDF['NORMD'] = \
                self.norm_Series(self.trendsDF.iloc[:,0])


    def _compute_rolling_averages(self):
        self.raw_df = pd.DataFrame(self.ts)
        rounder = (lambda x: round(x, 0))
        for wk_window in self.MA_params:
            boxcar = wk_window * 7
            col_name = '{}wk MA'.format(wk_window)
            self.raw_df[col_name] = \
                self.ts.fillna(0.0).rolling(window=boxcar).mean()
            self.trendsDF[col_name] = \
                self.raw_df[col_name][self.trendsDF.index].apply(rounder)

            # Shift moving averages to baseline
            if self.baseline == 't_zero':
                self.trendsDF[col_name + ' SHIFTED to t0=0'] = \
                    self.trendsDF[col_name] - self.trendsDF[col_name][0]
            if self.baseline == 'mean':
                self.trendsDF[col_name + ' SHIFTED to mean=0'] = \
                    self.trendsDF[col_name] - self.trendsDF[col_name].mean()
            if self.baseline == 'median':
                self.trendsDF[col_name + ' SHIFTED to median=0'] = \
                    self.trendsDF[col_name] - np.median(self.trendsDF[col_name])

            if self.normed:
                normed_col_name = '{}wk MA NORMD'.format(wk_window)
                self.trendsDF[normed_col_name] = \
                    self.norm_Series(self.trendsDF[col_name])


    def aggregate_stats(self):
        """Compute statistics on each data column and output trend_stats attribute
        (OrderedDict)"""
        self.trend_stats['product_name'] = self.product_name
        self.trend_stats['product_id'] = self.product_ID
        self.trend_stats['avg weekly ' + self.sales_col_name.lower()] = \
            round(self.trendsDF.iloc[:,0].sum() / self.period_wks, 0)

        if 'units' in self.sales_col_name.lower():
            sales_or_units = ' (units)'
        else:
            sales_or_units = ' ($)'

        for column in self.trendsDF.columns[1:]:
            if 'NORMD' in column:
                self.trend_stats[column + ' growth rate'] = \
                    (7 * self.compute_aggr_slope(self.trendsDF[column]))

            if 'SHIFTED' in column:
                self.trend_stats[column + ' avg weekly gain' + sales_or_units] = \
                    round(7 * self.compute_aggr_slope(self.trendsDF[column]), 0)


    def _slice_timeseries(self):
        """Construct ts_sample attribute"""
        date_index = pd.to_datetime(self.end_date)
        offset = pd.DateOffset(self._period_days - 1)
        if self.end_date:
            self.ts_sample = self.ts[date_index - offset:date_index]
        else: # else use most recent date available
            self.ts_sample = self.ts[-self._period_days:]


    def _trendsDF_name(self):
        """Construct string for trendsDF pandas DataFrame name attribute"""
        if not self.end_date:
            ending = self.ts.index[-1].strftime('%m/%d/%Y')
        else:
            ending = self.end_date

        DF_name = ('{} (ID: {}) Trends in {} over {} Weeks Ending {}').format(
            self.product_name,
            self.product_ID,
            self.sales_col_name,
            self.period_wks,
            ending
            )
        return DF_name


    def norm_Series(self, col):
        """Return time series rescaled then shifted to baseline.
        """
        values = col.fillna(0.0).values
        values = values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-50,50))
        scaler = scaler.fit(values)
        scaled_vals = scaler.transform(values).flatten()
        if self.baseline == 't_zero':
            normed_trend = pd.Series(scaled_vals - scaled_vals[0], index=col.index)
            return normed_trend
        if self.baseline == 'mean':
            normed_trend = pd.Series(scaled_vals - scaled_vals.mean(), index=col.index)
            return normed_trend
        if self.baseline == 'median':
            normed_trend = pd.Series(scaled_vals - np.median(scaled_vals), index=col.index)
            return normed_trend


    def trend_AUC(self, ts, log_scaled=False, sqrt_scaled=False):
        """Compute trend AUC or (optionally) log-scaled AUC or sqrt_scaled AUC
        for column in trendsDF
        """
        if log_scaled:
            if np.trapz(ts.values) < 0:
                return -1 * np.log(-1 * np.trapz(ts.fillna(0.0)))
            else:
                return np.log(np.trapz(ts.fillna(0.0)))

        elif sqrt_scaled:
            if np.trapz(ts.values) < 0:
                return -1 * np.sqrt(-1 * np.trapz(ts.fillna(0.0)))
            else:
                return np.sqrt(np.trapz(ts.fillna(0.0)))

        else:
            return np.trapz(ts.fillna(0.0))


    def compute_aggr_slope(self, ts):
        """Redistribute AUC under straight line and return slope of line. For
        raw figures, units represent avg sales (or units sold) gained/lost per day"""
        AUC = self.trend_AUC(ts)
        return (2 * AUC) / (len(ts)**2)



def ProductStatsDF(products, period_wks, end_date=None, MA_params=None,
                  exp_smooth_params=None, normed=True, baseline='t_zero',
                  compute_on_sales=True):
    """Construct DataFrame showing comparative sales stats among multiple products.
    See output DataFrame.name attribute for title.

    ARGUMENTS:
     -- products: (list of ints or strings) list of product names and/or IDs for
          statistical comparison
     -- period_wks: (int) sampling period in weeks

     OPTIONAL:
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date.
     -- MA_params: (list of ints, default=None) one or more rolling "boxcar"
          windows, in weeks, by which to compute moving averages
     -- exp_smooth_params: (list of floats, default=None) one or more alpha
          smoothing factors (0 < alpha < 1) by which to generate distinct columns
          of exponentially smoothed data
     -- normed: (bool, default=True) add a column for each rolling average or expon.
          smoothed column that computes on data that has been rescaled (-1, 1)
          and then shifted to baseline.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data by the mean
          * 'median' -- shift data by the median
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
    """
    data = []
    counter = 0
    df_name = None
    for product in products:
        raw_data = ImportSalesData(product)
        raw_data.main()
        if compute_on_sales:
            ts = raw_data.sales
        else:
            ts = raw_data.units_sold
        trends_data = ProductTrendsDF(ts, period_wks, end_date, MA_params,
                                     exp_smooth_params, normed, baseline)
        trends_data.main()
        data.append(trends_data.trend_stats)

        if counter < 1: # first loop, extract df name from ProductTrendsDF
            df_name = trends_data.trendsDF.name.split(') ')[1]
        counter += 1

    product_stats_df = pd.DataFrame(data, columns=data[0].keys())
    product_stats_df.name = df_name
    return product_stats_df



def CompTrendsDF(products, period_wks, end_date=None, MA_param=None,
                  exp_smooth_param=None, shifted=False, normed=False,
                  baseline='t_zero', compute_on_sales=True):
    """Construct DataFrame with time series across multiple products. Default
    arguments return a DataFrame with time series of raw sales data. Otherwise,
    assign value to either MA_param= or exp_smooth_param= (NOT BOTH). Optionally
    may assign True to either shifted= or normed= arguments (NOT BOTH).

    ARGUMENTS:
     -- products: (list of ints or strings) product names and/or IDs for
          statistical comparison
     -- period_wks: (int) sampling period in weeks
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date.
     -- MA_param: (int) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average
     -- exp_smooth_params: (float (0 < f < 1), default=None) return dataframe of
          exponentially smoothed trends; float provides alpha smoothing factor
     -- shifted: (bool, default=False) shift trend data to t0 = 0
     -- normed: (bool, default=False) rescale data to feature range (-1, 1)
          then shift data such that t0 = 0.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data to mean = 0
          * 'median' -- shift data to median = 0
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
    """

    col_index = column_sel(MA_param, exp_smooth_param, shifted, normed)
    if MA_param:
        A = '{}-Week Moving Average of '.format(MA_param) # DF title element
        MA_param = [MA_param] # insert single param to list for ProductTrendsDF class
    if exp_smooth_param:
        B = ', Exponentially Smoothed (alpha: {})'.format(exp_smooth_param)
        exp_smooth_param = [exp_smooth_param]
    counter = 0

    for product in products:
        if counter < 1: # Construct base dataframe from first product
            stage_1 = ImportSalesData(product)
            stage_1.main()
            if compute_on_sales:
                stage_1_ts = stage_1.sales
            else:
                stage_1_ts = stage_1.units_sold

            stage_2 = ProductTrendsDF(stage_1_ts, period_wks, end_date, MA_param,
                            exp_smooth_param, normed, baseline)
            stage_2.main()
            seed_df = stage_2.trendsDF

            # Construct comp_trends_df title
            bsln = baseline.capitalize() if baseline != 't_zero' else 'T0 = 0'
            C = ', Data Shifted to {}'.format(bsln)
            D = ', Data Rescaled (-50, 50) then Shifted to {}'.format(bsln)
            E = seed_df.name.split('in ')[1]
            if col_index == 0:
                title = E
            if col_index == 1:
                title = E + C
            if col_index == 2:
                title = E + D
            if col_index == 3 and MA_param and not shifted:
                title = A + E
            if col_index == 3 and exp_smooth_param and not shifted:
                title = E + B
            if col_index == 4 and MA_param and shifted:
                title = A + E + C
            if col_index == 4 and exp_smooth_param and shifted:
                title = E + B + C
            if col_index == 5 and MA_param and normed:
                title = A + E + D
            if col_index == 5 and exp_smooth_param and normed:
                title = E + B + D

            col_name = [stage_2.product_name]
            comp_trends_df = pd.DataFrame(seed_df[seed_df.columns[col_index]])
            comp_trends_df.columns = [stage_2.product_name]
            comp_trends_df.name = title
            counter += 1

        else:  # Populate dataframe with remaining product trends
            stage_1 = ImportSalesData(product)
            stage_1.main()
            if compute_on_sales:
                stage_1_ts = stage_1.sales
            else:
                stage_1_ts = stage_1.units_sold

            stage_2 = ProductTrendsDF(stage_1_ts, period_wks, end_date, MA_param,
                            exp_smooth_param, normed, baseline)
            stage_2.main()
            source_df = stage_2.trendsDF

            comp_trends_df[stage_2.product_name] = source_df.iloc[:,col_index]

    return comp_trends_df


def column_sel(MA_param=None, exp_smooth_param=None, shifted=False, normed=False):
    """Return integer for DataFrame column selection"""
    if MA_param or exp_smooth_param:
        smoothed = True
    else:
        smoothed = False

    if not smoothed and not (shifted or normed):
        return 0
    if not smoothed and shifted:
        return 1
    if not smoothed and normed:
        return 2
    if smoothed and not (shifted or normed):
        return 3
    if smoothed and shifted:
        return 4
    if smoothed and normed:
        return 5


class RankProducts(object):
    """Initialize with ProductStatsDF object and (optionally) by number of top
    results desired; Rank products by user_selected statistic

    METHOD:
     -- main(): Rank products and populate attributes using kwargs:
          * smoothed (bool, default = True) rank on statistics generated from
              trend lines smoothed via moving average or exponential alpha;
              False = rank on raw trend data
          * stat (str or Nonetype, default = 'sales')
              - 'sales' (default)= average weekly sales over period; NOTE:
              - 'gain' = uniform weekly gain or loss over period
              - 'rate' = growth rate index for products with data
                  normalized (rescaled -100, 100) for sales volumes
              -  None = prompts user for statistic from menu

    ATTRIBUTES:
     -- results: pandas DataFrame of products ranked by selected statistic
     -- ranked_IDs: numpy array of ranked product IDs
     -- ranked_df: same as RankProducts.results but including all other statistics

    """

    def __init__(self, product_stats_df, N_results=None):
        self.product_stats_df = product_stats_df
        self.N_results = N_results
        self.results = None
        self.ranked_IDs = None
        self.ranked_df = None


    def main(self, smoothed=True, stat='sales'):
        """Rank N-top products by a user-selected statistic specified either by
        smoothed and stat keyword arguments, or manually by selection off of menu

        OUTPUT: class attributes -- results, ranked_IDs, ranked_df

        ARGUMENTS:
          * smoothed (bool, default = True) rank on statistics generated from
              trend lines smoothed via moving average or exponentially;
              False = rank on raw trend data
          * stat (str or NoneType, default = 'sales') rank products on ...
              - 'sales' (default) = average weekly sales over period
              - 'gain' = uniform weekly gain or loss over period
              - 'rate' = growth rate for products with trend data rescaled (-50, 50)
                  to offset variation in overall sales volumes among products
              -  None = prompts user for selection of ranking statistic from menu

        """

        if stat:
            cols = self.product_stats_df.columns
            if stat == 'sales':
                stat_idx = 2
            if not smoothed and stat == 'gain':
                stat_idx = 3
            if not smoothed and stat == 'rate':
                stat_idx = 4
            if smoothed and stat == 'gain':
                stat_idx = 5
            if smoothed and stat == 'rate':
                stat_idx = 6

            stat_col = cols[stat_idx]

        else:
            stat_idx = self._sel_rank_by()
            stat_col = self.product_stats_df.columns[stat_idx]

        output_cols = list(self.product_stats_df.columns)
        output_cols.remove(stat_col)
        output_cols.insert(2, stat_col)

        ranked = self.product_stats_df.sort_values(by=stat_col, ascending=False)
        ranked.index = range(1, len(ranked.index) + 1)

        if self.N_results:
            self.ranked_df = ranked[output_cols][:self.N_results]
            self.ranked_IDs = self.ranked_df['product_id'].values
            self.results = self.ranked_df.iloc[:,:3]
        else:
            self.ranked_df = ranked[output_cols]
            self.ranked_IDs = self.ranked_df['product_id'].values
            self.results = self.ranked_df.iloc[:,:3]

        self.results.name = self.product_stats_df.name
        self.ranked_df.name = self.product_stats_df.name + \
                ', Ranked by {}'.format(stat_col)


    def _sel_rank_by(self):
        "Prompt user for column for ranking; return its index"
        cols = self.product_stats_df.columns[2:]
        index = range(1, len(cols) + 1)
        menu = dict(zip(index, cols))
        for k, v in menu.iteritems():
            print(str(k) + ' -- ' + v)
        selection = int(raw_input('\nSelect statistic for ranking.'))
        return selection + 1

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GENERATE DATA FOR BAR GRAPHS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def HbarData(product_IDs, period_wks=10, end_date=None,
               rank_on_sales=True, MA=5,
               rank_by=['rate'], fixed_order=True):
    """Return a dataframe configured for custom plotting in HbarRanked function"""

    boxcar = [MA] if MA else None
    prod_stats = ProductStatsDF(product_IDs, period_wks, end_date,
                MA_params=boxcar, compute_on_sales=rank_on_sales)
    if MA:
        base_name = prod_stats.name + ' -- {}-Week Moving Average'.format(MA)
    else:
        base_name = prod_stats.name + ' -- '


    if len(rank_by) < 2 or fixed_order: # just need the RankProducts.results object
        if len(rank_by) < 2:
            rank_1 = RankProducts(prod_stats)
            if MA:
                rank_1.main(stat=rank_by[0])
            else:
                rank_1.main(smoothed=False, stat=rank_by[0])
            data = rank_1.results
            data.drop(['product_id'], axis=1, inplace=True)

        else:
            rank_1 = RankProducts(prod_stats)
            rank_1.main(smoothed=MA, stat=rank_by[0])
            all_data = rank_1.ranked_df
            df_cols = all_data.columns
            cols = []
            for rank_stat in rank_by:
                cols.append('product_name')
                cols.append(grab_column(stat=rank_stat, smoothed=MA))

            data = all_data[cols]


    if len(rank_by) > 1 and not fixed_order:
            rank_1 = RankProducts(prod_stats)
            rank_1.main(smoothed=MA, stat=rank_by[0])
            data = rank_1.results

            for i, rank_stat in enumerate(rank_by[1:]):
                rank_next = RankProducts(prod_stats)
                rank_next.main(smoothed=MA, stat=rank_stat)
                next_ranked = rank_next.results
                data['Ranking By {}'.format(rank_stat)] = next_ranked.iloc[:,0].values
                data[next_ranked.columns[-1]] = next_ranked.iloc[:,-1].values

            data.drop(['product_id'], axis=1, inplace=True)

    data = data[::-1] # reverse row order for matplotlib bar graphing
    data.name = base_name

    return data


def grab_column(stat, smoothed):
    """Return index for data column in HbarData fixed_order bar graph"""
    if stat == 'sales':
        return 'avg weekly sales'
    if not smoothed and stat == 'gain':
        return 'SHIFTED to t0=0 avg weekly gain ($)'
    if not smoothed and stat == 'rate':
        return 'NORMD growth rate'
    if smoothed and stat == 'gain':
        return '{}wk MA SHIFTED to t0=0 avg weekly gain ($)'.format(smoothed)
    if smoothed and stat == 'rate':
        return '{}wk MA NORMD growth rate'.format(smoothed)





"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GENERATE BEST-SELLER DATA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def BestSellerData(products, end_date=None, period_wks=10, MA_param=5,
            compute_on_sales=True, N_periods=10, freq='7D', rank_by='rate'):
    """Return dataframes summarizing rankings for products over a series
    of N-identical-length periods spaced at equal intervals.

    OUTPUT: tuple (df_A, df_B)
     -- df_A: DataFrame of product_IDs ranked for each period
     -- df_B: DataFrame of rankings for each product indexed by date and column-
         sorted left to right by cumulative highest ranks over all periods; This
         df is suitable for slope-graphing.

    ARGUMENTS:
     -- products: (list of ints or strings) product names and/or IDs for
          statistical comparison
     -- end_date: (date string of form 'MM/DD/YYYY', default=None) end_date of most recent
         ranking period; default uses most recent date in dataset.
     -- period_wks: (int, default=10) length of sampling periods in weeks
     -- MA_param: (int or NoneType, default=5) rolling "boxcar" window, in weeks, by which to
          compute moving averages; None: ranks on non-smoothed (raw) trend data
     -- compute_on_sales: (bool, default=True) ranks on sales data; if False,
          ranks on units-sold data
     -- N_periods: (int, default=10) number of periods including latest for comparison
     -- freq: (str, default='7D') pandas date_range() argument; interval between
         periods for comparison. Other possible values: 'W' (Sunday week), 'M' (month),
         'Y', '2W', etc.
     -- rank_by: (string, default='rate') statistic by which to rank products.
          Values:
          * 'rate' = growth rate index for products with data
              normalized (rescaled -100, 100) for sales volumes
          * 'gain' = uniform weekly gain or loss over period
          * 'sales' = cumulative sales over period
    """

    # Generate list of end_dates for periods
    if end_date:
        end_dates = generate_dates(end_date, N_periods, freq)
    else: # If not provided as argument, grab most recent date in data index
        ### Bug?: if first product does not have latest date and dates do not overlap
        imp = ImportSalesData(products[0])
        imp.main()
        seed_ts = imp.sales
        dt_obj = seed_ts.index[-1]
        end_date = datetime.strftime(dt_obj, '%m/%d/%Y')
        end_dates = generate_dates(end_date, N_periods, freq)

    # Generate data consisting of product rankings over multiple periods
    data_A = OrderedDict()
###    name_dict = {} # to map names to IDs for df labels
    for i, end_d in enumerate(end_dates):
        # compute stats on products
        if MA_param:
            psdf = ProductStatsDF(products, period_wks=period_wks, end_date=end_d,
                    MA_params=[MA_param], normed=True, compute_on_sales=compute_on_sales
                            )
        else: # if no moving-avg sliding window provided, omit argument
            psdf = ProductStatsDF(products, period_wks=period_wks, end_date=end_d,
                    normed=True, compute_on_sales=compute_on_sales
                            )

        # Generate rankings and add to data dictionary (data_A) where keys =
        # the end_dates of the comparison periods; values = the product IDs
        # ordered by rank
        ranked = RankProducts(psdf)
        if MA_param:
            ranked.main(smoothed=True, stat=rank_by)
        else:
            ranked.main(smoothed=False, stat=rank_by)
        data_A[end_d] = ranked.ranked_IDs

    # Reconfigure data_A into a dictionary (data_B) of keys=products, vals=list
    # of a product's rankings over the series of comparison periods
    data_B = OrderedDict()
    for prod in products:
        if type(prod) == str:
            data_B[prod.lower()] = []
        else:
            data_B[product_name_from_ID(prod)] = []
    for prod_arr in data_A.itervalues():
        for i, prod in enumerate(prod_arr):
            if type(prod) == str:
                data_B[prod.lower()].append(i+1) # i+1 represents product rank
            else:
                data_B[product_name_from_ID(prod)].append(i+1)

    # Construct output dataframes
    mask = lambda x: datetime.strptime(x, '%m/%d/%Y')
    date_idx = [mask(dt) for dt in end_dates] # for DatetimeIndex of df_B

    title = best_seller_title(MA_param, compute_on_sales, N_periods,
                          period_wks, rank_by, freq)

    df_A = pd.DataFrame(data_A, index=range(1, len(products)+1))
    # index of df_A represents rank levels 1 to N
    df_B = pd.DataFrame(data_B, index=date_idx)

    # Sort df_B columns by cumulative rankings
    sum_o_ranks = df_B.sum()
    foo = sum_o_ranks.sort_values(ascending=True)
    sorted_by_best = list(foo.index)
    df_B = df_B[sorted_by_best]

    df_A.name = title
    df_B.name = title

    labels = [names_formatted[product_name_from_ID(prod_ID)] \
              for prod_ID in df_A.iloc[:,-1]]
    labeler = {}
    for i, prod in enumerate(labels):
        labeler[prod] = i + 1


    return df_A, df_B, labeler



def generate_dates(end_date, N_periods=10, freq='7D'):
    """Make list of end_dates for ranking periods based on BestSellerData params."""
    end_dates = []
    last_date = datetime.strptime(end_date, '%m/%d/%Y')
    d_range = pd.date_range(end=last_date, periods=N_periods, freq=freq)
    for d in d_range:
        str_d = d.strftime('%m/%d/%Y')
        end_dates.append(str_d)

    return end_dates

def parse_freq(freq):
    if freq == '7D' or freq == 'W':
        return 'at weekly intervals'
    if 'W' in freq and len(freq) > 1:
        mult = list(freq)[0]
        return 'at {}-week intervals'.format(mult)
    if freq == 'M':
        return 'at monthly intervals'
    if 'M' in freq and len(freq) > 1:
        mult = list(freq)[0]
        return 'at {}-month intervals'.format(mult)
    if freq == 'Y':
        return 'spaced annually'

def best_seller_title(MA_param, compute_on_sales, N_periods, period_wks,
                     rank_by, freq):
    "Construct title (pandas.DataFrame.name) for BestSellerData objects."

    if rank_by == 'rate':
        alpha = 'Relative Growth Rate'
    if rank_by == 'gain':
        alpha = 'Uniform Weekly Gain/Loss in Sales'
    if rank_by == 'sales':
        alpha = 'Average Weekly Sales'
    A = 'Successive Rankings on {}'.format(alpha)

    B = 'Rankings over {} consecutive {}-week periods, '.format(N_periods,
                                                    period_wks)
    beta = parse_freq(freq)
    C = 'spaced {}'.format(beta)

    gamma = 'sales.' if compute_on_sales else 'units sold.'
    if MA_param:
        D = '\nComputed on {}-week moving-average trends in daily {}\n'\
        .format(MA_param, gamma)
    else:
        D = '\nComputed on trends in daily {}'.format(gamma)

    return A + ' -- ' + B + C + D

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A LA CARTE FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def rank_products(product_stats_df, N_results=None):
    "Rank N-top products by user-selected statistic; output in pandas DataFrame"
    stat_idx = sel_rank_by(product_stats_df)
    stat_col = product_stats_df.columns[stat_idx]
    ranked_df = product_stats_df.sort_values(by=stat_col, ascending=False)
    if N_results:
        return ranked_df[['product_name', 'product_id', stat_col]][:N_results]
    else:
        return ranked_df[['product_name', 'product_id', stat_col]]


def sel_rank_by(product_stats_df):
    "Prompt user for column for ranking; return its index"
    cols = product_stats_df.columns[2:]
    index = range(1, len(cols) + 1)
    menu = dict(zip(index, cols))
    for k, v in menu.iteritems():
        print(str(k) + ' -- ' + v)
    selection = int(raw_input('\nSelect statistic for ranking.'))
    return selection + 1


def compute_rolling_avg(ts, window_wks):
    """
    INPUT: complete time series (Series) and moving 'boxcar' window in weeks
    OUTPUT: rolling average values
    """
    boxcar = window_wks * 7
    return ts.rolling(window=boxcar).mean()


def slice_timeseries(ts, period_wks, end_date=None):
    """Enter period in weeks and an optional end_date str ('07/31/2017')
    Returns sliced Series
    """
    days = period_wks * 7
    if end_date:
        return ts[end_date - days:end_date]
    else:
        return ts[-days:]

def norm_Series(ts):
    """Returns time series normalized then shifted such that t0 = 0
    NOTE: Due to shifting, some normed values may exceed the feature range (-1, 1)
    """
    values = ts.values
    values = values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(values)
    normed_trend = scaler.transform(values).flatten()
    normed_trend = pd.Series(normed_trend - normed_trend[0], index=ts.index)
    normed_trend.name = ts.name + ' NORMED'
    return normed_trend

def trend_AUC(ts, normalize=False):
    """
    INPUT: trend data in time series (pandas.Series)
    OUTPUT:
     -- default: area under curve (AUC) for shifted trend data
     -- normalize=True: AUC for normed then shifted trend data
    """
    if normalize:
        normed_trend = norm_Series(ts)
        values = normed_trend.values
        values = values - values[0]
        return np.trapz(values)

    else:
        values = ts.values
        values = values - values[0]
        return np.trapz(values)

def add_rolling_avg_col(df, window_wks, data_col='ttl_sales'):
    """Add rolling average column to ImportSalesData.product_df object"""
    boxcar = window_wks * 7
    col = 'rolling_{}wk'.format(window_wks)
    df[col] = df[data_col].rolling(window=boxcar).mean()



if __name__=='__main__':

    """Set input variables"""
    products = range(1, 10) # list of product IDs
    MAs = [5] # list of moving average window(s) in weeks
    sample_period = 20 # in weeks

    """Run ImportSalesData method and access class attributes"""
    product_3 = ImportSalesData(3)
    product_3.main()
    raw_df_3 = product_3.product_df # DataFrame of daily sales and units for product
    sales_3 = product_3.sales # time series (pd.Series) of daily sales
    units_3 = product_3.units_sold # time Series of daily units sold

    """Run ProductTrendsDF method and access class attributes"""
    trends_3 = ProductTrendsDF(sales_3, sample_period, MA_params=MAs)
    trends_3.main()
    trends_df_3 = trends_3._trendsDF # DataFrame with columns of transformed data
    stats_3 = trends_3.trend_stats # Single record (OrderedDict) of stats for product

    """Run ProductStatsDF function to generate comparative stats DF; Builds DF from
    individual records in the form of ProductTrendsDF.trend_stats objects"""
    comps_df = ProductStatsDF(products, sample_period, MA_params=MAs)

    """Print various attributes (names, DFs, Series) to test pipeline"""
    print(raw_df_3.name)
    print(raw_df_3.head(2))

    print('\n' + sales_3.name)
    print(sales_3.head(2))

    print('\n' + units_3.name)
    print(units_3.head(2))

    print('\n' + trends_df_3.name)
    print(trends_df_3.head(2))

    print('\n')
    print(stats_3)
    print('\n')
    print(comps_df)
