"""
UTILITIES FOR SALES TREND ANALYSIS

CLASSES:
 -- StrainSalesDF(strain_id)
 -- StrainTrendsDF(ts, period_wks, end_date=None, MA_params=None,
                   exp_smooth_params=None, normed=True)
 -- RankStrains(strain_stats_df, N_results=None)

MAJOR FUNCTIONS:
 -- StrainStatsDF(strain_IDs, period_wks, end_date=None, MA_params=None,
                   exp_smooth_params=None, normed=True, compute_on_sales=True)
 -- CompTrendsDF(strain_IDs, period_wks, end_date=None, MA_param=None,
                   exp_smooth_param=None, shifted=False, normed=False,
                   compute_on_sales=True)

A LA CARTE FUNCTIONS:
 -- compute_rolling_avg(df, window_wks, data_col='ttl_sales')
 -- slice_timeseries(data, period_wks, end_date=None)
 -- norm_Series(ts)
 -- trend_AUC(ts, normalize=False)
 -- add_rolling_avg_col(df, window_wks, data_col='ttl_sales')
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


class StrainSalesDF(object):
    """
    Initialize with strain_id (int) then run main() method to populate
    attributes

    ATTRIBUTES:
     -- strain_df: pandas time series (DataFrame) with daily sales in dollars and units
     -- sales: pandas time series (Series) of total daily sales
     -- units_sold: pandas time series (Series) of total daily units sold
     -- strain_id (int)
     -- strain_name (string)

    NOTE: DataFrame and Series title strings with strain name and ID may be accessed
        via DataFrame.name and Series.name attributes
    """

    def __init__(self, strain_id):
        self.strain_id = strain_id
        self.strain_name = None
        self._query = None
        self._connection_str = 'postgresql:///uplift'
        self._conn = None
        self.strain_df = None
        self.sales = None
        self.units_sold = None

    def main(self):
        self._query_strain_sales()
        self._connect_to_postgres()
        self._SQL2pandasdf()

    def _query_strain_sales(self):
        self._query = ("""
        SELECT CAST(DATE_TRUNC('day', ds.date_of_sale) AS DATE) as date
         , st.strain_display_name as strain_name
         , st.generic_strain_id as strain_id
         , ROUND(SUM(ds.retail_price)) as ttl_sales
         , ROUND(SUM(ds.retail_units)) as ttl_units_sold
        FROM daily_retail_sales ds
        JOIN strains st
        ON ds.strain_name = st.strain_display_name
        WHERE st.generic_strain_id = {}
        GROUP BY date, st.strain_display_name, st.generic_strain_id
        ORDER BY date;
        """).format(self.strain_id)

    def _connect_to_postgres(self):
        self._conn = create_engine(self._connection_str)

    def _SQL2pandasdf(self):
        raw_df = pd.read_sql_query(self._query, self._conn)
        self.strain_df = pd.DataFrame(raw_df[['ttl_sales', 'ttl_units_sold']])
        self.strain_df.index = pd.DatetimeIndex(raw_df['date'])
        self.strain_name = raw_df['strain_name'].unique()[0]
        df_name = '{} (ID: {})'.format(self.strain_name, self.strain_id)
        self.strain_df.name = df_name

        self.sales = self.strain_df['ttl_sales']
        self.sales.name = df_name + ' -- Daily Sales'

        self.units_sold = self.strain_df['ttl_units_sold']
        self.units_sold.name = df_name + ' -- Daily Units Sold'



class StrainTrendsDF(object):
    """Convert raw time series sales or unit-sales data for a single strain into
    engineered trend data, including rolling averages and exponentially smoothed
    trends for both absolute and normalized (rescaled) values.

    INPUT:
     -- ts: StrainSalesDF.sales or .units_sold object (pandas Series)
     -- period_wks: (int) date span in weeks measured back from most recent datum,
          used to define sampling period
     -- end_date: (date string of form: '07/15/2016', default=None) alternative
          end-date for sampling period (default=None)
     -- MA_params: (list of ints, default=None) one or more rolling "boxcar"
          windows, in weeks, by which to generate distinct columns of moving-
          average data
     -- exp_smooth_params: (list of floats, default=None) one or more alpha
          smoothing factors (0 < alpha < 1)by which to generate distinct columns
          of exponentially smoothed data
     -- normed: (bool, default=True) add a column for each moving-average or
          exponentially smoothed column that computes on data rescaled (-1, 1)
          and then shifted such that datum at t0 = 0.

    ATTRIBUTES:
     -- trendsDF: (pandas DataFrame)
     -- trend_stats: (OrderedDict) aggregate statistics, single record for insertion
          into comparison DF
     -- strain_name: (str) extracted from ts.name
     -- strain_ID: (int) extracted from ts.name
     -- sales_col_name: (str) either 'daily sales' or 'daily units sold', extracted
          from ts.name

    METHODS:
     -- main(): run after initialization to populate trendsDF
     -- aggregate_stats(): populates trend_stats containing record for strain
          aggregated from trendsDF object
     -- norm_Series(col): rescales (-1, 1) and shifts selected data column
     -- trend_AUC(ts): computes area under curve for time series
     -- compute_aggr_slope(ts): returns slope of line describing avg growth rate
          over selected time series data
    """

    def __init__(self, ts, period_wks, end_date=None, MA_params=None,
                    exp_smooth_params=None, normed=True):
        self.ts = ts
        self.raw_df = None
        self.period_wks = period_wks
        self._period_days = period_wks * 7
        self.end_date = end_date
        self.MA_params = MA_params
        self.exp_smooth_params = exp_smooth_params
        self.normed = normed
        self.strain_name = self.ts.name.split('(')[0].strip()
        self.strain_ID = int(self.ts.name.split(')')[0].split(' ')[-1])
        self.sales_col_name = self.ts.name.split(' -- ')[-1]
        self.ts_sample = None
        self.trendsDF = None
        self.trend_stats = OrderedDict()


    def main(self):
        self._constuct_basic_trendsDF()
        if self.MA_params:
            self._compute_rolling_averages()
        if self.exp_smooth_params:
            self._compute_exp_smoothed_trends()
        self.aggregate_stats()


    def _constuct_basic_trendsDF(self):
        """DF with sales over period"""
        self._slice_timeseries()
        self.trendsDF = pd.DataFrame(data=self.ts_sample.values,
                                    columns=[self.sales_col_name.lower()],
                                    index=self.ts_sample.index
                                    )

        self.trendsDF.name = self._trendsDF_name()


    def _compute_rolling_averages(self):
        self.raw_df = pd.DataFrame(self.ts)
        rounder = (lambda x: round(x, 0))
        for wk_window in self.MA_params:
            boxcar = wk_window * 7
            col_name = '{}wk MA'.format(wk_window)
            self.raw_df[col_name] = self.ts.rolling(window=boxcar).mean()
            self.trendsDF[col_name] = \
                self.raw_df[col_name][self.trendsDF.index].apply(rounder)
            # Shift moving averages to t0 = 0
            self.trendsDF[col_name + ' SHIFTED'] = \
                self.trendsDF[col_name] - self.trendsDF[col_name][0]

            if self.normed:
                normed_col_name = '{}wk MA NORMD'.format(wk_window)
                self.trendsDF[normed_col_name] = \
                    self.norm_Series(self.trendsDF[col_name])
                    # This takes the shifted MA values that start with zero,
                    #  rescales them (-1, 1), then shifts them again to zero.


    def aggregate_stats(self):
        """Construct trend_stats from trendsDF: output is a dictionary for appending
        to list of dicts as input for pandas DF"""
        self.trend_stats['strain_name'] = self.strain_name
        self.trend_stats['strain_id'] = self.strain_ID
        self.trend_stats['cumulative ' + self.sales_col_name.lower()] = \
            sum(self.trendsDF[self.trendsDF.columns[0]])
        if 'units' in self.sales_col_name.lower():
            sales_or_units = ' (units)'
        else:
            sales_or_units = ' ($)'

        for column in self.trendsDF.columns[1:]:
            if 'NORMD' in column:
                self.trend_stats[column + ' AUC'] = \
                    self.trend_AUC(self.trendsDF[column])
                self.trend_stats[column + ' SLOPE'] = \
                    self.compute_aggr_slope(self.trendsDF[column])

            elif 'SHIFTED' in column:
                ## Un-comment two lines below for un-scaled AUC
                # self.trend_stats[column + ' AUC'] = \
                #     self.trend_AUC(self.trendsDF[column])
                self.trend_stats[column + ' log-scaled AUC'] = \
                    self.trend_AUC(self.trendsDF[column], log_scaled=True)
                self.trend_stats[column + ' avg weekly gain' + sales_or_units] = \
                    round(7 * self.compute_aggr_slope(self.trendsDF[column]), 0)

            else:
                self.trend_stats[column + ' log-scaled AUC'] = \
                    self.trend_AUC(self.trendsDF[column], log_scaled=True)

    def _slice_timeseries(self):
        """Construct ts_sample attribute"""
        date_index = pd.to_datetime(self.end_date)
        offset = pd.DateOffset(self._period_days - 1)
        if self.end_date:
            self.ts_sample = self.ts[date_index - offset:date_index]
        else:
            self.ts_sample = self.ts[-self._period_days:]


    def _trendsDF_name(self):
        """Construct string for trendsDF pandas DataFrame name attribute"""
        if not self.end_date:
            ending = self.ts.index[-1].strftime('%m/%d/%Y')
        else:
            ending = self.end_date

        DF_name = ('{} (ID: {}) Trends in {} over {} Weeks Ending {}').format(
            self.strain_name,
            self.strain_ID,
            self.sales_col_name,
            self.period_wks,
            ending
            )
        return DF_name


    def norm_Series(self, col):
        """Returns time series rescaled then shifted such that t0 = 0
        NOTE: Due to shifting, some normed values may exceed the feature range (-1, 1)
        """
        values = col.values
        values = values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(values)
        scaled_vals = scaler.transform(values).flatten()
        normed_trend = pd.Series(scaled_vals - scaled_vals[0], index=col.index)
        return normed_trend


    def trend_AUC(self, ts, log_scaled=False, sqrt_scaled=False):
        """Compute trend AUC or (optionally) log-scaled AUC or sqrt_scaled AUC
        for column in trendsDF
        """
        if log_scaled:
            if np.trapz(ts.values) < 0:
                return -1 * np.log(-1 * np.trapz(ts.values))
            else:
                return np.log(np.trapz(ts.values))

        elif sqrt_scaled:
            if np.trapz(ts.values) < 0:
                return -1 * np.sqrt(-1 * np.trapz(ts.values))
            else:
                return np.sqrt(np.trapz(ts.values))

        else:
            return np.trapz(ts.values)


    def compute_aggr_slope(self, ts):
        """Redistribute AUC under straight line and return slope of line. For
        raw figures, units represent avg sales (or units sold) gained/lost per day"""
        AUC = self.trend_AUC(ts)
        return (2 * AUC) / (len(ts)**2)



def StrainStatsDF(strain_IDs, period_wks, end_date=None, MA_params=None,
                  exp_smooth_params=None, normed=True, compute_on_sales=True):
    """Construct DataFrame showing comparative sales stats among multiple products.
    See output DataFrame.name attribute for title.

    ARGUMENTS:
     -- strain_IDs: (list of ints) list of strain IDs for statistical comparison
     -- period_wks: (int) sampling period in weeks

     OPTIONAL
     -- end_date: (date string: '07/15/2016', default=None) date string defining
          end of sampling period. Default uses most recent date.
     -- MA_params: (list of ints, default=None) one or more rolling "boxcar"
          windows, in weeks, by which to compute moving averages
     -- exp_smooth_params: (list of floats, default=None) one or more alpha
          smoothing factors (0 < alpha < 1) by which to generate distinct columns
          of exponentially smoothed data
     -- normed: (bool, default=True) add a column for each rolling average or expon.
          smoothed column that computes on data that has been rescaled (-1, 1)
          and then shifted such that datum at t0 = 0.
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
    """
    data = []
    counter = 0
    df_name = None
    for strain in strain_IDs:
        raw_data = StrainSalesDF(strain)
        raw_data.main()
        if compute_on_sales:
            ts = raw_data.sales
        else:
            ts = raw_data.units_sold
        trends_data = StrainTrendsDF(ts, period_wks, end_date, MA_params,
                                     exp_smooth_params, normed)
        trends_data.main()
        data.append(trends_data.trend_stats)
        if counter < 1:
            df_name = trends_data.trendsDF.name.split(') ')[1]
        counter += 1

    strain_stats_df = pd.DataFrame(data, columns=data[0].keys())
    strain_stats_df.name = df_name
    return strain_stats_df



def CompTrendsDF(strain_IDs, period_wks, end_date=None, MA_param=None,
                  exp_smooth_param=None, shifted=False, normed=False,
                  compute_on_sales=True):
    """Construct DataFrame with time series across multiple strains. Default
    arguments return a DataFrame with time series of raw sales data. Otherwise,
    assign value to either MA_param= or exp_smooth_param= (NOT BOTH). Optionally
    may assign True to either shifted= or normed= arguments (NOT BOTH).

    ARGUMENTS:
     -- strain_IDs: (list of ints) list of strain IDs for comparison
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
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
    """

    col_index = column_sel(MA_param, exp_smooth_param, shifted, normed)
    if MA_param:
        A = '{}-Week Moving Average of '.format(MA_param) # DF title element
        MA_param = [MA_param] # insert single param to list for StrainTrendsDF class
    if exp_smooth_param:
        B = ', Exponentially Smoothed (alpha: {})'.format(exp_smooth_param)
        exp_smooth_param = [exp_smooth_param]
    counter = 0

    for strain in strain_IDs:
        # Construct base dataframe from first strain
        if counter < 1:
            stage_1 = StrainSalesDF(strain)
            stage_1.main()
            if compute_on_sales:
                stage_1_ts = stage_1.sales
            else:
                stage_1_ts = stage_1.units_sold

            stage_2 = StrainTrendsDF(stage_1_ts, period_wks, end_date, MA_param,
                            exp_smooth_param, normed)
            stage_2.main()
            seed_df = stage_2.trendsDF

            # Construct comp_trends_df title
            C = ', Data Shifted t0=0'
            D = ', Data Rescaled (-1, 1) and Shifted (t0=0)'
            E = seed_df.name.split('in ')[1]
            if col_index == 0:
                title = E
            if col_index == 1 and MA_param and not shifted:
                title = A + E
            if col_index == 1 and exp_smooth_param and not shifted:
                title = E + B
            if col_index == 2 and MA_param and shifted:
                title = A + E + C
            if col_index == 2 and exp_smooth_param and shifted:
                title = E + B + C
            if col_index == 3 and MA_param and normed:
                title = A + E + D
            if col_index == 3 and exp_smooth_param and normed:
                title = E + B + D

            col_name = [stage_2.strain_name]
            comp_trends_df = pd.DataFrame(seed_df[seed_df.columns[col_index]])
            comp_trends_df.columns = [stage_2.strain_name]
            comp_trends_df.name = title
            counter += 1

        # Populate dataframe with remaining strain trends
        else:
            stage_1 = StrainSalesDF(strain)
            stage_1.main()
            if compute_on_sales:
                stage_1_ts = stage_1.sales
            else:
                stage_1_ts = stage_1.units_sold

            stage_2 = StrainTrendsDF(stage_1_ts, period_wks, end_date, MA_param,
                            exp_smooth_param, normed)
            stage_2.main()
            source_df = stage_2.trendsDF

            comp_trends_df[stage_2.strain_name] = source_df.iloc[:,col_index]


    return comp_trends_df



def column_sel(MA_param=None, exp_smooth_param=None, shifted=False, normed=False):
    """Return integer for DataFrame column selection"""
    if MA_param or exp_smooth_param:
        smoothed = True
    else:
        return 0
    if smoothed and not (shifted or normed):
        return 1
    if smoothed and shifted:
        return 2
    if smoothed and normed:
        return 3


class RankStrains(object):
    """Initialize with StrainStatsDF object and (optionally) by number of top
    results desired; Rank strains/products by user_selected statistic

    METHOD:
     -- main(): Rank strains and populate attributes

    ATTRIBUTES:
     -- results: pandas DataFrame of strains ranked by selected statistic
     -- ranked_IDs: numpy array of ranked strain IDs
     -- ranked_df: same as RankStrains.results but including all other statistics

    """

    def __init__(self, strain_stats_df, N_results=None):
        self.strain_stats_df = strain_stats_df
        self.N_results = N_results
        self.results = None
        self.ranked_IDs = None
        self.ranked_df = None


    def main(self):
        "Rank N-top strains by user-selected statistic; output in pandas DataFrame"
        stat_idx = self._sel_rank_by()
        stat_col = self.strain_stats_df.columns[stat_idx]
        output_cols = list(self.strain_stats_df.columns)
        output_cols.remove(stat_col)
        output_cols.insert(2, stat_col)

        ranked = self.strain_stats_df.sort_values(by=stat_col, ascending=False)
        ranked.index = range(1, len(ranked.index) + 1)

        if self.N_results:
            self.ranked_df = ranked[output_cols][:self.N_results]
            self.ranked_IDs = self.ranked_df['strain_id'].values
            self.results = self.ranked_df.iloc[:,:3]
        else:
            self.ranked_df = ranked[output_cols]
            self.ranked_IDs = self.ranked_df['strain_id'].values
            self.results = self.ranked_df.iloc[:,:3]

        self.results.name = self.strain_stats_df.name
        self.ranked_df.name = self.strain_stats_df.name + \
                ', Ranked by {}'.format(stat_col)


    def _sel_rank_by(self):
        "Prompt user for column for ranking; return its index"
        cols = self.strain_stats_df.columns[2:]
        index = range(1, len(cols) + 1)
        menu = dict(zip(index, cols))
        for k, v in menu.iteritems():
            print(str(k) + ' -- ' + v)
        selection = int(raw_input('\nSelect statistic for ranking.'))
        return selection + 1


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A LA CARTE FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def rank_strains(strain_stats_df, N_results=None):
    "Rank N-top strains by user-selected statistic; output in pandas DataFrame"
    stat_idx = sel_rank_by(strain_stats_df)
    stat_col = strain_stats_df.columns[stat_idx]
    ranked_df = strain_stats_df.sort_values(by=stat_col, ascending=False)
    if N_results:
        return ranked_df[['strain_name', 'strain_id', stat_col]][:N_results]
    else:
        return ranked_df[['strain_name', 'strain_id', stat_col]]


def sel_rank_by(strain_stats_df):
    "Prompt user for column for ranking; return its index"
    cols = strain_stats_df.columns[2:]
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
    """Add rolling average column to StrainSalesDF.strain_df object"""
    boxcar = window_wks * 7
    col = 'rolling_{}wk'.format(window_wks)
    df[col] = df[data_col].rolling(window=boxcar).mean()



if __name__=='__main__':

    """Set input variables"""
    strains = range(1, 10) # list of strain IDs
    MAs = [5] # list of moving average window(s) in weeks
    sample_period = 20 # in weeks

    """Run StrainSalesDF method and access class attributes"""
    strain_3 = StrainSalesDF(3)
    strain_3.main()
    raw_df_3 = strain_3.strain_df # DataFrame of daily sales and units for strain
    sales_3 = strain_3.sales # time series (pd.Series) of daily sales
    units_3 = strain_3.units_sold # time Series of daily units sold

    """Run StrainTrendsDF method and access class attributes"""
    trends_3 = StrainTrendsDF(sales_3, sample_period, MA_params=MAs)
    trends_3.main()
    trends_df_3 = trends_3._trendsDF # DataFrame with columns of transformed data
    stats_3 = trends_3.trend_stats # Single record (OrderedDict) of stats for strain

    """Run StrainStatsDF function to generate comparative stats DF; Builds DF from
    individual records in the form of StrainTrendsDF.trend_stats objects"""
    comps_df = StrainStatsDF(strains, sample_period, MA_params=MAs)

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
