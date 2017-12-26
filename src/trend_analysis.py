"""
Utilities for trend analysis

CLASSES:
 -- StrainSalesDF(strain_id)
 -- StrainTrendsDF(ts, period_wks, end_date=None, RA_params=None,
                   exp_smooth_params=None, normed=True)

FUNCTIONS
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
    Initialize with strain_id (int) then run construct_df() method to populate
    attributes

    ATTRIBUTES:
     -- strain_df: pandas time series (DataFrame) with daily sales in dollars and units
     -- sales: pandas time series (Series) of total daily sales
     -- units_sold: pandas time series (Series) of total daily units sold
     -- strain_id (int)
     -- strain_name (string)

    NOTE: DataFrame and Series titles with strain name and ID contained in
          df.name and Series.name attributes
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

    def construct_df(self):
        self._query_strain_sales()
        self._connect_to_postgres()
        self._SQL2pandasdf()

    def _query_strain_sales(self):
        """Enter strain id (int); returns query for strain's daily sales"""
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
    trends for both absolute and normalized values.

    INPUT:
     -- ts: StrainSalesDF.sales or .units_sold object, time series (pandas Series)
     -- period_wks: date span in weeks reaching back from most recent datum,
        used to define sampling period (int)
     -- end_date: optional, date string (e.g., '07/15/2016') other than most recent
        datum before which to extend sampling period (str, default=None)
     -- RA_params: one or more rolling "boxcar" windows, in weeks, by which
        to generate distinct columns of rolling average data (list of ints, default=None)
     -- exp_smooth_params: one or more alpha smoothing factors (0 < alpha < 1)
        by which to generate distinct columns of exponentially smoothed columns
        (list of floats, default=None)
     -- normed: (default = True) add a column for each rolling average or exp
        smoothed column that computes on data rescaled (-1, 1) and then shifted
        such that datum at t0 = 0.

    ATTRIBUTES:
     -- trendsDF: (pandas DataFrame)
     -- trend_stats: (dict)

    METHODS:
     -- main(): run after initialization to populate trendsDF
     -- aggregate_stats(): populates trend_stats containing record for strain
        aggregated from trendsDF object
     -- norm_Series(col): rescales (-1, 1) and shifts selected data column
     -- trend_AUC(ts): computes area under curve for time series
     -- compute_aggr_slope(ts): returns slope of line describing avg growth rate
        over selected time series data
    """

    def __init__(self, ts, period_wks, end_date=None, RA_params=None,
                    exp_smooth_params=None, normed=True):
        self.ts = ts
        self.raw_df = None
        self.period_wks = period_wks
        self._period_days = period_wks * 7
        self.end_date = end_date
        self.RA_params = RA_params
        self.exp_smooth_params = exp_smooth_params
        self.normed = normed
        self.strain_name = self.ts.name.split('(')[0].strip()
        self.strain_ID = int(self.ts.name.split(')')[0].split(' ')[-1])
        self.ts_sample = None
        self.trendsDF = None
        self.trend_stats = {}


    def main(self):
        self._constuct_basic_trendsDF()
        if self.RA_params:
            self._compute_rolling_averages()
        if self.exp_smooth_params:
            self._compute_exp_smoothed_trends()
        self.aggregate_stats()


    def _constuct_basic_trendsDF(self):
        """DF with sales over period"""
        self._slice_timeseries()
        sales_col_name = self.ts.name.split(' -- ')[-1].lower()
        self.trendsDF = pd.DataFrame(data=self.ts_sample.values,
                                    columns=[sales_col_name],
                                    index=self.ts_sample.index
                                    )
        self.trendsDF.name = self._trendsDF_name()


    def _compute_rolling_averages(self):
        self.raw_df = pd.DataFrame(self.ts)
        for wk_window in self.RA_params:
            boxcar = wk_window * 7
            col_name = '{}WK RA'.format(wk_window)
            self.raw_df[col_name] = self.ts.rolling(window=boxcar).mean()
            self.trendsDF[col_name] = self.raw_df[col_name][self.trendsDF.index]
            if self.normed:
                # DEBUG THIS! NOT NORMING THE RIGHT THING
                normed_col_name = '{}WK RA normd'.format(wk_window)
                self.trendsDF[normed_col_name] = self.norm_Series(self.trendsDF[col_name])


    def aggregate_stats(self):
        """Construct trend_stats from trendsDF"""
        # DEBUG THIS! object is not coming out in proper order; can't import to df
        self.trend_stats = {'strain_name': self.strain_name,
                            'strain_ID': self.strain_ID,
                            'ttl_sales': None
                                        }

    def _slice_timeseries(self):
        """Construct ts_sample attribute"""
        if self.end_date:
            self.ts_sample = self.ts[self.end_date - self._period_days:self.end_date]
        else:
            self.ts_sample = self.ts[-self._period_days:]


    def _trendsDF_name(self):
        """Construct string for trendsDF pandas DataFrame name attribute"""
        sales_or_units = self.ts.name.split(' -- ')[-1]
        if not self.end_date:
            ending = self.ts.index[-1].strftime('%m/%d/%Y')
        else:
            ending = self.end_date

        DF_name = ('{} (ID: {}) Trends in {} over {} Weeks Ending {}').format(
            self.strain_name,
            self.strain_ID,
            sales_or_units,
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

    def trend_AUC(self, ts):
        """Compute trend AUC for column in trendDF
        """
        return np.trapz(ts.values)

    def compute_aggr_slope(self, ts):
        """Redistribute AUC under straight line and return slope of line
        in units of avg sales or units sold gained/lost per week"""
        AUC = self.trend_AUC(ts)
        return (14 * AUC) / (len(ts)**2)









def compute_rolling_avg(ts, window_wks):
    """
    INPUT: complete time series (Series) and moving 'boxcar' window in weeks
    OUTPUT: rolling average values
    """
    boxcar = window_wks * 7
    return ts.rolling(window=boxcar).mean()

def smooth_exponentially(ts, alpha):
    """Apply exponential smoothing to sliced time series"""
    pass


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

def strain_trendsDF():
    pass

def strain_stats():
    pass

def strain_rankDF():
    pass



if __name__=='__main__':
    pass
