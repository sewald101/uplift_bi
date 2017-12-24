"""
Utilities for trend analysis

CLASSES:
 -- StrainSalesDF(strain_id)

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
