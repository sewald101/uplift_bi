"""
Utilities for trend analysis

CLASSES:
 -- StrainSalesDF(strain_id)

FUNCTIONS
 -- compute_rolling_avg(df, window_wks, data_col='ttl_sales')
 -- slice_timeseries(data, period_wks, end_date=None)
 -- trend_AUC(data, normalize=False, normed_Series=False)
 -- add_rolling_avg_col(df, window_wks, data_col='ttl_sales')

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


class StrainSalesDF(object):
    """
    Initialize with strain_id (int) then run public method construct_df()
    ATTRIBUTES:
     -- strain_df: pandas time series (DataFrame) with daily sales in dollars and units
     -- sales: pandas time series (Series) of total daily sales
     -- units_sold: pandas time series (Series) of total daily units sold
    """

    def __init__(self, strain_id):
        self.strain_id = strain_id
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
         , st.generic_strain_id as strain_id
         , ROUND(SUM(ds.retail_price)) as ttl_sales
         , ROUND(SUM(ds.retail_units)) as ttl_units_sold
        FROM daily_retail_sales ds
        JOIN strains st
        ON ds.strain_name = st.strain_display_name
        WHERE st.generic_strain_id = {}
        GROUP BY date, strain_id
        ORDER BY date;
        """).format(self.strain_id)

    def _connect_to_postgres(self):
        self._conn = create_engine(self._connection_str)

    def _SQL2pandasdf(self):
        raw_df = pd.read_sql_query(self._query, self._conn)
        self.strain_df = pd.DataFrame(raw_df[['strain_id',
                                            'ttl_sales',
                                            'ttl_units_sold']
                                            ])
        self.strain_df.index = pd.DatetimeIndex(raw_df['date'])
        self.sales = self.strain_df['ttl_sales']
        self.units_sold = self.strain_df['ttl_units_sold']



def compute_rolling_avg(ts, window_wks):
    """INPUT: time series (Series) and moving window in weeks
    OUTPUT: rolling average values"""
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


def trend_AUC(ts, normalize=False, normed_Series=False):
    """
    INPUT: trend data in time series (pandas.Series)
    OUTPUT:
     -- default: area under curve (AUC) for shifted trend data
     -- normalize=True, normed_Series=False: AUC for normed then shifted trend data
     -- normalize=True, normed_Series=True: pandas Series for normed then shifted data
    NOTE: Data shifted such that value at t0 = 0; as a consequence, some normed
    values may exceed the feature range (-1, 1)
    """
    if normalize:
        values = ts.values
        values = values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(values)
        normed_trend = scaler.transform(values).flatten()
        normed_trend = pd.Series(normed_trend - normed_trend[0], index=ts.index)
        if normed_Series:
            return normed_trend
        else:
            return np.trapz(normed_trend.values)
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
    pass
