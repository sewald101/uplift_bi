import numpy as np
from numpy import trapz
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2 as ps2


class StrainSalesDF(object):
    """
    Initialize with strain_id (int)
    Run public method construct_df()
    Returns pandas dataframe for strain as attribute strain_df
    """

    def __init__(self, strain_id):
        self.strain_id = strain_id
        self._query = None
        self._connection_str = 'postgresql:///uplift'
        self._conn = None
        self.strain_df = None

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
        self.strain_df.index = raw_df['date']














if __name__=='__main__':
    pass
