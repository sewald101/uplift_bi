import numpy as np
import pandas as pd


class TimeSeriesData(object):
    """Takes in generic-strain csv from product_skus table; converts to time-series dataframe
    INPUT:
     - input_file: (string) path to csv source file

    METHODS:
     - construct(): run after initialization to construct time-series dataframe
         and other class attributes

    ATTRIBUTES:
     - time_series: dataframe containing daily total sales and gross profits
         for generic strain over total time span of available inventory_id records
     - by_inv_id: dataframe recording dates and statistics by inventory_id record
     - raw_df: input_file converted to dataframe
    """

    def __init__(self, input_file):
        """input_file = path to csv source file (string)"""
        self.input_file = input_file
        self.raw_df = pd.read_csv(self.input_file, parse_dates=[2, 3])
        self.by_inv_id = None
        self.time_series = None
        self._date_idx = None
        self._bool_matrix = None
        self._sales_matrix = None


    def construct_tables(self):
        """Run to construct all public attribute dataframes"""
        self._construct_by_inv_id()
        self._construct_date_idx()
        self._construct_bool_matrix()
        self._construct_sales_matrix()
        self._construct_time_series()


    def _construct_by_inv_id(self):
        """Construct by_inv_id dataframe from raw_df"""
        self.by_inv_id = self.raw_df.copy()
        self.by_inv_id['days_sales'] = (
            self.by_inv_id['latest_rtl_sale'] - self.by_inv_id['first_rtl_sale']
            )
        # Reassign days_sales timedelta zeros to ones for division in calculating
        # daily averages
        x = '1 days'
        self.by_inv_id.loc[
            self.by_inv_id['days_sales'].apply(lambda x: x.days) == 0,
            'days_sales'
            ] = pd.to_timedelta(x)

        self.by_inv_id['gross_profit'] = (
            self.by_inv_id['ttl_rtl_sales'] - self.by_inv_id['wholesale_cogs']
            )
        self.by_inv_id['avg_daily_sales'] = (
            self.by_inv_id['ttl_rtl_sales'] /
            self.by_inv_id['days_sales'].apply(lambda x: x.days)
            )
        self.by_inv_id['avg_daily_gross'] = (
            self.by_inv_id['gross_profit'] /
            self.by_inv_id['days_sales'].apply(lambda x: x.days)
            )
        cols = ['wa_inventory_id', 'generic_strain_id', 'first_rtl_sale',
                'latest_rtl_sale', 'days_sales', 'wholesale_cogs',
                'ttl_rtl_sales', 'gross_profit', 'avg_daily_sales',
                'avg_daily_gross', 'units_sold'
                ]
        self.by_inv_id = self.by_inv_id[cols]


    def _construct_date_idx(self):
        """Construct index of consecutive dates from by_inv_id dataframe"""
        self._date_idx = pd.date_range(
            self.by_inv_id['first_rtl_sale'].min(),
            self.by_inv_id['latest_rtl_sale'].max()
        )


    def _construct_bool_matrix(self):
        """Construct boolean transformation matrix on: date IN date range per inventory id"""
        # dates_idx = self._date_idx
        A = self.by_inv_id['first_rtl_sale'].as_matrix()
        B = self.by_inv_id['latest_rtl_sale'].as_matrix()
        first_rtl_sales, date_range = np.meshgrid(A, self._date_idx)
        lastsale_dates, date_range = np.meshgrid(B, self._date_idx)
        bool_A = date_range >= first_rtl_sales
        bool_B = date_range <= lastsale_dates
        matrix_data = bool_A * bool_B

        self._bool_matrix = pd.DataFrame(matrix_data)


    def _construct_sales_matrix(self):
        """Construct matrix of avg daily sales and margin by inventory_id,
        to be dot-multiplied with _bool_matrix"""
        cols = ['avg_daily_sales', 'avg_daily_gross']
        self._sales_matrix = pd.DataFrame(
            self.by_inv_id[cols].values.reshape(
                               len(self.by_inv_id['wa_inventory_id']), 2
                               ))


    def _construct_time_series(self):
        cols = ['ttl_sales', 'ttl_gross_profit']
        self.time_series = self._bool_matrix.dot(self._sales_matrix)
        self.time_series.columns = cols
        self.time_series.index = self._date_idx
