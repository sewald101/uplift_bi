import numpy as np
import pandas as pd


class TimeSeriesData(object):
    """Takes in generic-strain csv from product_skus table; converts to time-series dataframe
    INPUT:
     - input_file: (string) path to csv source file

    METHODS:
     - construct_tables(): run after initialization to construct time-series dataframe
         and other class attributes

    ATTRIBUTES:
     - time_series: DataFrame containing daily total sales and gross profits
         for generic strain over total time span of available inventory_id records
     - sales_daily: Pandas Series
     - sales_weekly: Pandas Series, by Monday-week
     - gross_daily: Pandas Series
     - gross_weekly: Pandas Series, by Monday-week
     - by_inv_id: dataframe recording dates and statistics by inventory_id record
     - raw_df: input_file converted to dataframe
    """

    def __init__(self, input_file):
        """input_file = path to csv source file (string)"""
        self.input_file = input_file
        self.raw_df = pd.read_csv(self.input_file, parse_dates=[2, 3])
        self.by_inv_id = None
        self.time_series = None
        self.sales_daily = None
        self.sales_weekly = None
        self.gross_daily = None
        self.gross_weekly = None
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
        A = self.by_inv_id['first_rtl_sale'].as_matrix()
        B = self.by_inv_id['latest_rtl_sale'].as_matrix()
        date_tiled = np.broadcast_to(
                        np.array(self._date_idx).reshape(len(self._date_idx), 1),
                        (len(self._date_idx), len(self.by_inv_id.index))
                        )
        bool_A = date_tiled >= A
        bool_B = date_tiled <= B
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

        self.sales_daily = pd.Series(self.time_series['ttl_sales'],
                                    index=self._date_idx)
        self.sales_weekly = self.sales_daily.resample('W-MON').sum()
        self.gross_daily = pd.Series(self.time_series['ttl_gross_profit'],
                                    index=self._date_idx)
        self.gross_weekly = self.gross_daily.resample('W-MON').sum()


if __name__=='__main__':
    path = '../data/lemon_haze_18.csv'

    lemon_haze = TimeSeriesData(path)
    lemon_haze.construct_tables()

    raw = lemon_haze.raw_df
    inv = lemon_haze.by_inv_id
    ts = lemon_haze.time_series
    sales_dy = lemon_haze.sales_daily
    sales_wk = lemon_haze.sales_weekly
    gross_dy = lemon_haze.gross_daily
    gross_wk = lemon_haze.gross_weekly

    print(ts.info())
