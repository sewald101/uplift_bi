"""
UTILITIES FOR SALES TREND ANALYSIS

CLASSES:
 -- ImportSalesData(product_id)
 -- SalesTrendsDF(ts, period_wks, end_date=None, MA_params=None,
                   exp_smooth_params=None, normed=True)
 -- RankProductsPlacesPlaces(product_stats_df, N_results=None)

MAJOR FUNCTIONS:
 -- SalesStatsDF(product_IDs, period_wks, end_date=None, MA_params=None,
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

from id_dict import (strain_dict, names_formatted, locations_dict,
                     product_name_from_ID, locations_name_from_ID)


class ImportSalesData(object):
    """
    Query weekly aggregated sales data from postgres and import to pandas data objects.
    Initialize with any one of the following filter options:

    A. ONE product ID (int) or product name (string)
    B. COMBINATION OF ONE product (ID or name) AND NO MORE THAN ONE location (ID or name)
        OR city (str) OR zipcode (5-digit int)
    C. OMIT product AND SPECIFY NO MORE THAN ONE location OR city OR zipcode
    D. DEFAULT (all filters = None), imports sales data aggregated statewide for all products

    To retain data at original weekly frequency (versus daily), initialize with
       upsample=False

    Then run main() method to populate attributes.

    ATTRIBUTES:
     -- product_df: pandas time series (DataFrame) with daily sales in dollars and units
     -- sales: pandas time series (Series) of total daily sales
     -- units_sold: pandas time series (Series) of total daily units sold
     -- product_id (int or None)
     -- product_name (string or None)
     -- location_id (int or None)
     -- location_name (string or None)
     -- ts_start, ts_end (Datetime) start and end dates for time series, to assist
           testing for continuity and synchronization among comparative products

    NOTE: DataFrame and Series title strings with product name and ID may be accessed
        via DataFrame.name and Series.name attributes
    """

    def __init__(self, product=None, location=None, city=None, zipcode=None, upsample=True):
        self.product = product
        self.location = location
        self.city = city
        self.zipcode = zipcode
        self.upsample = upsample
        self._connection_str = 'postgresql:///uplift'

        # populated via main() method
        self.product_id = None
        self.product_name = None
        self.location_id = None
        self.location_name = None
        self._query = None
        self._conn = None
        self.product_df = None
        self.sales = None
        self.units_sold = None
        self.ts_start, self.ts_end = None, None

        # elements for SQL query and for pd.DataFrame and pd.Series names
        self.toggles = {'toggle_1': '--', 'filter_1': '', 'value_1': '',
                        'filter_2': '', 'value_2': '',
                        'toggle_2': '--'}
        self.city_formatted = (
            "\'" + self.city.upper() + "\'" if self.city is not None else None
            )
        self.df_name = None


    def main(self):
        self._retrieve_IDs()
        self._set_query_toggles()
        self._compose_df_name()
        self._query_sales()
        self._connect_to_postgres()
        self._SQL2pandasdf()

    def _retrieve_IDs(self):
        if self.product is not None:
            if type(self.product) == str:
                key = self.product.lower()
                try:
                    self.product_id = strain_dict[key]
                except KeyError:
                    return
                else:
                    self.product_name = names_formatted[self.product.lower()]
            else:
                self.product_id = self.product
                try:
                    self.product_name = names_formatted[product_name_from_ID(self.product)]
                except KeyError:
                    return

        if self.location is not None:
            if type(self.location) == str:
                self.location_name = self.location.upper()
                key = self.location_name
                try:
                    self.location_id = locations_dict[key]
                except KeyError:
                    return
            else:
                self.location_name = locations_name_from_ID(self.location)
                self.location_id = self.location

    def _set_query_toggles(self):
        """Convert initialization parameters into query terms placed in a dictionary"""
        arg_list = [self.product_id, self.location_id, self.city_formatted, self.zipcode]
        fields = ['strain_id', 'location_id', 'city', 'zip']

        mask = lambda x: x is not None
        bool_list = map(mask, arg_list)
        if bool_list.count(True) != 0:
            idx_1 = bool_list.index(True)

        if bool_list.count(True) == 1: # if only one filter specified...
            self.toggles['toggle_1'] = ''
            self.toggles['filter_1'] = fields[idx_1]
            self.toggles['value_1'] = arg_list[idx_1]

        if bool_list.count(True) == 2: # if two filters specified...
            self.toggles['toggle_1'] = ''
            self.toggles['filter_1'] = fields[0] # Toggle on product filter
            self.toggles['value_1'] = arg_list[0]
            idx_2 = bool_list[1:].index(True) + 1 # Grab index of second filter
            self.toggles['toggle_2'] = ''
            self.toggles['filter_2'] = fields[idx_2]
            self.toggles['value_2'] = arg_list[idx_2]

    def _compose_df_name(self):
        # Statewide data for all products
        if self.toggles['toggle_1'] == '--':
            self.df_name = 'All Cannabis Products, Statewide'

        # If only ONE product or geographic filter is specified
        if self.toggles['toggle_1'] == '' and self.toggles['toggle_2'] == '--':
            if self.toggles['filter_1'] == 'strain_id':
                self.df_name = '{} (ID: {}) Statewide'.format(self.product_name,
                                                              self.product_id)
            elif self.toggles['filter_1'] == 'location_id':
                self.df_name = 'Location: {} (ID: {})'.format(self.location_name,
                                                              self.location_id)
            elif self.toggles['filter_1'] == 'city':
                self.df_name = 'City: {}'.format(self.city.upper())
            else:
                self.df_name = 'Zipcode: {}'.format(self.zipcode)

        # If one product and one geographic filter are specified
        if self.toggles['toggle_1'] == '' and self.toggles['toggle_2'] == '':
            A = '{} (ID: {})'.format(self.product_name, self.product_id)
            if self.toggles['filter_2'] == 'location_id':
                B = ', Location: {} (ID: {})'.format(self.location_name, self.location_id)
            elif self.toggles['filter_2'] == 'city':
                B = ', City: {}'.format(self.city.upper())
            else:
                B = ', Zipcode: {}'.format(self.zipcode)
            self.df_name = A + B


    def _query_sales(self):
        self._query = ("""
        SELECT CAST(DATE_TRUNC('day', week_beginning) AS DATE) as week_beg
         , ROUND(SUM(retail_price)) as ttl_sales
         , ROUND(SUM(retail_units)) as ttl_units_sold
        FROM weekly_sales
        {0:}WHERE {1:} = {2:}
        {3:}AND {4:} = {5:}
        GROUP BY week_beg
        ORDER BY week_beg;
        """).format(self.toggles['toggle_1'],
                    self.toggles['filter_1'],
                    self.toggles['value_1'],
                    self.toggles['toggle_2'],
                    self.toggles['filter_2'],
                    self.toggles['value_2']
                    )

    def _connect_to_postgres(self):
        self._conn = create_engine(self._connection_str)

    def _SQL2pandasdf(self):
        stage_1 = pd.read_sql_query(self._query, self._conn)
        stage_2 = pd.DataFrame(stage_1[['ttl_sales', 'ttl_units_sold']])
        stage_2.index = pd.DatetimeIndex(stage_1['week_beg'])

        try:
            # Construct continuous time series even if data is discontinuous
            self.ts_start, self.ts_end = stage_2.index[0], stage_2.index[-1]
        except IndexError:
            print "\nERROR: NO SALES DATA FOR THE FILTERS SPECIFIED"
            return None
        else:
            if self.upsample: # Construct daily DF from weekly aggregated data
                main_idx = pd.date_range(start=self.ts_start, end=self.ts_end, freq='W-MON')
                stage_3 = pd.DataFrame(index=main_idx)

                stage_3['ttl_sales'] = stage_2['ttl_sales']
                stage_3['ttl_units_sold'] = stage_2['ttl_units_sold']

                stage_4 = stage_3 / 7 # convert weekly values to daily values
                stage_5 = stage_4.resample('D').asfreq() # upsample from weeks to days
                extended_idx = pd.DatetimeIndex(
                    start=stage_5.index[0], end=stage_5.index[-1] + 6, freq='D'
                    ) # extend upsampled index to include days from last week of data
                self.product_df = stage_5.reindex(extended_idx).ffill(limit=6)
                # forward-fill available daily values, retaining NaNs over those weeks
                # where NaNs appear in original weekly data

                self.product_df.name = self.df_name

                self.sales = self.product_df['ttl_sales']
                self.sales.name = self.df_name + ' -- Daily Sales'

                self.units_sold = self.product_df['ttl_units_sold']
                self.units_sold.name = self.df_name + ' -- Daily Units Sold'

            else: # preserve data in original weekly aggregated form
                main_idx = pd.date_range(start=self.ts_start, end=self.ts_end, freq='W-MON')
                self.product_df = pd.DataFrame(index=main_idx)

                self.product_df['ttl_sales'] = stage_2['ttl_sales']
                self.product_df['ttl_units_sold'] = stage_2['ttl_units_sold']

                self.product_df.name = self.df_name

                self.sales = self.product_df['ttl_sales']
                self.sales.name = self.df_name + ' -- Weekly Sales'

                self.units_sold = self.product_df['ttl_units_sold']
                self.units_sold.name = self.df_name + ' -- Weekly Units Sold'


class SalesTrendsDF(object):
    """Convert raw time series sales or unit-sales data for a single product into
    engineered trend data, including rolling averages and exponentially smoothed
    trends for both absolute and normalized (rescaled) values.

    INPUT:
     -- ts: ImportSalesData.sales or .units_sold object (pandas Series)
     -- period_wks: (int) date span of sampling period in weeks measured back
          from most recent datum or from user-supplied end_date
     -- end_date: (date string of form: '07/15/2016', default=None) alternative
          end-date for sampling period; default uses most recent datum
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
     -- NaN_filler: (float or None, default=0.0) fillna value for raw data, allows marking
          converted NaNs by using tag value such as 0.0001. Set to None to generate
          a trendsDF with only raw sample data, NaNs in place. Note: All computations
          and statistical aggregation on smoothed trends use fillna = 0.0.


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
                    exp_smooth_params=None, normed=True, baseline='t_zero',
                    NaN_filler=0.0):
        self.ts = ts
        self.raw_df = None
        self.period_wks = period_wks
        self._period_days = period_wks * 7
        self.end_date = end_date
        self.MA_params = MA_params
        self.exp_smooth_params = exp_smooth_params
        self.normed = normed
        self.baseline = baseline
        self.NaN_filler = NaN_filler
        self.product_name = None
        self.product_ID = None
        self.place_name = None
        self.place_ID = None
        self.ts_sample = None
        self.trendsDF = None
        self.trend_stats = OrderedDict()
        self.NaNs_ratio = None
        try:
            self.sales_col_name = self.ts.name.split(' -- Daily ')[-1]
        except AttributeError:
            pass

    def main(self):
        if self.ts is not None:
            self._constuct_basic_trendsDF()

            if self.NaN_filler is not None:
                if self.MA_params:
                    self._compute_rolling_averages()

            self.aggregate_stats()
        else:
            print "\nERROR: Initialization ts is NoneType."

    def _constuct_basic_trendsDF(self):
        """DF with sales over period"""
        # populate ts_sample attribute
        self._slice_timeseries()
        # compute ratio of NaNs in ts_sample
        self.NaNs_ratio = (
            self.ts_sample.isnull().sum() / float(len(self.ts_sample))
            )
        # contruct base trendsDF object
        self.trendsDF = pd.DataFrame(
                    data=self.ts_sample.values,
                    columns=[self.sales_col_name.lower()],
                    index=self.ts_sample.index
                    )

        if self.NaN_filler is not None:
            self.trendsDF.fillna(self.NaN_filler, inplace=True)

        self.trendsDF.name = self._trendsDF_name()

        # Only add columns for shifted and normed trends on NaN-filled data
        if self.NaN_filler is not None:
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
                self.ts.fillna(0).rolling(window=boxcar).mean()
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
        self.trend_stats['place_name'] = self.place_name
        self.trend_stats['place_id'] = self.place_ID
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
        offset = pd.DateOffset(self._period_days - 1)
        if self.end_date:
            idx_end = pd.to_datetime(self.end_date)
            sample_idx = pd.date_range(
                start=idx_end - offset, end=idx_end
                )
            sample_df = pd.DataFrame(index=sample_idx)
            sample_df['vals'] = self.ts[sample_idx]
            self.ts_sample = sample_df.iloc[:,0]
            self.ts_sample.name = self.ts.name
        else: # else use most recent date available
            idx_end = self.ts.index[-1]
            sample_idx = pd.date_range(
                start=idx_end - offset, end=idx_end
                )
            sample_df = pd.DataFrame(index=sample_idx)
            sample_df['vals'] = self.ts[sample_idx]
            self.ts_sample = sample_df.iloc[:,0]
            self.ts_sample.name = self.ts.name


    def _trendsDF_name(self):
        """Construct string for trendsDF pandas DataFrame name attribute"""
        A = ''
        B = 'Statewide '

        first_parse = self.ts.name.split(' -- Daily ')
        if 'ID' in first_parse[0] and 'Location' not in first_parse[0]:
            self.product_name = first_parse[0].split(' (ID: ')[0]
            self.product_ID = int(first_parse[0].split(' (ID: ')[1].split(')')[0])
            A = '{} (ID: {}), '.format(self.product_name, self.product_ID)

        if 'Location' in first_parse[0] and first_parse[0].count('ID') == 1:
            self.place_name = first_parse[0].split(': ')[-2][:-4]
            self.place_ID = int(
                first_parse[0].split(' (ID: ')[-1].split(')')[0]
                )
            B = 'Location: {} (ID: {}), '.format(self.place_name, self.place_ID)

        if first_parse[0].count('ID') == 2:
            self.product_name = first_parse[0].split(' (ID: ')[0]
            self.product_ID = int(first_parse[0].split(' (ID: ')[1].split(')')[0])
            A = '{} (ID: {}), '.format(self.product_name, self.product_ID)
            self.place_name = first_parse[0].split(': ')[-2][:-4]
            self.place_ID = int(
                first_parse[0].split(' (ID: ')[-1].split(')')[0]
                )
            B = 'Location: {} (ID: {}), '.format(self.place_name, self.place_ID)

        if 'City' in first_parse[0]:
            self.place_name = first_parse[0].split('City: ')[-1]
            B = 'City: {}, '.format(self.place_name)

        if 'Zipcode' in first_parse[0]:
            self.place_name = first_parse[0].split('Zipcode: ')[-1]
            B = 'Zipcode: {}, '.format(self.place_name)

        if not self.end_date:
            ending = self.ts.index[-1].strftime('%m/%d/%Y')
        else:
            ending = self.end_date

        DF_name = (A + B + 'Trends in {} over {} Weeks Ending {}').format(
                                                        self.sales_col_name,
                                                        self.period_wks,
                                                        ending
                                                        )
        return DF_name


    def norm_Series(self, col):
        """Return time series rescaled then shifted to baseline.
        """
        values = col.fillna(0).values
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
                return -1 * np.log(-1 * np.trapz(ts.fillna(0)))
            else:
                return np.log(np.trapz(ts.fillna(0)))

        elif sqrt_scaled:
            if np.trapz(ts.values) < 0:
                return -1 * np.sqrt(-1 * np.trapz(ts.fillna(0)))
            else:
                return np.sqrt(np.trapz(ts.fillna(0)))

        else:
            return np.trapz(ts.fillna(0))


    def compute_aggr_slope(self, ts):
        """Redistribute AUC under straight line and return slope of line. For
        raw figures, units represent avg sales (or units sold) gained/lost per day"""
        AUC = self.trend_AUC(ts)
        return (2 * AUC) / (len(ts)**2)



def SalesStatsDF(period_wks, end_date, products=[None], locations=[None],
                   cities=[None], zipcodes=[None], MA_params=[5], normed=True,
                   baseline='t_zero', compute_on_sales=True, NaN_allowance=5,
                   print_rejects=False, return_rejects=False):
    """Construct DataFrame showing comparative sales stats among multiple products
    or places. See output DataFrame.name attribute for title.

    ARGUMENTS:
     -- period_wks: (int) sampling period in weeks
     -- end_date: (date string of form: '07/15/2016') date string defining
          end of sampling period for comparison

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

     OPTIONAL:
     -- MA_params: (list of ints, default=5) one or more rolling "boxcar"
          windows, in weeks, by which to compute moving averages
     -- normed: (bool, default=True) add a column for each rolling average or expon.
          smoothed column that computes on data that has been rescaled (-1, 1)
          and then shifted to baseline.
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data by the mean
          * 'median' -- shift data by the median
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
     -- NaN_allowance: (int or float from 0 to 100, default=5) max allowable
          percentage of NaNs in product ts samples for statistical aggregation;
          products exceeding allowance are discarded from output DataFrame and
          reported in rejection dictionary
     -- print_rejects: (bool, default=False) If True, print any products rejected
          for excess null values in sample with their corresponding ratio of nulls
          present in the dataset
     -- return_rejects: (bool, default=False) If True, returns dictionary of
          of products rejected for excess nulls along with main output dataframe.
    """

    data = []
    rejected = {}
    counter = 0
    df_name = None

    product_place_args = [products, locations, cities, zipcodes]
    import_type, var_index = select_import_params(product_place_args)

    if import_type == 'E':
        print (
        '\nERROR: CONFLICTING VALUES ENTERED AMONG PRODUCTS, LOCATIONS, CITIES, '
        'AND/OR ZIPCODES ARGUMENTS.\n'
        'ONLY ONE OF THOSE FOUR LIST-ARGUMENTS MAY CONTAIN MORE THAN ONE VALUE.\n'
             )
        return

    if import_type == 'A': # Statewide data for all products
        stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
             MA_params=MA_params, normed=normed, baseline=baseline,
             compute_on_sales=compute_on_sales, NaN_allowance=NaN_allowance)

        data.append(stats)
        df_name = name

    if import_type == 'B': # Single product or place specified
        stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
            product=products[0], location=locations[0], city=cities[0],
            zipcode=zipcodes[0], MA_params=MA_params, normed=normed, baseline=baseline,
            compute_on_sales=compute_on_sales, NaN_allowance=NaN_allowance)

        if stats is not None:
            # If null vals in sample exceed allowance threshold, dump product or place
            # into rejected dict and exclude from output DF
            if NaN_ratio > NaN_allowance / 100.:
                if products[0] is not None:
                    rejected[stats['product_name']] = NaN_ratio
                else:
                    rejected[stats['place_name']] = NaN_ratio
            else:
                data.append(stats)
                df_name = name
        else:
            return

    if import_type == 'C': # Iterate on products
        for prod in products:
            stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
                product=prod, location=locations[0], city=cities[0],
                zipcode=zipcodes[0], MA_params=MA_params, normed=normed, baseline=baseline,
                compute_on_sales=compute_on_sales, NaN_allowance=NaN_allowance)

            if stats is None:
                continue
            else:
                if NaN_ratio > NaN_allowance / 100.:
                    rejected[stats['product_name']] = NaN_ratio
                else:
                    data.append(stats)

                if counter < 1: # first loop, grab name for output DF
                    if locations[0] is not None:
                        df_name = ('Product Comparison, '
                                   + name.split(', ')[-1]
                                   + ', Business: {}'.format(locations[0].upper())
                                   )
                    elif cities[0] is not None:
                        df_name = ('Product Comparison, '
                                   + name.split(', ')[-1]
                                   + ', City: {}'.format(cities[0].upper())
                                   )
                    elif zipcodes[0] is not None:
                        df_name = ('Product Comparison, '
                                   + name.split(', ')[-1]
                                   + ', Zipcode: {}'.format(zipcodes[0])
                                   )
                    else:
                        df_name = 'Product Comparison, ' + name.split(', ')[-1]
                counter += 1

    if import_type == 'D': # iterate on a place
        if var_index == 1:
            for loc in locations:
                stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
                    product=products[0], location=loc, MA_params=MA_params,
                    normed=normed, baseline=baseline,
                    compute_on_sales=compute_on_sales, NaN_allowance=NaN_allowance)
                if stats is None:
                    continue
                else:
                    if NaN_ratio > NaN_allowance / 100.:
                        rejected[stats['place_name']] = NaN_ratio
                    else:
                        data.append(stats)

                    if counter < 1:
                        if products[0] is None:
                            df_name = 'Comparison by Business, ' + name.split(', ')[-1]
                        else:
                            df_name = (name.split(', ')[0]
                                       + ', Comparison by Business, '
                                       + name.split(', ')[-1]
                                       )
                    counter += 1

        if var_index == 2:
            for city in cities:
                stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
                    product=products[0], city=city, MA_params=MA_params, normed=normed,
                    baseline=baseline, compute_on_sales=compute_on_sales,
                    NaN_allowance=NaN_allowance)
                if stats is None:
                    continue
                else:
                    if NaN_ratio > NaN_allowance / 100.:
                        rejected[stats['place_name']] = NaN_ratio
                    else:
                        data.append(stats)

                    if counter < 1:
                        if products[0] is None:
                            df_name = 'Comparison by City, ' + name.split(', ')[-1]
                        else:
                            df_name = (name.split(', ')[0]
                                       + ', Comparison by City, '
                                       + name.split(', ')[-1]
                                       )
                    counter += 1

        if var_index == 3:
            for zipcode in zipcodes:
                stats, NaN_ratio, name = import_ala_params(period_wks, end_date,
                    product=products[0], zipcode=zipcode, MA_params=MA_params,
                    normed=normed, baseline=baseline,
                    compute_on_sales=compute_on_sales, NaN_allowance=NaN_allowance)
                if stats is None:
                    continue
                else:
                    if NaN_ratio > NaN_allowance / 100.:
                        rejected[stats['place_name']] = NaN_ratio
                    else:
                        data.append(stats)

                    if counter < 1:
                        if products[0] is None:
                            df_name = 'Comparison by Zipcode, ' + name.split(', ')[-1]
                        else:
                            df_name = (name.split(', ')[0]
                                       + ', Comparison by Zipcode, '
                                       + name.split(', ')[-1]
                                       )
                    counter += 1

    try:
        product_stats_df = pd.DataFrame(data, columns=data[0].keys())
    except IndexError:
        print ('\nNO DATA AVAILABLE IN SPECIFIED PERIOD FOR PRODUCT AND/OR PLACE.\n'
               'Utilize PlotRawData function to view data availability over time.\n'
               )
    else:
        product_stats_df.name = df_name

        if print_rejects:
            if len(rejected) > 0:
                print('Data for the following product(s) and/or place(s) exceed allowance for '
                      'null values \nand are excluded from statistical aggregation '
                      'and/or ranking:\n')
                for k, v in rejected.iteritems():
                    print('{} -- Percent Null: {}%').format(k, round(v * 100, 2))
                print '\n'

        if return_rejects:
            return product_stats_df, rejected
        else:
            return product_stats_df



def CompTrendsDF(period_wks, end_date, products=[None], locations=[None],
                 cities=[None], zipcodes=[None], MA_param=None,
                 shifted=False, normed=False,
                 baseline='t_zero', compute_on_sales=True, NaN_filler=0.0):
    """Construct DataFrame with time series across multiple products or places.
    Default kwargs return a DataFrame with time series of raw sales data, NaNs
    filled with 0.0. Otherwise, assign value to MA_param=.

    Optionally may assign bool True to either shifted= or normed= arguments (NOT BOTH).

    To preserve discontinuities in data (i.e., NaNs) set NaN_filler to None or
    to a tag value close to zero such as 0.0001

    ARGUMENTS:
     -- period_wks: (int) sampling period in weeks
     -- end_date: (date string: '07/15/2016') date string defining
          end of sampling period.

    SPECIFICATIONS FOR COMPARISON AND FILTERING:
    Provide one argument for comparison, optionally add a second argument, either
    a single product or a single place, as a filter. More than two arguments or
    more than one argument that contains multiple values will produce an error.
     -- products: (list of ints or strings) list of product names and/or IDs
     -- locations: (list of ints or strings) list of business names and/or
          IDs
     -- cities: (list of strings) list of cities
     -- zipcodes: (list of 5-digit zipcodes as ints) list of zipcodes

    KEYWORD ARGUMENTS:
     -- MA_param: (int) return dataframe of moving averages; int defines "boxcar"
          window, in weeks, by which to compute moving average
          NOTE: requires value for NaN_filler (!= None)
     -- shifted: (bool, default=False) shift trend data to t0 = 0
          NOTE: requires a non-null value for NaN_filler
     -- normed: (bool, default=False) rescale data to feature range (-1, 1)
          then shift data such that t0 = 0.
          NOTE: requires a non-null value for NaN_filler
     -- baseline: (str, default='t_zero') baseline for shifing data; values:
          * 't_zero' -- shift data by value at t0
          * 'mean' -- shift data to mean = 0
          * 'median' -- shift data to median = 0
     -- compute_on_sales: (bool, default=True) computes on sales data; if False,
          computes on units-sold data
     -- NaN_filler: (float or None, default=0.0) fillna value for raw data, allows marking
          converted NaNs by using tag value such as 0.0001. Set to None to generate
          a CompTrendsDF with only raw, unsmoothed sample data, NaNs in place.
    """
    # Notify user of special arguments error
    if NaN_filler is None and (MA_param or exp_smooth_param or shifted or normed):
        print ("ERROR: MISSING NaN_filler VALUE\n"
        "Value (int or float) must be provided for NaN_filler with MA_param, shifted=True"
        " or normed=True arguments."
        )
        return None

    # Column number to grab specified trend-type from TrendsDF object
    col_index = column_sel(MA_param, shifted, normed)

    # Insert single MA_param into list for SalesTrendsDF class
    if MA_param is not None:
        MA_param = [MA_param]

    counter = 0

    # From user specifications, set variable for comparison and filter
    product_place_args = [products, locations, cities, zipcodes]
    import_type, var_index = select_import_params(product_place_args)

    if import_type == ('A' or 'E'):
        print (
        '\nERROR: CONFLICTING ARGUMENTS ENTERED (OR MISSING) AMONG PRODUCTS, LOCATIONS\n'
        '       CITIES, AND/OR ZIPCODES.\n\n'
        'Provide one argument for comparison and (optionally) add a second argument\n'
        'as a filter. The filter may be a single product or a single place.\n'
        'Check that NOT MORE THAN ONE ARGUMENT contains multiple values.\n'
             )
        return

    if import_type == 'B': # Single product or place specified
        category = 'product' if products[0] is not None else 'place'
        t_df, col_category, filter_name, filter_ID = import_ala_params(
            period_wks, end_date, product=products[0], location=locations[0],
            city=cities[0], zipcode=zipcodes[0], MA_params=MA_param, normed=normed,
            baseline=baseline, compute_on_sales=compute_on_sales, NaN_allowance=100,
            return_trendsDF=True, var_type=category)

        comp_trends_df = build_seed_or_source_df(t_df, NaN_filler, col_category,
                                                 col_index)
        df_title = CompTrendsDF_title(t_df, col_index, var_index, baseline,
                          MA_param, shifted, normed, filter_name, filter_ID)
        comp_trends_df.name = df_title

    if import_type == 'C': # Iterate on multiple products
        for prod in products:
            t_df, col_category, filter_name, filter_ID = import_ala_params(
                period_wks, end_date, product=prod, location=locations[0],
                city=cities[0], zipcode=zipcodes[0], MA_params=MA_param,
                normed=normed, baseline=baseline, compute_on_sales=compute_on_sales,
                NaN_allowance=100, return_trendsDF=True, var_type='product')

            if counter < 1: # Build the seed (base) df with first product
                comp_trends_df = build_seed_or_source_df(t_df, NaN_filler,
                                 col_category, col_index)
                df_title = CompTrendsDF_title(t_df, col_index, var_index,
                    baseline, MA_param, shifted, normed, filter_name, filter_ID)
                comp_trends_df.name = df_title

            else: # Add columns with subsequent products
                build_seed_or_source_df(t_df, NaN_filler, col_category,
                col_index, constr_seed_df=False, seed_df=comp_trends_df)

            counter += 1


    if import_type == 'D': # iterate on places
        if var_index == 1:
            for loc in locations:
                t_df, col_category, filter_name, filter_ID = import_ala_params(
                    period_wks, end_date, product=products[0], location=loc,
                    MA_params=MA_param, normed=normed, baseline=baseline,
                    compute_on_sales=compute_on_sales, NaN_allowance=100,
                    return_trendsDF=True, var_type='place')

                if counter < 1: # Build the seed df with first place
                    comp_trends_df = build_seed_or_source_df(t_df, NaN_filler,
                    col_category, col_index)

                    df_title = CompTrendsDF_title(t_df, col_index, var_index,
                             baseline, MA_param, shifted, normed, filter_name,
                             filter_ID)
                    comp_trends_df.name = df_title

                else: # Add columns with subsequent places
                    build_seed_or_source_df(t_df, NaN_filler, col_category,
                    col_index, constr_seed_df=False, seed_df=comp_trends_df)

                counter += 1

        if var_index == 2:
            for city in cities:
                t_df, col_category, filter_name, filter_ID = import_ala_params(
                    period_wks, end_date, product=products[0], city=city,
                    MA_params=MA_param, normed=normed, baseline=baseline,
                    compute_on_sales=compute_on_sales, NaN_allowance=100,
                    return_trendsDF=True, var_type='place')

                if counter < 1:
                    comp_trends_df = build_seed_or_source_df(t_df, NaN_filler,
                    col_category, col_index)
                    df_title = CompTrendsDF_title(t_df, col_index, var_index,
                             baseline, MA_param, shifted, normed, filter_name,
                             filter_ID)
                    comp_trends_df.name = df_title

                else:
                    build_seed_or_source_df(t_df, NaN_filler, col_category,
                    col_index, constr_seed_df=False, seed_df=comp_trends_df)

                counter += 1

        if var_index == 3:
            for zipcode in zipcodes:
                t_df, col_category, filter_name, filter_ID = import_ala_params(
                    period_wks, end_date, product=products[0], zipcode=zipcode,
                    MA_params=MA_param, normed=normed, baseline=baseline,
                    compute_on_sales=compute_on_sales, NaN_allowance=100,
                    return_trendsDF=True, var_type='place')

                if counter < 1:
                    comp_trends_df = build_seed_or_source_df(t_df, NaN_filler,
                    col_category, col_index)
                    df_title = CompTrendsDF_title(t_df, col_index, var_index,
                             baseline, MA_param, shifted, normed, filter_name,
                             filter_ID)
                    comp_trends_df.name = df_title

                else:
                    build_seed_or_source_df(t_df, NaN_filler, col_category,
                    col_index, constr_seed_df=False, seed_df=comp_trends_df)

                counter += 1

    try:
        return comp_trends_df

    except UnboundLocalError:
        print (
        '\nERROR: CONFLICTING ARGUMENTS ENTERED (OR MISSING) AMONG PRODUCTS, LOCATIONS\n'
        '       CITIES, AND/OR ZIPCODES.\n\n'
        'Provide one argument for comparison and (optionally) add a second argument\n'
        'as a filter. The filter may be a single product or a single place.\n'
        'Check that NOT MORE THAN ONE ARGUMENT contains multiple values.\n'
             )
        return


def column_sel(MA_param=None, shifted=False, normed=False):
    """Return integer for DataFrame column selection"""
    if MA_param is not None:
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

def build_seed_or_source_df(t_df, NaN_filler, col_category, col_index,
        constr_seed_df=True, seed_df=None):
    if constr_seed_df: # Create base dataframe
        if NaN_filler is not None:
            t_df.fillna(NaN_filler, inplace=True)
        comp_trends_df = pd.DataFrame(t_df[t_df.columns[col_index]])
        comp_trends_df.columns = [col_category]

        return comp_trends_df

    else: # Add new columns to base dataframe
        if NaN_filler is not None:
            t_df.fillna(NaN_filler, inplace=True)

        seed_df[col_category] = t_df.iloc[:,col_index]

def CompTrendsDF_title(t_df, col_index, var_index, baseline, MA_param, shifted,
                       normed, filter_name, filter_ID):
    """Construct df.name for CompTrendsDF."""

    if filter_name is not None:
        if var_index is not None: # filter is a single PRODUCT
                A = '{} (ID: {}), '.format(filter_name, filter_ID)
        else: # filter is a PLACE
            if filter_ID is not None: # filter is a business
                A = 'Business: {} (ID: {}), '.format(filter_name, filter_ID)
            elif not filter_name.isdigit():
                A = 'City: {}, '.format(filter_name)
            else:
                A = 'Zipcode: {}, '.format(filter_name)
    else: # no filter specified
        A = ''

    if MA_param is not None:
        B = '{}-Week Moving Average of '.format(MA_param[0])
    bsln = baseline.capitalize() if baseline != 't_zero' else 'T0 = 0'
    C = ', Data Shifted to {}'.format(bsln)
    D = ', Data Rescaled (-50, 50) then Shifted to {}'.format(bsln)
    E = t_df.name.split('in ')[1]
    if col_index == 0:
        title = A + E
    if col_index == 1:
        title = A + E + C
    if col_index == 2:
        title = A + E + D
    if col_index == 3 and MA_param and not shifted:
        title = A + B + E
    if col_index == 4 and MA_param and shifted:
        title = A + B + E + C
    if col_index == 5 and MA_param and normed:
        title = A + B + E + D

    return title


class RankProductsPlaces(object):
    """Initialize with SalesStatsDF object and (optionally) by number of top
    results desired; Rank products or places by user_selected statistic.

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
     -- results: pandas DataFrame of products or places ranked by selected statistic
     -- ranked_products: numpy array of ranked product IDs
     -- ranked_places: numpy array of ranked places or place IDs
     -- ranked_df: same as RankProducts.results but including all other statistics

    """

    def __init__(self, sales_stats_df, N_results=None):
        self.sales_stats_df = sales_stats_df
        self.N_results = N_results
        self.results = None
        self.ranked_products = None
        self.ranked_places = None
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

        # Grab the column with statistic for ranking
        if stat:
            cols = self.sales_stats_df.columns
            if stat == 'sales':
                stat_idx = 4
            if not smoothed and stat == 'gain':
                stat_idx = 5
            if not smoothed and stat == 'rate':
                stat_idx = 6
            if smoothed and stat == 'gain':
                stat_idx = 7
            if smoothed and stat == 'rate':
                stat_idx = 8

            stat_col = cols[stat_idx]

        else:
            stat_idx = self._sel_rank_by()
            stat_col = self.sales_stats_df.columns[stat_idx]

        output_cols = list(self.sales_stats_df.columns)
        output_cols.remove(stat_col)
        output_cols.insert(4, stat_col)

        ranked = self.sales_stats_df.sort_values(by=stat_col, ascending=False)
        ranked.index = range(1, len(ranked.index) + 1)

        if self.N_results:
            # if ranking by product...
            if len(self.sales_stats_df['product_name'].unique()) > 1:
                self.ranked_df = ranked[output_cols][:self.N_results]
                self.ranked_products = self.ranked_df['product_id'].values
                self.results = self.ranked_df.iloc[:,:5]
                self.results.drop(['product_id' ,'place_name' ,'place_id'],
                                    axis=1, inplace=True)
            else: # if ranking by place...
                self.ranked_df = ranked[output_cols][:self.N_results]
                self.ranked_places = self.ranked_df['place_name'].values
                self.results = self.ranked_df.iloc[:,:5]
                self.results.drop(['product_name', 'product_id', 'place_id'],
                                    axis=1, inplace=True)

        else:
            # if ranking by product...
            if len(self.sales_stats_df['product_name'].unique()) > 1:
                self.ranked_df = ranked[output_cols]
                self.ranked_products = self.ranked_df['product_id'].values
                self.results = self.ranked_df.iloc[:,:5]
                self.results.drop(['product_id' ,'place_name' ,'place_id'],
                                    axis=1, inplace=True)
            else: # if ranking by place...
                self.ranked_df = ranked[output_cols]
                self.ranked_places = self.ranked_df['place_name'].values
                self.results = self.ranked_df.iloc[:,:5]
                self.results.drop(['product_name', 'product_id', 'place_id'],
                                    axis=1, inplace=True)

        self.results.name = self.sales_stats_df.name
        self.ranked_df.name = self.sales_stats_df.name + \
                ', Ranked by {}'.format(stat_col)


    def _sel_rank_by(self):
        "Prompt user for column for ranking; return its index"
        cols = self.sales_stats_df.columns[4:]
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

def HbarData(period_wks, end_date, products=[None], locations=[None],
             cities=[None], zipcodes=[None], MA=5,
             rank_on_sales=True, NaN_allowance=5, print_rejects=False,
             rank_by=['sales'], fixed_order=True):
    """Return a dataframe configured for custom plotting in HbarRanked function
    Called within HBarRanked in graph_trends.py. See documentation there for
    further details.

    ARGUMENTS
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
          * 'sales' = cumulative sales over period
          * 'gain' = uniform weekly gain or loss over period
          * 'rate' = growth rate index for products with data
              normalized (rescaled -100, 100) for sales volumes
     -- fixed_order: (bool, default=True) only rank products in the primary
          bar graph and maintain that rank-order in secondary graphs; if False,
          rank products in each bar graph.
     -- NaN_allowance: (int from 0 to 100, default=5) max allowable percentage of
          NaNs in product ts samples for statistical aggregation; products
          exceeding allowance are discarded from rankings.
     -- print_rejects: (bool, default=False) If True, print report of products
          rejected for excess null values in sample, with their corresponding
          percentage of nulls in sample."""

    boxcar = [MA] if MA is not None else None

    product_place_args = [products, locations, cities, zipcodes]
    import_type, var_index = select_import_params(product_place_args)

    if import_type == 'C': # ranking products
        drop_cols = ['product_id', 'place_name', 'place_id']
        append_col = 'product_name'
    if import_type == 'D': # ranking places
        drop_cols = ['product_id', 'product_name', 'place_id']
        append_col = 'place_name'

    prod_stats = SalesStatsDF(period_wks, end_date, products, locations,
                cities, zipcodes, MA_params=boxcar, compute_on_sales=rank_on_sales,
                NaN_allowance=NaN_allowance, print_rejects=print_rejects)

    if MA is not None:
        base_name = prod_stats.name + ' -- {}-Week Moving Average'.format(MA)
    else:
        base_name = prod_stats.name + ' -- '


    if len(rank_by) == 1 or fixed_order: # just need the RankProductsPlaces.results object
        if len(rank_by) == 1:
            rank_1 = RankProductsPlaces(prod_stats)
            if MA is not None:
                rank_1.main(stat=rank_by[0])
            else:
                rank_1.main(smoothed=False, stat=rank_by[0])
            data = rank_1.ranked_df
            data.drop(drop_cols, axis=1, inplace=True)

        else:
            rank_1 = RankProductsPlaces(prod_stats)
            rank_1.main(smoothed=MA, stat=rank_by[0])
            all_data = rank_1.ranked_df
            df_cols = all_data.columns
            cols = []
            for rank_stat in rank_by:
                cols.append(append_col)
                cols.append(grab_column(stat=rank_stat, smoothed=MA))

            data = all_data[cols]


    if len(rank_by) > 1 and not fixed_order:
            rank_1 = RankProductsPlaces(prod_stats)
            rank_1.main(smoothed=MA, stat=rank_by[0])
            data = rank_1.results

            for i, rank_stat in enumerate(rank_by[1:]):
                rank_next = RankProductsPlaces(prod_stats)
                rank_next.main(smoothed=MA, stat=rank_stat)
                next_ranked = rank_next.results
                data['Ranking By {}'.format(rank_stat)] = next_ranked.iloc[:,0].values
                data[next_ranked.columns[-1]] = next_ranked.iloc[:,-1].values

            # data.drop(drop_cols, axis=1, inplace=True)

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

def BestSellerData(period_wks, end_date, products=[None], locations=[None],
                   cities=[None], zipcodes=[None], MA_param=None, NaN_allowance=5,
                   print_rejects=False, compute_on_sales=True, N_periods=10,
                   freq='7D', rank_by='rate'):
    """Return objects for graphing in graph_trends.PlotBestSellers():
    1) BestSellers_df: dataframe summarizing rankings for products over a series of
        N-identical-length periods spaced at equal intervals and
    2) labeler: dictionary containing label names and positions for plot


    ARGUMENTS:
     -- period_wks: (int, default=10) length of sampling periods in weeks
     -- end_date: (date string of form 'MM/DD/YYYY', default=None) end_date of most recent
         ranking period

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

    OPTIONAL:
     -- MA_param: (int or NoneType, default=None) rolling "boxcar" window, in weeks, by which to
          compute moving averages; None: ranks on non-smoothed (raw) trend data
     -- NaN_allowance: (int or float from 0 to 100, default=5) max allowable
          percentage of NaNs in product ts samples for statistical aggregation;

     -- print_rejects: (bool, default=False) If True, print any products rejected
          for excess null values in sample with their corresponding ratio of nulls
          present in the dataset
     -- compute_on_sales: (bool, default=True) ranks on sales data; if False,
          ranks on units-sold data
     -- N_periods: (int, default=10) number of periods including latest for comparison
     -- freq: (str, default='7D') pandas date_range() argument; interval between
         periods for comparison. Other possible values: 'W' (Sunday-ending week),
         'M' (month), 'Y', '2W', etc. See documentation for Pandas base time-series
         frequencies.
     -- rank_by: (string, default='rate') statistic by which to rank products.
          Values:
          * 'rate' = growth rate index for products with data
              normalized (rescaled -100, 100) for sales volumes
          * 'gain' = uniform weekly gain or loss over period
          * 'sales' = cumulative sales over period
    """

    # Identify variable and filter for comparison
    product_place_args = [products, locations, cities, zipcodes]
    import_type, var_index = select_import_params(product_place_args)

    # Generate list of end_dates for multiple ranking periods
    end_dates = generate_dates(end_date, N_periods, freq)

    # Generate data consisting of product rankings over multiple periods
    data_A = OrderedDict()

    excess_null_error_msg = ('\nDATA FOR SOME PRODUCTS OR PLACES CONTAINED TOO MANY NULLS TO RANK.\n\n'
          '** For details, re-run function with print_rejects keyword argument'
          ' set to True.\n\n** To ignore null values and proceed with rankings'
          ' (substituting zero for\nnulls in computations),'
          ' re-run function and set keyword argument NaN_allowance to 100.'
          )

    # Generate rankings and add to data dictionary (data_A) where keys = specified
    # end_dates of the comparison periods; values = the product IDs ordered by rank
    for i, end_d in enumerate(end_dates):
        # On user command, executes diagnostic to reveal products w excess nulls
        if print_rejects:
            print('EXCESS NULL VALUES FROM PERIOD ENDING {}:\n').format(end_d)
        else: pass

        # Compute stats on products for one test period at a time
        if MA_param is not None:
            psdf, rej = SalesStatsDF(period_wks=period_wks, end_date=end_d,
                    products=products, locations=locations, cities=cities,
                    zipcodes=zipcodes,
                    MA_params=[MA_param], NaN_allowance=NaN_allowance,
                    print_rejects=print_rejects, return_rejects=True,
                    normed=True, compute_on_sales=compute_on_sales
                         )
            # If undiagnosed excess nulls, print error message and exit from function
            if len(rej) > 0:
                if not print_rejects:
                    print(excess_null_error_msg)
                    return None, None
                else: pass
            else: pass

        else: # if no moving-avg window specified . . .
            psdf, rej = SalesStatsDF(period_wks=period_wks, end_date=end_d,
                    products=products, locations=locations, cities=cities,
                    zipcodes=zipcodes,
                    NaN_allowance=NaN_allowance,
                    print_rejects=print_rejects, return_rejects=True,
                    normed=True, compute_on_sales=compute_on_sales
                            )
            if len(rej) > 0:
                if not print_rejects:
                    print(excess_null_error_msg)
                    return None, None
                else: pass
            else: pass

        ranked = RankProductsPlaces(psdf)
        if MA_param is not None:
            ranked.main(smoothed=True, stat=rank_by)
        else:
            ranked.main(smoothed=False, stat=rank_by)
        if import_type == 'C':
            data_A[end_d] = ranked.ranked_products
        else:
            data_A[end_d] = ranked.ranked_places

    # Reconfigure data_A into a dictionary (data_B) of keys=products or places,
    # values=list of a product/place's rankings over the series of comparison periods
    data_B = OrderedDict()
    if import_type == 'C': # variable = products
        # create dictionary with keys as products and values as empty lists
        for prod in products:
            if type(prod) == str:
                data_B[prod.lower()] = []
            else:
                data_B[product_name_from_ID(prod)] = []
        # append rankings to list-values in dictionary
        for prod_arr in data_A.itervalues():
            for i, prod in enumerate(prod_arr):
                if type(prod) == str:
                    data_B[prod.lower()].append(i+1) # i+1 represents product rank
                else:
                    data_B[product_name_from_ID(prod)].append(i+1)

    if import_type == 'D': # variable = places
        for place in product_place_args[var_index]:
            if type(place) == str:
                data_B[place.lower()] = []
            else:
                if var_index == 3: # variable = zipcodes
                    data_B[str(place)] = []
                else:
                    data_B[locations_name_from_ID(place)] = []

        for place_arr in data_A.itervalues():
            for i, place in enumerate(place_arr):
                if type(place) == str:
                    data_B[place.lower()].append(i+1) # i+1 represents product rank
                else:
                    if var_index == 3: # variable = zipcodes
                        data_B[str(place)].append(i+1)
                    else: # variable is location by ID
                        data_B[locations_name_from_ID(place)].append(i+1)

    # Construct output dataframes
    mask = lambda x: datetime.strptime(x, '%m/%d/%Y')
    date_idx = [mask(dt) for dt in end_dates] # for DatetimeIndex of BestSellers_df

    title = best_seller_title(MA_param, compute_on_sales, N_periods,
                          period_wks, rank_by, freq)

    try: # Exits function if data contains excess null values
        if import_type == 'C':
            df_A = pd.DataFrame(data_A, index=range(1, len(products)+1))
        else:
            df_A = pd.DataFrame(data_A,
                      index=range(1, len(product_place_args[var_index])+1)
                      )
    except ValueError:
        if not print_rejects:
            print(excess_null_error_msg)
        return None, None
    else:
        # index of df_A represents rank levels 1 to N
        BestSellers_df = pd.DataFrame(data_B, index=date_idx)

        # Sort BestSellers_df columns by cumulative rankings
        sum_o_ranks = BestSellers_df.sum()
        foo = sum_o_ranks.sort_values(ascending=True)
        sorted_by_best = list(foo.index)
        BestSellers_df = BestSellers_df[sorted_by_best]

        df_A.name = title
        BestSellers_df.name = title

        # Create labeler dict with labels (keys) and their positions (vals)
        if import_type == 'C':
            labels = [names_formatted[product_name_from_ID(prod_ID)] \
                      for prod_ID in df_A.iloc[:,-1]]

        if import_type == 'D':
            if var_index == 1 and type(locations[0]) != str:
                # if variable is location by ID number
                labels = [locations_name_from_ID(place_ID) \
                          for place_ID in df_A.iloc[:,-1]]
            elif var_index == 3: # variable is zipcode
                labels = [str(place_ID) for place_ID in df_A.iloc[:,-1]]
            else:
                labels = [place_ID for place_ID in df_A.iloc[:,-1]]


        labeler = {}
        for i, var in enumerate(labels):
            labeler[var] = i + 1


        return BestSellers_df, labeler



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
SUPPORTING AND STAND-ALONE FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def biz_lookup(conn_str, cities=None, zipcodes=None):
    """
    Return a list of IDs (ints) for all businesses in either:
     -- A city (str) or list of cities, OR
     -- A zipcode (5-digit int) or list of zipcodes
    """
    if cities is not None and zipcodes is not None:
        print "\nERROR: Enter value(s) for city OR zip; NOT BOTH."
        return

    conn = create_engine(conn_str)
    biz_list = []

    if type(cities) != list:
        cities = [cities]
    if type(zipcodes) != list:
        zipcodes = [zipcodes]

    if cities[0] is not None:
        A = """
            SELECT DISTINCT(locs.name)
            FROM locations locs
            JOIN weekly_sales ws
            ON locs.wa_location_id = ws.location_id
            WHERE locs.city = '{}'
            """.format(cities[0].upper())

        B = ''

        if len(cities) > 1:
            for city in cities[1:]:
                B += '\nOR locs.city = {}'.format(city)

        C = ';'

        query_by_city = A + B + C
        results = conn.execute(query_by_city)
        for shop in results:
            biz_list.append(locations_dict[shop[0]])
    else:
        A = """
            SELECT DISTINCT(location_id)
            FROM weekly_sales
            WHERE zip = '{}'
            """.format(zipcodes[0])

        B = ''

        if len(zipcodes) > 1:
            for zipcode in zipcodes[1:]:
                B += '\nOR zip = {}'.format(zipcode)

        C = ';'

        query_by_zip = A + B + C
        results = conn.execute(query_by_zip)
        for shop in results:
            biz_list.append(shop[0])

    return biz_list


def select_import_params(arg_list):
    """
    Return tuple (str, int or None) containing ImportSalesData initialization configs with
    iterable 'variable' and index of 'filter' as derived from user-inputs in parent function.
    Return error trigger if user arguments contain conflicting values.
    Types:

    A = Import sales data statewide for all products (no variable or filter)
    B = Import sales data for a single product OR a single location OR a single product
          in a single location
    C = Import sales data for multiple products (variable), optionally within a single
          location (filter)
    D = Import sales data for multiple locations (variable), optionally filtered by a
          single product (filter)
    E = ERROR_1: Conflicing values contained in arguments
    """

    # find variable; the product or place category for comparison via iteration
    mask_1 = lambda x: len(x) > 1
    find_var = map(mask_1, arg_list)

    # find arguments with user-specified values
    mask_2 = lambda x: x[0] is not None
    find_args = map(mask_2, arg_list)


    if find_var.count(True) == 0: # If no arguments contain more than one value...
        if find_args.count(True) == 0: # user did not specify variable or filter
            return 'A', None
        elif find_args.count(True) > 2: # user entered too many arguments
            return 'E', None
        else: # User specified a single value in a single argument, a product or a place
            return 'B', None

    elif find_var.count(True) > 1: # if user arguments exceed the limit of one variable
        return 'E', None

    elif find_var.count(True) == 1: # If one argument contains multiple values (the 'variable')
        if find_var.index(True) == 0: # Compare / iterate on product
            return 'C', None
        else: # Compare / iterate on places
            return 'D', find_var.index(True)


def import_ala_params(period_wks, end_date, product=None, location=None, city=None, zipcode=None,
                      MA_params=None, normed=True, baseline='t_zero', compute_on_sales=True,
                      NaN_allowance=5, return_trendsDF=False, var_type='product'):
    """
    Import sales data per user params in parent function and return the following objects (tup):
    If return_trendsDF set to False (for SalesStatsDF):
        -- SalesTrends.trend_stats
        -- SalesTrends.NaNs_ratio
        -- SalesTrends.trendsDF.name

    Else (for CompTrendsDF):
        -- SalesTrends.trendsDF
        -- (str) SalesTrends.product_name (if var_type set to 'product')
             OR SalesTrends.place_name (if var_type != 'product')
    """
    raw_data = ImportSalesData(product, location, city, zipcode)
    raw_data.main()
    if raw_data.product_df is not None:
        if compute_on_sales:
            ts = raw_data.sales
        else:
            ts = raw_data.units_sold

        trends_data = SalesTrendsDF(ts, period_wks, end_date, MA_params, normed, baseline,
                                      NaN_filler=0.0)
        trends_data.main()

        if return_trendsDF:
            if var_type == 'product':
                var_name = trends_data.product_name
                filter_name = \
                None if trends_data.place_name is None else trends_data.place_name
                filter_ID = \
                None if trends_data.place_ID is None else trends_data.place_ID
            else:
                var_name = trends_data.place_name
                filter_name = \
                None if trends_data.product_name is None else trends_data.product_name
                filter_ID = \
                None if trends_data.product_ID is None else trends_data.product_ID

            return (trends_data.trendsDF,
                    var_name,
                    filter_name,
                    filter_ID)

        else:
            return (trends_data.trend_stats,
                    trends_data.NaNs_ratio,
                    trends_data.trendsDF.name)

    else:
        params_lst = [product, location, city, zipcode]
        print ('ERROR: Invalid entry, no sales data associated with this combination\n'
               'of values: {}\n'.format(params_lst)
               )
        if not return_trendsDF:
            return None, None, None
        else:
            return None, None, None, None


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



class ImportSalesDataOLD(object):
    """
    FOR USE IN "TOY DATABASE" CONSISTING OF DATA AGGREGATED TO DAILY FREQUENCY
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

    """Run SalesTrendsDF method and access class attributes"""
    trends_3 = SalesTrendsDF(sales_3, sample_period, MA_params=MAs)
    trends_3.main()
    trends_df_3 = trends_3._trendsDF # DataFrame with columns of transformed data
    stats_3 = trends_3.trend_stats # Single record (OrderedDict) of stats for product

    """Run SalesStatsDF function to generate comparative stats DF; Builds DF from
    individual records in the form of SalesTrendsDF.trend_stats objects"""
    comps_df = SalesStatsDF(products, sample_period, MA_params=MAs)

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
