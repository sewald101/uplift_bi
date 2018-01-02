"""From Max Caughron, 2017-Dec
"""


# libraries used
from tabulate import tabulate
from os import getenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy
import geopy.distance
from scipy.spatial import cKDTree
from scipy import inf
%matplotlib inline

import pyodbc
import gf_logins



def mssqldb2pandasdf(sql_to_import= 'SELECT name, loclatitude, loclongitude, locationtype FROM dbo.locations;' ):
    config = dict(server=   "dev02.getgrowflow.com",
                  port=      1433,
                  database= 'biotrack_2017_04',
                  username= 'gf',
                  password= 'Must25ard!')
    conn_str = ('SERVER={server},{port};'   +
                'DATABASE={database};'      +
                'UID={username};'           +
                'PWD={password}')
    conn = pyodbc.connect(
        r'DRIVER={ODBC Driver 13 for SQL Server};' +
        conn_str.format(**config)
        )
    cursor = conn.cursor()
    sql = sql_to_import
    return pd.read_sql(sql, conn)

def locationer(l):
    for place in set(l):
        return list(df.loc[df.long_lat == place, "name"])

def locations_in_n_dist(df, max_distance = 0.001): # in degree distance, needst ob
    df['long_lat'] = list(zip(df.loclatitude, df.loclongitude))
    tree = cKDTree(list(np.array(df.long_lat)))
    points = list(np.array(df.long_lat))
    point_neighbors_list = []
    for point in points:
        distances, indicies = tree.query(point, len(points), p=2, distance_upper_bound=max_distance)
        point_neighbors = []
        for index, distance in zip(indicies, distances):
            if distance == inf:
                break
            point_neighbors.append(points[index])
        point_neighbors_list.append(point_neighbors)
    df['neighbor_long_lats'] = point_neighbors_list
    return df

def locationer(l):
    for place in set(l):
        return list(df.loc[df.long_lat == place, "name"])

def neighborlookup(df):
    df['n_radius_list'] = df.neighbor_long_lats.apply(lambda x: locationer(x))
    short_df = df[["name", "n_radius_list", "long_lat", "locationtype"]]
    return short_df


if __name__ == "__main__":
    df = mssqldb2pandasdf()
    df = locations_in_n_dist(df=df)
    df = neighborlookup(df=df)
    print df
