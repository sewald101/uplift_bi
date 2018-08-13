# uplift_bi
### Suite of sales trend analysis tools for Washington State's Initiative 502 legal cannabis industry
Delivered to Uplift BI, LLC (April 2017)

Presentation: https://tinyurl.com/y888jp7w

Data sources:
* Uplift BI, LLC (Seattle, WA)  
* Washington State Liquor and Cannabis Board
              
Core code:
* [trend_analysis.py](https://github.com/sewald101/uplift_bi/blob/master/src/trend_analysis.py): Classes and back-end functions that query 
the database and transform time-series data for statistical aggregation and sales performance ranking and comparison
* [graph_trends.py](https://github.com/sewald101/uplift_bi/blob/master/src/graph_trends.py): Functions implemented in Matplotlib
that provide a front-end user interface for exploring and visualizing sales trends and performance along user-defined period,
product and location/geographic parameters
