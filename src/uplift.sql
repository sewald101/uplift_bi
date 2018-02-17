
/* DAILY SALES for STRAIN */
SELECT CAST(DATE_TRUNC('day', ds.date_of_sale) AS DATE) as date
 , st.strain_display_name as strain_name
 , st.generic_strain_id as strain_id
 , ROUND(SUM(ds.retail_price)) as ttl_sales
 , ROUND(SUM(ds.retail_units)) as ttl_units_sold
FROM daily_retail_sales ds
JOIN strains st
ON ds.strain_name = st.strain_display_name
WHERE st.generic_strain_id = 4
GROUP BY date, st.strain_display_name, strain_id
ORDER BY date;

/* COPY lemon_haze_18 TABLE to CSV */
COPY id0017_blue_cheese
TO '/data/uplift/Data/strain_data/0017_blue_cheese.csv'
DELIMITER ',' CSV HEADER;


/* PROTOTYPE RAW DATA TABLE FOR SINGLE STRAIN (18, LEMON HAZE) */
CREATE TABLE id0017_blue_cheese AS (
SELECT wa_inventory_id
 , generic_strain_id
 , CAST(first_retail_sale_at AS DATE) AS first_rtl_sale
 , CAST(most_recent_retail_sale_at AS DATE) AS latest_rtl_sale
 , seller_price AS wholesale_COGS
 , ROUND(total_retail_sales_amount, 2) AS ttl_rtl_sales
 , CAST(total_retail_sales_units AS INTEGER) AS units_sold
FROM product_skus
WHERE generic_strain_id = 17
AND retail_location_id IS NOT NULL /* FILTER FOR RETAIL TRANSACTIONS */
AND first_retail_sale_at IS NOT NULL /* FILTER  FOR PRESENCE OF FIRST SALE DATE*/
AND most_recent_retail_sale_at IS NOT NULL /* DITTO FOR LATEST SALE DATE*/
ORDER BY seller_transfer_date
);


/* CREATE strains TABLE: STRAIN ID, NAME, COUNT OF PRODUCTS, COUNT OF RETAIL RECORDS
Filters for retail only commented out */
CREATE TABLE strains AS (
WITH retail_inv_ids AS (
SELECT generic_strain_id
 , COUNT(wa_inventory_id) AS num_inv_ids
FROM product_skus
--WHERE retail_location_id IS NOT NULL /*FILTER FOR RETAIL TRANSACTIONS */
--AND seller_transfer_date IS NOT NULL /*FILTER FOR PRESENCE OF WHOLESALE DATE*/
--AND most_recent_retail_sale_at IS NOT NULL /*DITTO FOR LATEST RETAIL SALE DATE*/
--AND generic_strain_id BETWEEN 1 AND 20
GROUP BY generic_strain_id
ORDER BY generic_strain_id
)

SELECT ps.generic_strain_id
 , ps.strain_display_name
 , COUNT(DISTINCT(ps.user_product_description)) AS num_products
 , ri.num_inv_ids
FROM product_skus ps
JOIN retail_inv_ids ri
ON ps.generic_strain_id = ri.generic_strain_id
--WHERE ps.generic_strain_id BETWEEN 1 AND 20
GROUP BY ps.generic_strain_id
 , ps.strain_display_name
 , ri.num_inv_ids
ORDER BY ps.generic_strain_id
);


/* SINGLE SALES RECORD BY INVENTORY ID(s)*/
SELECT ps.wa_inventory_id
 , ps.generic_strain_id
 , ps.strain_display_name
 , ps.user_product_description AS product
 , ps.inventory_type_id AS product_type_id
 , l.name AS wholesaler
 , ps.seller_transfer_date AS wholesale_date
 , ps.seller_price AS wholesale_price
 , ps.total_retail_sales_amount AS total_rtl_sales
 , ps.total_retail_sales_amount - ps.seller_price AS gross_profit
 , ps.total_retail_sales_units AS units_sold
 , ps.retailer_name AS retailer
 , ps.first_retail_sale_at AS first_sale
 , ps.most_recent_retail_sale_at AS last_sale
FROM product_skus ps
JOIN locations l
ON ps.seller_location_id = l.id
WHERE ps.wa_inventory_id = 6033240110000277;
/*
OR ps.wa_inventory_id = 6033240110000315
OR ps.wa_inventory_id = 6033240110000330
OR ps.wa_inventory_id = 6033240110000456
OR ps.wa_inventory_id = 6033240110000313
OR ps.wa_inventory_id = 6033240110000309
OR ps.wa_inventory_id = 6033240110000312  */



/* IDs FOR SAMPLING RECORDS VIA QUERY ABOVE*/
Parent Inventory_ID: 8783735443597681 (record # 300)
Retail Inventory_ID: 6034295410005354 (record # 20) NO strain info here!


/* QUERY VARIETIES of STRAIN NAME*/
SELECT DISTINCT(strain_display_name)
FROM product_skus
WHERE strain_display_name LIKE '%Gorilla%';


/* COUNT OF VARIETIES OF STRAIN NAME*/
SELECT COUNT(DISTINCT(user_strain_name))
FROM product_skus
WHERE user_strain_name LIKE '%Gorilla%';


/* TOTAL WHOLESALE AND RETAIL SALES BY STRAIN DISPLAY NAME*/
SELECT strain_display_name
 , COUNT(strain_display_name) as num_records
 , SUM(total_retail_sales_amount) as total_sales
 , SUM(seller_price) as total_by_seller
FROM product_skus
WHERE strain_display_name LIKE '%Purple Gorilla%'
GROUP BY strain_display_name;

/*Max's query for Daily Sales by Retailer and Strain*/
SET TIME ZONE '-08';
SELECT
    date_trunc('day', d.sessiontime)       AS date_retail_sales
    , d.location                           AS retailer
    , p.strain_display_name                AS strain
    , SUM(price)                           AS retail_price
    , SUM(weight)                           AS retail_units
FROM biotrackthc_dispensing AS d
JOIN products AS p ON p.wa_inventory_id = d.inventoryid
WHERE (d.deleted = 0)
    --AND (d.refunded = 0)
    AND (p.generic_strain_id < 10)
    AND d.sessiontime > '01-01-2016'
GROUP BY date_retail_sales
        , retailer
        , strain

/* CREATE TABLE FOR daily_sales*/
CREATE TABLE daily_retail_sales (
  record INTEGER,
  date_of_sale TIMESTAMP,
  retailer_id INTEGER,
  strain_name VARCHAR,
  retail_price NUMERIC(15,2),
  retail_units NUMERIC(15,4)
);

/* COPY DATA INTO daily_retail_sales TABLE */
COPY daily_retail_sales
FROM '/data/uplift/Data/TopRetailStrains.csv'
DELIMITER ',' CSV HEADER;

COPY daily_sales
FROM '/data/uplift/Data/midLowRetailStrains.csv'
DELIMITER ',' CSV HEADER;

COPY daily_sales
FROM '/data/uplift/Data/inventoryretailsales.csv'
DELIMITER ',' CSV HEADER;

COPY locations
TO '/data/uplift/Data/locations.csv'
DELIMITER ',' CSV HEADER;


/* CREATE TABLE FOR DAILY_SALES_DATA*/
CREATE TABLE  daily_sales AS (
  SELECT drs.record
   , drs.date_of_sale
   , drs.retailer_id
   , drs.strain_name
   , st.generic_strain_id
   , drs.retail_price
   , drs.retail_units
  FROM daily_retail_sales drs
  JOIN strains st
  ON drs.strain_name = st.strain_display_name
);


/* QUERY for ImmportSalesData function*/
SELECT CAST(DATE_TRUNC('day', date_of_sale) AS DATE) as date
 , strain_name as product_name
 , generic_strain_id as product_id
 , ROUND(SUM(retail_price)) as ttl_sales
 , ROUND(SUM(retail_units)) as ttl_units_sold
FROM daily_sales
WHERE generic_strain_id = 1118
GROUP BY date, strain_name, generic_strain_id
ORDER BY date;



/* CREATE TABLE FOR product_skus */
CREATE TABLE product_skus (
  id INTEGER,
  wa_seller_org_id INTEGER,
  seller_organization_id INTEGER,
  wa_inventory_id NUMERIC,
  wa_transaction_id NUMERIC,
  wa_inventory_parent_id NUMERIC,
  wa_seller_location_id INTEGER,
  seller_location_id INTEGER,
  first_retail_sale_at TIMESTAMP,
  most_recent_retail_sale_at TIMESTAMP,
  wa_retail_location_id INTEGER,
  retail_location_id INTEGER,
  lowest_retail_unit_price NUMERIC(15,2),
  highest_retail_unit_price NUMERIC(15,2),
  avg_retail_unit_price NUMERIC(15,2),
  unit_weight NUMERIC(10,3),
  generic_strain_id INTEGER,
  strain_display_name VARCHAR,
  user_strain_name VARCHAR,
  strain_type_id INTEGER,
  is_prerolled VARCHAR,
  user_product_description VARCHAR,
  total_thc_amount NUMERIC(15,4),
  total_cbd_amount NUMERIC(15,4),
  total_potency_amount NUMERIC(15,4),
  thc_potency_range_id INTEGER,
  cbd_potency_range_id INTEGER,
  total_potency_range_id INTEGER,
  cannabinoid_type_id INTEGER,
  unit_weight_range_id INTEGER,
  retail_unit_price_range_id INTEGER,
  inventory_type_id INTEGER,
  retailer_name VARCHAR,
  generic_sku VARCHAR,
  tight_sku VARCHAR,
  total_retail_sales_amount NUMERIC(15,4),
  total_retail_sales_weight NUMERIC(15,4),
  total_retail_sales_units NUMERIC(15,4),
  seller_price NUMERIC(15,2),
  seller_unit_price NUMERIC(15,2),
  wa_transfer_original_id INTEGER,
  seller_transfer_date TIMESTAMP,
  seller_units NUMERIC(15,4),
  buyer_location_id INTEGER,
  wa_buyer_location_id INTEGER,
  wa_inventory_transfer_id INTEGER
);

/* COPY DATA INTO product_skus TABLE */
COPY product_skus
FROM '/data/uplift/Data/transactions.csv'
DELIMITER E'\t' CSV HEADER;

/* CREATE locations TABLE*/
CREATE TABLE locations (
  id INTEGER,
  organization_id INTEGER,
  wa_location_id INTEGER,
  name VARCHAR,
  address1 VARCHAR,
  address2 VARCHAR,
  city VARCHAR,
  state VARCHAR,
  zip VARCHAR,
  wa_location_type_id VARCHAR,
  licensenum VARCHAR,
  lat NUMERIC(9, 6),
  lon NUMERIC(9, 6),
  producer VARCHAR,
  processor VARCHAR,
  retail VARCHAR,
  deleted VARCHAR
);

/* COPY DATA INTO locations TABLE */
COPY locations
FROM '/data/uplift/Data/locations_table_Nov2017_Do_Not_Distribute.csv'
DELIMITER ',' CSV HEADER;

/* SELECT CATEGORIES FOR POSSIBLE PRICE AGGREGATION*/
SELECT generic_strain_id
 , strain_display_name
 , user_strain_name
 , user_product_description
 , total_retail_sales_amount
 , total_retail_sales_units
 , seller_price
 , seller_unit_price
 , seller_units
FROM product_skus
LIMIT 5;

/* NUMBERS of VARIOUS DISTINCT VALUES*/
SELECT COUNT(DISTINCT(wa_inventory_id)) as inv_id
 , COUNT(DISTINCT(generic_strain_id)) as generic_strain_id
 , COUNT(DISTINCT(strain_display_name)) as strain_display_name
 , COUNT(DISTINCT(user_strain_name)) as user_strain_name
 , COUNT(DISTINCT(user_product_description)) as usr_prod_descr
FROM product_skus;

/* Key: STRAIN ID | DISPLAY NAME*/
SELECT generic_strain_id
 , strain_display_name
FROM product_skus
WHERE generic_strain_id BETWEEN 1 AND 15
GROUP BY strain_display_name, generic_strain_id
ORDER BY generic_strain_id;
