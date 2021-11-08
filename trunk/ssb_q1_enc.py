import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# encode and filter out records for SSB Q1 emulation

cwd = os.getcwd()
os.chdir(cwd+'/test/dbgen')

lineorder = pd.read_csv("lineorder.tbl", sep='|',
                       names=["lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate",
                              "lo_orderpriority", "lo_shippriority", "lo_quantity", "lo_extendedprice", "lo_ordtotalprice", "lo_discount",
                              "lo_revenue", "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode", "null"])

date = pd.read_csv("date.tbl", sep='|',
                       names=["d_datekey", "d_date", "d_dayofweek", "d_month", "d_year", "d_yearmonthnum",
                              "d_yearmonth", "d_daynuminweek", "d_daynuminmonth", "d_daynuminyear", "d_monthnuminyear", "d_weeknuminyear",
                              "d_sellingseason", "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl", "null"])

# convert date type into int32
lineorder['lo_orderdate'] = lineorder['lo_orderdate'].astype('int32')
date['d_datekey'] = date['d_datekey'].astype('int32')

# filter out unnecessary data
filtered_lineorder = lineorder.query('lo_discount >= 1 & lo_discount <= 3 & lo_quantity < 25')
filtered_date      = date.query('d_year == 1993')

# check if there is NaN
print("lineorder contains NaN? {}".format(filtered_lineorder['lo_orderdate'].isnull().any()))
print("date contains NaN? {}".format(filtered_date['d_datekey'].isnull().any()))

filtered_lineorder = filtered_lineorder.reset_index(drop=True)
filtered_date = filtered_date.reset_index(drop=True)

matrix_M, matrix_N = len(filtered_lineorder), len(filtered_date)
print("M: {} N: {}".format(matrix_M, matrix_N))
print("lo_orderdate.nunique() {}, datekey.nunique() {}".format(filtered_lineorder['lo_orderdate'].nunique(), filtered_date['d_datekey'].nunique()))

# encode joined column to minimize the memory usage
le = LabelEncoder()

lo_uniques = filtered_lineorder['lo_orderdate'].unique()
d_uniques  = filtered_date['d_datekey'].unique()
uniques = np.unique(np.concatenate((lo_uniques,d_uniques),0))

le.fit(uniques)
filtered_lineorder['lo_orderdate'] = le.transform(filtered_lineorder['lo_orderdate'])
filtered_date['d_datekey'] = le.transform(filtered_date['d_datekey'])

# write out to table
filtered_lineorder = filtered_lineorder.iloc[: , :-1]
print("writing out lineorder table...")
filtered_lineorder.to_csv("lo_q1_test.tbl",sep='|', header=False, index=False, line_terminator='|\n')

filtered_date = filtered_date.iloc[: , :-1]
print("writing out date table...")
filtered_date.to_csv("d_q1_test.tbl",sep='|', header=False, index=False, line_terminator='|\n')
print("finished encoding and filtering.")