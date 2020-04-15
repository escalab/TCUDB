select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_weeknuminyear = 6
and d_year = 1994
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;

