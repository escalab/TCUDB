select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_yearmonthnum = 199401
and lo_discount>=4
and lo_discount<=6
and lo_quantity>=26
and lo_quantity<=35;

