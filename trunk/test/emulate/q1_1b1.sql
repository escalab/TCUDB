select sum(lo_extendedprice*lo_discount)
--from mata, matb
from lineorder, ddate
where lo_orderdate = d_datekey;
