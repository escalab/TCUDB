/*
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/
select
  mat1.i
from  (select * from matrices where matrix_id = 1) as mat1
;
