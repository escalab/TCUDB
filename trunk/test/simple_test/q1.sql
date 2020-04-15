/*
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/
select
  matrix_1.i,
  matrix_2.j,
  sum(matrix_1.val * matrix_2.val) AS val
from (select * from matrices where matrix_id = 1) matrix_1
JOIN (select * from matrices where matrix_id = 2) matrix_2
  ON matrix_1.j = matrix_2.i
group by matrix_1.i, matrix_2.j;
