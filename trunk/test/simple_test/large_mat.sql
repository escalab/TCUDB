/*
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/
SELECT
  LARGE_MAT1.i,
  LARGE_MAT2.j,
  sum(LARGE_MAT1.val * LARGE_MAT2.val) as res
FROM LARGE_MAT1, LARGE_MAT2
WHERE LARGE_MAT1.j = LARGE_MAT2.i 
GROUP BY LARGE_MAT1.i, LARGE_MAT2.j;
