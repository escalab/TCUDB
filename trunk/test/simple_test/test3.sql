/*
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/
SELECT
  MAT1.i,
  MAT2.j,
  sum(MAT1.val * MAT2.val) as res
FROM MAT1, MAT2
WHERE MAT1.j = MAT2.i 
GROUP BY MAT1.i, MAT2.j;
