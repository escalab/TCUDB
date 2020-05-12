/*
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,ddate
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;
*/
SELECT
  MAT3.j,
  MAT4.i,
  sum(MAT3.val * MAT4.val) as res
FROM MAT3, MAT4
WHERE MAT3.i = MAT4.j 
GROUP BY MAT3.j, MAT4.i;
