/*
Q3 -- group-by aggregates over joins
*/
SELECT
  SUM(MATA.VAL),
  MATB.VAL
FROM MATA, MATB
WHERE MATA.ID = MATB.ID
GROUP BY MATB.VAL; 
