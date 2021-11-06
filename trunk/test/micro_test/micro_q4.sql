/*
Q4 -- aggregate without group-by 
*/
SELECT
  SUM(MATA.VAL*MATB.VAL)
FROM MATA, MATB
WHERE MATA.ID = MATB.ID;
