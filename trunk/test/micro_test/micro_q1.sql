/*
Q1 -- two-way natural join
*/
SELECT
  MATA.VAL,
  MATB.VAL
FROM MATA, MATB
WHERE MATA.ID = MATB.ID; 
