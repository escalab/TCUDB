/*
Initialization for PageRank
change number_of_node
*/
SELECT
  NODE.ID,
  (1-0.85)/1024 as degree
FROM NODE, OUTDEGREE
WHERE NODE.ID = OUTDEGREE.ID;
