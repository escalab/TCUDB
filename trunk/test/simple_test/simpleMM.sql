SELECT MAT1.j, MAT2.i, SUM(MAT1.val * MAT2.val)
FROM MAT1, MAT2
WHERE MAT1.i = MAT2.j
GROUP BY MAT1.j, MAT2.i;
