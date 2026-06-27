-- Lists records with a score of 10 or more in second_table, best score first
SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;
