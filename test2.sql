CREATE TABLE TEST_TABLE (
    id integer,
    value integer,
    descript varchar(20)
    );
    
INSERT INTO TEST_TABLE VALUES (1, 20, 'Val1');
INSERT INTO TEST_TABLE VALUES (2, 500, 'Val2');
INSERT INTO TEST_TABLE VALUES (3, 30, 'Val3');
INSERT INTO TEST_TABLE VALUES (4, 60, 'Val4');
INSERT INTO TEST_TABLE VALUES (5, 100, 'Val5');
INSERT INTO TEST_TABLE VALUES (6, 60, 'Val6');
INSERT INTO TEST_TABLE VALUES (7, 100, 'Val7');
INSERT INTO TEST_TABLE VALUES (8, 60, 'Val8');
INSERT INTO TEST_TABLE VALUES (9, 30, 'Val9');

SELECT * FROM TEST_TABLE;

-- 1
SELECT CAST(SUBSTR(DESCRIPT,4,1) AS INT) FROM TEST_TABLE;
SELECT REGEXP_SUBSTR(DESCRIPT,'(\d)') FROM TEST_TABLE;

-- 2
WITH A AS (
SELECT SUM(VALUE) AS SUM_RES 
FROM TEST_TABLE
)
SELECT (VALUE/A.SUM_RES) AS RES
FROM TEST_TABLE, A
ORDER BY (VALUE/A.SUM_RES);

-- 3
SELECT * FROM TEST_TABLE
WHERE VALUE IN (
    SELECT VALUE
    FROM TEST_TABLE
    GROUP BY VALUE HAVING COUNT(ID) > 1
    )
;

-- 4
SELECT * FROM TEST_TABLE
WHERE ID NOT IN (
    SELECT MIN(ID) FROM TEST_TABLE
    GROUP BY VALUE
    )
;

-- 5
SELECT MIN(ID) 
FROM TEST_TABLE
WHERE ID NOT IN (
    SELECT MIN(ID) FROM TEST_TABLE
    GROUP BY VALUE 
    --HAVING COUNT(ID) > 0
    )
GROUP BY VALUE 
--HAVING COUNT(ID) > 0
;

-- 6
SELECT MEDIAN(VALUE) FROM TEST_TABLE;

-- 7
SELECT EXP(SUM(LN(value))) from TEST_TABLE;

-- 8
WITH T1 AS (SELECT ID, VALUE FROM TEST_TABLE 
WHERE VALUE IN (SELECT MAX(VALUE) FROM TEST_TABLE))
SELECT T1.ID, T1.VALUE, T2.ID, T2.VALUE FROM TEST_TABLE T2, T1
WHERE ID IN (SELECT * FROM TEST_TABLE ORDER BY ROWNUM DESC
FETCH FIRST 3 ROWS ONLY)
;

SELECT MAX(VALUE) FROM TEST_TABLE
ORDER BY ROWNUM DESC
FETCH FIRST 3 ROWS ONLY;