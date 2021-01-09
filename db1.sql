DROP SCHEMA "budget" CASCADE;
CREATE SCHEMA "budget";
COMMENT ON SCHEMA "budget" IS 'Для учета бухгалтерии';

create table "budget".transactions
(
id INT,
date_id DATE,
sum INT,
flow VARCHAR(25),
source VARCHAR(25),
target VARCHAR(25),
comment VARCHAR(100),
category VARCHAR(25)
);

INSERT INTO "budget".transactions values (1, '01.01.2021', 1, 'test','test','test','test','test');

SELECT * FROM "budget".transactions;

COMMIT;

SELECT * --DISTINCT TABLE_SCHEMA 
FROM information_schema.tables
--where UPPER(TABLE_SCHEMA) like '%BUDGET%'
;

DROP TABLE "public".transactions;

DROP TABLE "BUDGET".transactions;

DROP SCHEMA "BUDGET" CASCADE;

COMMIT;

