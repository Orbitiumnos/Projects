select * from sys.all_tables
where 1=1
--and owner in ('SYSTEM')
and table_name like '%DEPEN%'
;

alter session set "_ORACLE_SCRIPT"=true;



DROP TABLE spaces_dependence;
COMMIT;

alter session set "_oracle_script"=true;

select * --SID, SERIAL#, USERNAME 
from V$SESSION;

alter system kill session '383,32304' immediate;

DROP USER orbitiumnos CASCADE; 

commit;

CREATE TABLESPACE tbs_02 
   DATAFILE 'diskb:tbs_f5.dat' SIZE 500K REUSE
   AUTOEXTEND ON NEXT 500K MAXSIZE 100M;
	 



CREATE USER orbitiumnos 
    IDENTIFIED BY legomania97
    DEFAULT TABLESPACE tbs_02
    QUOTA 5M ON tbs_02
    TEMPORARY TABLESPACE temp
    QUOTA 5M ON system;

grant create session, create table to orbitiumnos;
commit;




CREATE TABLE SPACES_DEPENDENCE
(
father varchar2(30),
child varchar2(30),
status varchar2(10)
);

commit;

select * from spaces_dependence;
insert into spaces_dependence (father, child, status) values ('STG','ODS','OK');
insert into spaces_dependence (father, child, status) values ('ODS','CONV','OK');
insert into spaces_dependence (father, child, status) values ('ODS','SYN','OK');
insert into spaces_dependence (father, child, status) values ('CONV','DICT','OK');
insert into spaces_dependence (father, child, status) values ('DICT','LINK_APP_REQUEST','OK');
insert into spaces_dependence (father, child, status) values ('LINK_APP_REQUEST','DDS','ERROR');
insert into spaces_dependence (father, child, status) values ('DDS','CALC',null);
insert into spaces_dependence (father, child, status) values ('CALC','MARTS',null);
insert into spaces_dependence (father, child, status) values ('CALC','PDN',null);
insert into spaces_dependence (father, child, status) values ('CALC','CALC_LINK_APP_PREAPP',null);
insert into spaces_dependence (father, child, status) values ('MARTS','REPORTS',null);
commit;

select * from spaces_dependence for update;

CREATE TABLE reglament_status
(
stage varchar2(30),
status varchar2(10)
);

insert into reglament_status (stage, status) values ('STG','OK');
insert into reglament_status (stage, status) values ('ODS','OK');
insert into reglament_status (stage, status) values ('CONV','OK');
insert into reglament_status (stage, status) values ('SYN','OK');
insert into reglament_status (stage, status) values ('LINK_APP_REQUEST','ERROR');
insert into reglament_status (stage, status) values ('DDS',null);
insert into reglament_status (stage, status) values ('CALC',null);
insert into reglament_status (stage, status) values ('PDN',null);
insert into reglament_status (stage, status) values ('CALC_LINK_APP_PREAPP',null);
insert into reglament_status (stage, status) values ('MARTS',null);
insert into reglament_status (stage, status) values ('REPORTS',null);
insert into reglament_status (stage, status) values ('DICT','OK');
commit;
