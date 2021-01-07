CREATE TABLE nikolay_fedorov_5 as
SELECT nikolay_fedorov_4.uid, nikolay_fedorov_4.mclass, COUNT(*) FROM nikolay_fedorov_4
GROUP BY nikolay_fedorov_4.uid, nikolay_fedorov_4.mclass HAVING count(nikolay_fedorov_4.mclass)>9
ORDER BY nikolay_fedorov_4.uid ASC;


CREATE TABLE nikolay_fedorov_5 as
SELECT nikolay_fedorov_4.uid, nikolay_fedorov_4.class, nikolay_fedorov_4.mclass, COUNT(*) as cnt FROM nikolay_fedorov_4
GROUP BY nikolay_fedorov_4.uid, nikolay_fedorov_4.class, nikolay_fedorov_4.mclass HAVING count(nikolay_fedorov_4.mclass)>9
ORDER BY nikolay_fedorov_4.uid ASC;


CREATE TABLE nikolay_fedorov_6 as 
SELECT uid, 
count(case when class=1 then 1 else 0 end) as user_cat1_flag, 
count(case when class=2 then 1 else 0 end) as user_cat2_flag, 
count(case when class=3 then 1 else 0 end) as user_cat3_flag, 
count(case when class=4 then 1 else 0 end) as user_cat4_flag 
FROM nikolay_fedorov_5 GROUP BY uid ORDER BY uid ASC;



#user_cat1_flag, user_cat2_flag, user_cat3_flag, user_cat4_flag



SELECT uid, count(case when class=1 then 1 else 0 end) as cnt FROM nikolay_fedorov_5 limit 20;



SELECT uid, count(*) as cnt 
FROM nikolay_fedorov_6 
group by uid having count(*)>1
limit 20;




CREATE TABLE nikolay_fedorov_7 as 
SELECT uid, 
count(case when user_cat1_flag>0 then 1 else 0 end) as user_cat1_flag, 
count(case when user_cat2_flag>0 then 1 else 0 end) as user_cat2_flag, 
count(case when user_cat3_flag>0 then 1 else 0 end) as user_cat3_flag, 
count(case when user_cat4_flag>0 then 1 else 0 end) as user_cat4_flag 
FROM nikolay_fedorov_6 GROUP BY uid ORDER BY uid ASC;



INSERT OVERWRITE DIRECTORY 'hdfs://spark-master.newprolab.com:8020/user/nikolay.fedorov/lab03result'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
SELECT uid, user_cat1_flag, user_cat2_flag,user_cat3_flag, user_cat4_flag
FROM nikolay_fedorov_7
ORDER BY uid ASC;
