create table brand
(
BrandId int,
BrandName varchar2(10),
BrandCountry varchar2(10)
);

create table auto
(
AutoId int,
AutoModel varchar2(10),
BrandId int,
Price int
);

commit;

select * from brand for update;

select * from auto 
--for update
;

select * from brand t1
left join auto t2
on t1.brandid = t2.brandid
;


-- 1

/*
��������� ����� �� �������������� ������ � ��������� Brand - ����� �����������, 
Auto - ��������� ��� ������� ����������, ��������� ������ ��� ������ ���������� � ����� ��������� ����������� ������ ����� 
(� ������������ ������ ������� � ������ ����� �� ���� ����������� ���������� �����)
*/

select t1.brandid, t1.brandname, count(autoid) as cnt_auto, nvl(sum(price),0) as sum_price
from brand t1
left join auto t2
on t1.brandid = t2.brandid
group by t1.brandid, t1.brandname
order by 3 desc,4 desc
;

-- 2

/*
��� ����������� � ������ 1 ����� ������ ��������� ������� ��� �����������:
1. ����� ���������� � ����� ������� ������� ���������� ����������� ���� �����
2. ���������� �������� �����������
3. ������ ����� ������� ������� ����������� ������� ������
*/

-- 2.1

select brandname, avg_sum as max_avg_sum from 
(
select t1.brandid, t1.brandname, nvl(avg(price),0) as avg_sum
from brand t1
left join auto t2
on t1.brandid = t2.brandid
group by t1.brandid, t1.brandname
order by 3 desc
)
where rownum = 1
;

-- 2.2

select t1.brandcountry, count(autoid) as cnt_auto
from brand t1
left join auto t2
on t1.brandid = t2.brandid
where brandcountry = 'Germany'
group by t1.brandcountry
order by 2 desc
;

-- 2.3

select t1.brandname, t2.automodel as most_expensive_model
from brand t1
left join
(
select brandid, automodel, price, 
rank() over(partition by brandid order by price desc) price_rank
from auto
group by brandid, automodel, price
order by 1 asc, 2 asc
) t2
on t1.brandid = t2.brandid
where price_rank = 1
;
