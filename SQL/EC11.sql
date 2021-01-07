select * from hr.employees;

select last_name, department_id, salary from hr.employees where salary = (select max(salary) from hr.employees group by department_id);

select max(salary), department_id from hr.employees group by department_id;

select last_name, department_id, salary from hr.employees a where a.salary = (select max(salary) from hr.employees b where a.department_id = b.department_id);

select count(employee_id), department_id from hr.employees group by department_id having count(employee_id) < 3;

select a.employee_id, a.department_id, a.manager_id, b.department_id 
from hr.employees a, hr.employees b 
where b.employee_id = a.manager_id and a.department_id != b.department_id;

select a.employee_id, a.department_id, a.manager_id
from   hr.employees a
left   join hr.employees b on (b.employee_id = a.manager_id and b.department_id = a.department_id)
where  b.employee_id is null;

select department_id max(sum(salary)) from hr.employees group by department_id;