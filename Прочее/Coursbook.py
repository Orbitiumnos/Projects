
# coding: utf-8

# In[1]:


#sample
for i in range(1, 11):
    for j in range(1, 11):
        print(i * j, end=' ')
    print()


# In[13]:


#5_ryad_1
a = int(input())
b = int(input())
for i in range (a,b+1):
    print(i, end = ' ')


# In[17]:


#5_ryad_2
a = int(input())
b = int(input())
if a<=b:
    for i in range (a,b+1,1):
        print(i, end = ' ')
else:
    for i in range (a,b-1,-1):
        print(i, end = ' ')


# In[12]:


#5_ryad_3
n = int(input())

for i in range(10**(n),0,-1):
    if i%2 != 0:
        print(i, end = ' ')


# In[18]:


#5_summ_kvad
n = int(input())
sum = 0

for i in range(1,n+1,1):
    sum = sum + i**2

print(sum)


# In[55]:


#5_flagi
n = int(input())
i = 1

flag = ('+___ ','|3 / ','|__\ ','|    ')

for j in range(4):
        if j == 1:
            for i in range(n):
                print('|'+str(i+1)+' /', end = ' ')
            print('')
        else:
            print(flag[j] * n)


# In[6]:


#5_colvo nuley

n = int(input())
i = 0
tupl = ()

while i != n:
    num = str(input())
    tupl = tupl + tuple(num)
    i = i+1

num = 0

for i in range(n):
    if int(tupl[i]) == 0:
        num = num + 1
        
print(num)


# In[13]:


#5_lesenka

n = int(input())
for j in range(n+1):
    for i in range(j):
        print(i+1, end='')
    print('')


# In[17]:


#5_zamech_chisla_1
for i in range(10,100,1):
    #print(i)
    if i == 2*(i//10)*(i%10):
        print(i)


# In[7]:


#5_diafant_2
a = int(input())
b = int(input())
c = int(input())
d = int(input())
e = int(input())
res = 0

for x in range(0,1001,1):
    if x != e:
        if (a*(x**3)+b*(x**2)+(c*x)+d)/(x-e) == 0:
            res = res+1
print(res)


# In[31]:


#5_sum_of_fact
def fac(n):
    if n == 0:
        return 1
    return fac(n-1) * n

x = int(input())
s = 0

for i in range(1,x+1):
    s = s+fac(i)
print(s)


# In[37]:


#5_sum_of_fact -- просто офигенная рекусрсия в рекурсии
def fac(n):
    if n == 0:
        return 1
    return fac(n-1) * n

def s(n):
    if n == 0:
        return 0
    return s(n-1) + fac(n)

x = int(input())
print(s(x))


# In[45]:


#5 карточки - офигенное решение поиска отсутствующего числа из набора чисел
n = int(input())

def fac(n):
    if n == 0:
        return 1
    return fac(n-1) * n

res = fac(n)

for i in range(1,n):
    num = int(input())
    res = res/num
print(int(res))


# In[61]:


#5 Замечательные числа - 4
A = int(input())
B = int(input())

for i in range(A,B+1):
    if i == ((i%10%10%10)*1000 + (i%100//10)*100 + (i%1000//100)*10 + (i//1000)):
        print(i)


# In[21]:


#5 Четные индексы
A = list(map(int,input().split()))
B=[]
for i in range(0,len(A),2):
    B.append(A[i])  
print(' '.join(list(map(str,B))))


# In[22]:


#5 Четные элементы
A = list(map(int,input().split()))
B=[]
for i in range(0,len(A)):
    if A[i]%2==0:
        B.append(A[i])  
print(' '.join(list(map(str,B))))


# In[28]:


#5 Количество положительных
A = list(map(int,input().split()))
B=[]
for i in range(0,len(A)):
    if A[i]>=0:
        B.append(A[i])  
print(len(B))


# In[40]:


#5 Последний максимум
def findmax(x,i,maxn,maxi):
    #print(x,i,maxn)
    if maxn < x:
        maxn = x
        maxi = i
    return (maxn, maxi)

A = list(map(int,input().split()))
maxn = 0
maxi = 0

for i in range(0,len(A)):
    maxn, maxi = findmax(A[i],i,maxn, maxi)
print(maxn, maxi)

#5 Больше предыдущего
A = list(map(int,input().split()))
B = []
for i in range(1,len(A)):
    if A[i]>A[i-1]:
        B.append(A[i])
print(' '.join(map(str,B)))

#5 Возрастает ли список?
A = list(map(int,input().split()))
res='YES'
for i in range(1,len(A)):
    if A[i] <= A[i-1]:
        res = 'NO'
print(res)

#5 Соседи одного знака
A = list(map(int,input().split()))
for i in range(1,len(A)):
    if min(A[i],A[i-1])>0:
        print(A[i-1],A[i])
        break

#5 Больше своих соседей
A = list(map(int,input().split()))
res = 0
for i in range(1,len(A)-1):
    if A[i] == max([A[i-1],A[i],A[i+1]]):
        res = res + 1
print(res)

#5 Наибольший элемент
A = list(map(int,input().split()))
res = 0
for i in range(1,len(A)-1):
    if A[i] == max(A):
        print(A[i],i)

#5 Наименьший положительный
A = list(map(int,input().split()))
B = []
for i in range(0,len(A)):
    if A[i]>=0:
        B.append(A[i])
print(min(B))

#5 Наименьший нечетный
A = list(map(int,input().split()))
B = []
for i in range(0,len(A)):
    if A[i]%2 != 0:
        B.append(A[i])
print(min(B))

#5 Вывести в обратном порядке
A = list(map(int,input().split()))
B = []
for i in range(len(A)-1,-1,-1):
    B.append(A[i])
print(' '.join(map(str,B)))

#5 Переставить в обратном порядке
A = list(map(int,input().split()))
n = len(A)
for i in range(0,int(n/2),1):
    back = A[n-1-i]
    A[n-1-i] = A[i]
    A[i] = back
print(' '.join(map(str,A)))

#5 Удалить элемент
A = list(map(int,input().split()))
inp = int(input())
for i in range(inp,len(A)-1):
    A[i] = A[i+1]
    i = i+1
A.pop()
print(' '.join(map(str,A)))

#5 Вставить элемент
A = list(map(int,input().split()))
k = int(input())
C = int(input())
A.append(C)
for i in range(len(A)-1,k,-1):
    back = A[i-1]
    A[i-1] = A[i]
    A[i] = back
print(A)

#5 Ближайшее число
A = list(map(int,input().split()))
k = int(input())
minl = 10000

for i in range(len(A)):
    if abs(A[i]-k) < minl:
        minl = abs(A[i]-k)
        res = A[i]
print(res)

#5 Шеренга
A = list(map(int,input().split()))
k = int(input())
minl = 10000

for i in range(len(A)):
    if abs(A[i]-k) < minl:
        minl = abs(A[i]-k)
        mini = i+1
A.insert(mini,k)
print(' '.join(map(str,A)))

#5 Количество различных элементов
A = list(map(int,input().split()))
res = 0
for i in range(len(A)):
    if A.count(i) != 0:
        res = res + 1
print(res)

#5 Переставить соседние
A = list(map(int,input().split()))
for i in range(0,len(A)//2*2,2):
    back = A[i]
    A[i] = A[i+1]
    A[i+1] = back
print(' '.join(map(str,A)))

#5 Цикличный сдвиг вправо
A = list(map(int,input().split()))
A.insert(0,A[len(A)-1])
A.pop()
print(*A)

#5 Переставь местами max и min
A = list(map(int,input().split()))

mini = A.index(min(A)) 
maxi = A.index(max(A))

A[mini], A[maxi] = A[maxi], A[mini]
print(A)

#5 Наибольшее произведение двух чисел
A = list(map(int,input().split()))

min1 = A.pop(A.index(min(A)))
max1 = A.pop(A.index(max(A)))
min2 = min(A)
max2 = max(A)

print(min1,min2,max2,max1)
if min1*min2 > max1*max2:
    print(min1, min2)
else:
    print(max2, max1)

#5 Наибольшее произведение трех чисел
A = list(map(int,input().split()))

min1 = A.pop(A.index(min(A)))
max1 = A.pop(A.index(max(A)))
min2 = min(A)
max2 = max(A)

print(min1,min2,max2,max1)
if min1*min2 > max1*max2:
    print(min1, min2)
else:
    print(max2, max1)

