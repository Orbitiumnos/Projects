n = int(input())
a = n//1000
b = n%((n // 1000)*1000)//100
c = n%((n // 100)*100)//10
d = n%((n // 10)*10)
print(n,a,b,c,d)
print((a*10+b)//(c+d*10))
