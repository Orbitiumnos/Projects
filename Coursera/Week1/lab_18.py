# 18 +
n = int(input())
d = (n // 3600 // 24)
h = (n // 3600 % 24)
m = (str(n % 3600 // 60 // 10) + str(n % 3600 // 60 % 10))
s = (str(n % 3600 % 60 // 10) + str(n % 3600 % 60 % 10))
print(h, m, s, sep=':')
