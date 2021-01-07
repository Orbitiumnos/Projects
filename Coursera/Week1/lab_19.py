# 19 +
h1 = int(input())
m1 = int(input())
s1 = int(input())
h2 = int(input())
m2 = int(input())
s2 = int(input())
res = (h1 * 3600 + m1 * 60 + s1) - (h2 * 3600 + m2 * 60 + s2)
print(int((res ** 2) ** (1 / 2)))
