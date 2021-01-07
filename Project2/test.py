# 1
name = input()
print('Hello, ', name, '!', sep='')

# 2
n = int(input())
print(str('   _~_    ') * n)
print(str('  (o o)   ') * n)
print(str(' /  V  \\  ') * n)
print(str('/(  _  )\\ ') * n)
print(str('  ^^ ^^   ') * n)

# 3
N = int(input())
K = int(input())
print(K // N)

# 4
N = int(input())
K = int(input())
print(K % N)

# 5
N = int(input())
print(2 ** N)

# 6
N = int(input())
print(N % 10)

# 7
N = int(input())
print(N // 10)

# 8
N = int(input())
print((N % 100) // 10)

# 9
N = int(input())
print((N // 100) + ((N % 100) // 10) + (N % 10))

# 10
print('A' * 100)

# 11 +
N = int(input())
print(N // 60, N % 60)

# 12 +
N = int(input())
A = int(input())
B = int(input())
C = A * 100 + B
print((C * N) // 100, (C * N) % 100)

# 13 +
N = int(input())
print("The next number for the number ", N, " is ", N + 1, '.', sep='')
print("The previous number for the number ", N, " is ", N - 1, '.', sep='')

# 14 +
n = int(input())
print(((n % 1) + (n // 1) - 1) * (-1))

# 15 +
n = int(input())
print((n + 2) - (n % 2))

# 16 +
n = str(input())
print(int(n * 100) ** 2)

# 17 +
v = int(input())
t = int(input())
print((v * t) % 109)

# 18 +
n = int(input())
m = (str(n % 3600 // 60 // 10) + str(n % 3600 // 60 % 10))
s = (str(n % 3600 % 60 // 10) + str(n % 3600 % 60 % 10))
print(n // 3600, m, s, sep=':')

# 19 +
h1 = int(input())
m1 = int(input())
s1 = int(input())
h2 = int(input())
m2 = int(input())
s2 = int(input())
res = (h1 * 3600 + m1 * 60 + s1) - (h2 * 3600 + m2 * 60 + s2)
print(int((res ** 2) ** (1 / 2)))

# 20 +
s1 = int(input())
s2 = int(input())
print(s2 // s1 + 1)

#21
H = int(input())
A = int(input())
B = int(input())

print(((h-a)//(a-b))+((h-a)%(a-b))+1)

#22
n = int(input())
a = n//1000
b = n%((n // 1000)*1000)//100
c = n%((n // 100)*100)//10
d = n%((n // 10)*10)
print(n,a,b,c,d)
print((a*10+b)//(c+d*10))

#23
