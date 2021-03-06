import numpy as np

len1 = int(input())
x_new = np.zeros([len1, len1])
print(x_new)

data = list(map(int, input().split()))
c = 0
for i in range(x_new.shape[0]):
    for j in range(x_new.shape[1]):
        x_new[i][j] = data[c]
        c = c + 1
print(x_new)

m = x_new.shape[0] * x_new.shape[1]
x_vec = np.zeros(m)

c = 0
for i in range(x_new.shape[0]):
    for j in range(x_new.shape[1]):
        x_vec[c] = x_new[i][j]
        c = c + 1
print(x_vec)
print()

if len(x_vec.shape) != 1:
    print("The input is not vector")
else:
    w = np.zeros([len(x_vec), len(x_vec)])

    for i in range(len(x_vec)):
        for j in range(i, len(x_vec)):
            if i == j:
                w[i, j] = 0
            else:
                w[i, j] = x_vec[i] * x_vec[j]
                w[j, i] = w[i, j]
print(w)