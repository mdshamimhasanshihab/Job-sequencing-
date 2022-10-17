import numpy as np
n=int(input('Number of jobs:'))
m=int(input('Number of machines:'))

a=np.zeros((n,m),dtype=int)

for i in range(len(a)):
    for j in range(len(a[i])):
        x=input(f"Enter the value for a[{i+1},{j+1}]:")
        a[i][j]=x
print(a)
b=a.min(axis=0)
c=a.max(axis=0)
c[0]=0
c[m-1]=0
d=int(c.max())
if (b[0]>=d or b[m-1]>=d):
    print("\nIt is possible to iteration\n")
g=np.zeros((n,2),dtype=int)
for i in range(n):
    k=a[i].sum()-a[i][m-1]
    g[i][0]=int(k)
for i in range(n):
    k=a[i].sum()-a[i][0]
    g[i][1]=int(k)
print(g)

h = np.zeros((n), dtype=int)
p = 0
q = 0
for l in range(n):
    print(f'\nIteration no-{l+1}')
    z = int(g.min())
    for i in range(len(g)):
        for j in range(len(g[i])):
            if (int(g[i][j]) == z):
                break
        if (int(g[i][j]) == z):
            break

    g = np.delete(g, i, axis=0)
    g = np.insert(g, i, 10000, axis=0)
    print(g)

    if j == 0:
        h[p] = i+1
        p += 1
    else:
        q += 1
        h[n - q] = i + 1
print('\n',h)