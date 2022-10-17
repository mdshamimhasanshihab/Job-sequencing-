import numpy as np
n=int(input('Number of row:'))
m=int(input ('Number of column:'))
a=np.zeros((n,m),dtype=float)
for i in range (len(a)):
    for j in range (len(a[i])):
        x=input(f'Enter the value of a[{i+1},{j+1}]:')
        a[i][j]=x
print(a)

print("select an option:\n1.maximization\n2.minimization")
w=int(input())
while True:

    if (w==1):
        b=a[0].min()
    elif (w==2):
        b = a[0].max()

    else:
        print('Wrong input')
        break

    if (w==1):

        if (a[0].min()>=0):
            break
    else:
        if (a[0].max()<=0):
            break


    print(f'\n\n\nnew iteration:')

    i = 0
    for i in range(len(a[1])):

        if ((a[0][i]) == b):
            break
    q=i

    c=a[:,i]
    print(f'Entering column:{c}')
    d1 = a[:, (m - 1)]

    d = d1 / c
    print(f"solution column:{d}")

    i = 0
    p = d[1:]
    p = p[p > 0]

    e = p.min()
    i = 0
    for i in range(len(d)):
        if (d[i] == e):
            break

    print(f'element no:{i+1}')
    print(f'pivot row:{a[i, :]}')
    r=i

    for j in range(n):

        f = (a[i, :]/a[r][q]) * (-c[j])
        # print(f)
        if j == r:
            g = a[i, :]/a[r][q]


            a = np.delete(a, j, axis=0)
            a = np.insert(a, j, g, 0)
            continue
        g = f + a[j]


        #print(g)
        a = np.delete(a, j, axis=0)
        a = np.insert(a, j, g, 0)
    print(f"\n\n{a}")
print(f'z value= {a[0][m-1]}')

