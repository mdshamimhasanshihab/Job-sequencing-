import numpy as np
import pickle
n=int(input('Number of jobs:'))
m=int(input('Number of machines:'))

a=np.zeros((n,m),dtype=int)

for i in range(len(a)):
    for j in range(len(a[i])):
        x=input(f"Enter the value for a[{i+1},{j+1}]:")
        a[i][j]=x
print(f"\n\ngiven data--\n{a}")

p=0
h = np.zeros((n), dtype=int)

for l in range(n):
    print(f'\nIteration no-{l + 1}')
    pickle.dump(a,open("a.dat","wb"))
    b=a.min(axis=1)

    i=0
    j=0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] not in b:
                a[i][j]=10000

    w=pickle.load(open("a.dat","rb"))

    pickle.dump(w,open("w.dat","wb"))

    c=w.min(axis=0)

    i=0
    j=0
    for i in range(len(w)):
        for j in range(len(w[i])):
            if w[i][j] not in c:
                w[i][j]=10000


    for i in range(len(w)):
        for j in range(len(w[i])):
            if w[i][j] not in a:
                w[i][j]=10000


    d=w.min()

    x=pickle.load(open("a.dat","rb"))
    print(x)
    i=0
    j=0
    for i in range(len(w)):
        for j in range(len(w[i])):
            if (int(w[i][j]) == d):
                break
        if (int(w[i][j]) == d):
            break


    x = np.delete(x, i, axis=0)
    a = np.insert(x, i, 10000, axis=0)
    print(a)


    h[p] = i+1
    p += 1

print("\n",h)