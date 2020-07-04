import os

print("Choose the ratio in order to split data:/n1. 20/80/n2. 30/70/nInput 1 or 2/n")

k = input("Input ratio /n")

if(k==1):
    k=0.8
    u=0.2
else:
    k=0.7
    u=0.3

objects_test = []

#move to test
os.chdir('C:/Users/Yash Vardhan Singal/Desktop/MAJOR/data/test/')

objects_test = os.listdir(".")

n = len(objects_test)

print(objects_test)

print(n)

for i in range(n):
    os.chdir("C:/Users/Yash Vardhan Singal/Desktop/MAJOR/data/test/" + objects_test[i])

    m = len(str(n))

    for x in objects_test:
        for i in range(1,n*k):
            sno = "0" * (m-len(str(i)))
            os.remove(str(x)+"_"+sno+".jpg")

objects_train = []

#move to train
os.chdir("C:/Users/Yash Vardhan Singal/Desktop/MAJOR/data/train/")

objects_train = os.listdir(".")

q = len(objects_train)

for i in range(q):
    os.chdir("C:/Users/Yash Vardhan Singal/Desktop/MAJOR/data/train/" + str(objects_train[i]))

    m = len(str(q))

    for x in objects_train:
        for i in range(1,q*u):
            sno = "0" * (m-len(str(i)))
            os.remove(str(x)+"_"+sno+".jpg")
