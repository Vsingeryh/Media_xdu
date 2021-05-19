import numpy as np
from sklearn.decomposition import PCA
import os

def dataprocess(filepath='./ColorHistogram.asc'):
    A = np.zeros((68040, 33))
    f = open(filepath)
    lines = f.readlines()
    A_row = 0
    for line in lines:
        list1 = []
        list = line.strip('\n').split(' ')
        n = map(float, list)
        for i in n:
            list1.append(i)
        A[A_row] = np.array(list1)
        A_row+=1

    A = A[:, 1:]
    return A

Data=dataprocess()

#标准值
pca = PCA(n_components=5)
pca.fit(Data)
Std_pca=pca.transform(Data)

#my_pca
def pca(data,k):
    X_mean = np.mean(data, axis=0)
    sdata=(data-X_mean)
    ew,ev=np.linalg.eig(sdata.T.dot(sdata))
    ew_order=np.argsort(ew)[::-1]#从大到小
    ew_sort=ew[ew_order]
    ev_sort=ev[:,ew_order]
    feature=ev_sort[:,:k]
    new_data=sdata.dot(feature)
    return new_data

new_data=pca(Data,5)
print('原矩阵协方差矩阵：\n',np.cov(Data.T))
print('降维后矩阵协方差矩阵：\n',np.cov(new_data.T))
for i in range(20):
    print("std:",Std_pca[i])
    print("my :",new_data[i])
