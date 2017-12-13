# encoding:utf-8

'''
根据用户购买药品在只是苦衷对应的疾病和症状，对用户进行聚类
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 主成分分析

'''
对每一个用户建立一个向量， 疾病种类是a，症状种类是b， 向量维度是a+b
长度的话 用累加值 然后再除以向量所有元素累加值总和 得到的概率参与计算
(这里用不到)
'''
def loadSet(address):
 s = pd.read_csv(address)  # s for sale
 s.columns = ["MEM_ID", "HEALTH_CARE_TAG",
 "RHEUMATISM_TAG",
 "HEPATOBILIARY_TAG",
 "HYPERTENSION_TAG",
 "HYPERLIPEMIA_TAG",
 "URINARY_SYSTEM_TAG",
 "MATERNAL_BABY_TAG",
 "DIABETES_TAG"]

 for column in s:
  column= pd.to_numeric(column, errors='coerce')

 return s

'''
输入：k, data[n];
（1） 选择k个初始中心点，例如c[0]=data[0],…c[k-1]=data[k-1]；
（2） 对于data[0]….data[n]，分别与c[0]…c[k-1]比较，假定与c[i]差值最少，就标记为i；
（3） 对于所有标记为i点，重新计算c[i]={ 所有标记为i的data[j]之和}/标记为i的个数；
（4） 重复(2)(3)，直到所有c[i]值的变化小于给定阈值。
n_init = 10: 选10次不同的initial centriods
max_iter = 300: 每次run都iterate 100次
tol = 1e-04 (=0.0001) 当SSE的变化在0.0001以内就看做是达到convergence（in case有的run达不到convergence）
'''
def Kmeans(data):
    km = KMeans(n_clusters= 5,init = 'random', n_init = 10)
    # , max_iter = 300,tol = 1e-04,random_state = 0
    # Compute clustering and transform X to cluster-distance space.
    new = km.fit_transform(data)
    print new
    # 每个mem对应的cluster [0 0 1 ..., 0 0 0]
    print km.labels_
    # 每个center的坐标
    # print  km.cluster_centers_

    return new, km.labels_


'''降维画图看出用几个centriod合适 再取消降维'''
def pca(data):
    pca = PCA(n_components=2)
    newData = pca.fit_transform(data)
    return newData

#用降维后的数据聚类
def pca_k(trans):
  # transp is an array
  transp = pca(trans)
  list1 = []
  list2 = []
  list3 = []
  for i in range(len(transp)):
    sub = transp[i]
    list1.append(sub[0])
    list2.append(sub[1])
  new, labels = Kmeans(transp)
  # plt.scatter(list1, list2, c=labels)
  return labels

def main():
    address = '/Users/Evangeline0519/PycharmProjects/Kmeans/simple mem.csv'
    m = loadSet(address)
    trans = m.drop('MEM_ID',1)
    # represent the labels of each element in the matrix
    labels = pca_k(trans)
    # print labels
    # 降维后画图发现有6个cluster 所以用6
    # new, labels = Kmeans(trans)
    # plt.show()

    return

if __name__ == '__main__':
    main()

    '''[0 0 1 ..., 2 0 0]'''