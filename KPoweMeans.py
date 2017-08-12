# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:53:29 2017
KpowerMeans
@author: Zhiming
"""

#coding=utf-8
from numpy import *

#数据转换成向量 /delay/EoA/AoA/EoD/AoD/Power/

    
#计算两个向量的距离，用的是MCD
def MCD(vA, vB,tmx):
    mcd1=0.5*sum(array([[sin(vA[1])*cos(vA[2])-sin(vB[1])*cos(vB[2])],[sin(vA[1])*cos(vA[2])-sin(vB[1])*cos(vB[2])],[cos(vA[1])-cos(vB[1])]])**2)**0.5
    mcd2=0.5*sum(array([[sin(vA[3])*cos(vA[4])-sin(vB[3])*cos(vB[4])],[sin(vA[3])*cos(vA[4])-sin(vB[3])*cos(vB[4])],[cos(vA[3])-cos(vB[4])]])**2)**0.5
    mcd3=(vA[0]-vB[0])/tmx
    return sum(mcd1**2+mcd2**2+mcd3**2)**0.5
#随机生成初始的质心（ng的课说的初始方式是随机选K个点）    
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(array(dataSet)[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids
    
def kPowerMeans(dataSet, k, distMeas=MCD, createCent=randCent):
    m = shape(dataSet)[0]
    tmx=max(dataset[:,1])-min(dataset[:,1])
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = dataSet[i,-1]*distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print (centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster .A 转换成array类
            centroids[cent,:] = sum(multiply(ptsInCluster,ptsInCluster[:,-1])，axis=0)/sum(ptsInCluster)[0,-1]                  #mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment
    
def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt  
    numSamples, dim = dataSet.shape  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    for i in xrange(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
    plt.show()
      
def main():
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing= kMeans(dataMat,4)
    print myCentroids
    show(dataMat, 4, myCentroids, clustAssing)  
    
    
if __name__ == '__main__':
    main()