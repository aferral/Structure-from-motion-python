
from scipy.spatial import KDTree
import numpy as np
def matchSIFTdescr(descri, descrj, tolerancia=0.36):
    #La entrada de datos es en formato N,K
    kdTreeI = KDTree(descri)
    kdTreeJ =  KDTree(descrj)

    matchedI = []
    matchedJ = []
    for i in range(len(descri)):
        descriptor = descri[i]
        darrayJ,indexArrayJ = kdTreeJ.query(descriptor,k=2,p=2)
        realD1 = np.linalg.norm(descriptor-descrj[indexArrayJ[0]])**2
        realD2 = np.linalg.norm(descriptor - descrj[indexArrayJ[1]]) ** 2

        if realD1 < tolerancia*realD2:
            candidato = descrj[indexArrayJ[0]]
            darrayI, indexArrayI = kdTreeI.query(candidato, k=2,p=2)
            realD1 = np.linalg.norm(candidato - descri[indexArrayI[0]]) ** 2
            realD2 = np.linalg.norm(candidato - descri[indexArrayI[1]]) ** 2
            if (indexArrayI[0] == i) and (realD1 < tolerancia*realD2) :
                matchedI.append(i)
                matchedJ.append(indexArrayJ[0])
    return matchedI,matchedJ
