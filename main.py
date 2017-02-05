import os
import cv2
import numpy as np
from utils.bundleAjust import bundleAdjustment
from utils.dense import denseMatch, denseReconstruction, outputPly
from utils.fundamental import default, implementacionRansac
from utils.getPose import getPose
from utils.graph import createGraph, triangulateGraph, showGraph, visualizeDense
from utils.mergeGraph import mergeG, removeOutlierPts
from utils.paresDescript import getPairSIFT

#Creditos a % SFMedu: Structrue From Motion for Education Purpose
# % Written by Jianxiong Xiao (MIT License) el codigo se base en este


def mergeAllGraph(gL,imsize):
    graphMerged = gL[0]
    # merge de vistas parciales
    for i in range(len(gL) - 1):
        graphMerged = updateMerged(graphMerged, gL[i+1],imageSize)
    return graphMerged
def updateMerged(gA,gB,imsize):
    gt = mergeG(gA, gB)
    gt = triangulateGraph(gt, imsize)
    gt = bundleAdjustment(gt, False)
    gt = removeOutlierPts(gt, 10)
    gt = bundleAdjustment(gt)
    return gt

if __name__ == "__main__":

    #---------------------------SET PARAMETERS
    maxSize = 640 #maxima resolucion de imagen
    carpetaImagenes = 'example/'
    debug = True
    outName = "jirafa" #out name for ply file (open with mesh lab to see poitn cloud)
    validFile = ['jpg','png','JPG'] #tipos validos de imagenes
    # Intentar conseguir la distancia focal
    # TODO agregar calculo este valor deberia funcionar con imagenes 480x640 focalLen 4mm
    f = 719.5459

    # ---------------------------SET PARAMETERS


    algoMatrizFundamental = implementacionRansac

    graphList = []

    #Cargar imagenes
    listaArchivos = os.listdir(carpetaImagenes)
    listaImages = filter(lambda x : x.split('.')[-1] in validFile,listaArchivos )




    #Carga las imagenes
    listaImages = map(lambda x : cv2.imread(carpetaImagenes+x),listaImages)

    imageSize = listaImages[0].shape
    print "Dimensiones originales ",imageSize
    #todo Escala la imagen si pasa de maxSize
    if imageSize[0] > maxSize:
        print "Escalando"
        print "Size image ",imageSize," max size ",maxSize
        #480 640 funciona
        listaImages = map(lambda x: np.transpose(cv2.resize(x,(640,480)),axes=[1,0,2]), listaImages)
        imageSize = listaImages[0].shape
        print "Result size ",imageSize

    #calculo de matriz K
    K = np.eye(3)
    K[0][0] = f
    K[1][1] = f

    graphList = [0 for i in range(len(listaImages)-1)]
    #calcula pares a partir de SIFT u otro descriptor local
    #Se calculan como imagenes sucesivas
    print "Inicia calculo de pares SIFT"
    for i in range(len(listaImages)-1):
        keypointsA,keypointsB = getPairSIFT(listaImages[i],listaImages[i+1],show=debug)


        #Calcular la matriz fundamental o la matriz escencial
        #TODO conseguir las demas
        if type(keypointsA[0]) == np.ndarray:
            assert(len(keypointsA.shape) == 2)
            assert (len(keypointsB.shape) == 2)
            pointsA = keypointsA
            pointsB = keypointsB
        else:
            pointsA = np.array([(keypointsA[idx].pt) for idx in range(len(keypointsA))]).reshape(-1, 1, 2)
            pointsB = np.array([(keypointsB[idx].pt) for idx in range(len(keypointsB))]).reshape(-1, 1, 2)
        pointsA = pointsA[:,[1,0]]
        pointsB = pointsB[:, [1, 0]]

        F = np.array(algoMatrizFundamental(pointsA,pointsB))
        Fmat = F[0]
        K = np.array(K)
        E = np.dot(np.transpose(K),np.dot(Fmat,K))

        # Conseguir pose de las camaras
        Rtbest = getPose(E,K, np.hstack([pointsA,pointsB]),imageSize)

        #Crear grafico
        graphList[i] = createGraph(i,i+1,K, pointsA, pointsB, Rtbest, f)

        #Triangular
        graphList[i] = triangulateGraph(graphList[i],imageSize)

        #visualizar grafico
        # showGraph(graphList[i],imageSize)

        #Bundle ajustement
        graphList[i]=bundleAdjustment(graphList[i])

        #Visualiza con mejoras
        # showGraph(graphList[i], imageSize)

    gM = mergeAllGraph(graphList,imageSize)
    print "Merge de grafos finalizado"
    #Visualizar resultado parcial
    showGraph(gM,imageSize)
    #Dense matching
    for i in range(len(listaImages)-1):
        graphList[i] = denseMatch(graphList[i],listaImages[i],
                                  listaImages[i+1], imageSize, imageSize)

    print "Dense match finalizado"
    print "Inicializando dense Triangulation"
    #Dense reconstruction
    for i in range(len(listaImages) - 1):
        graphList[i] = denseReconstruction(graphList[i], gM,K,imageSize)
    print "Dense reconstruct finalizado"
    data = visualizeDense(graphList, gM, imageSize)

    outputPly(data,outName)












