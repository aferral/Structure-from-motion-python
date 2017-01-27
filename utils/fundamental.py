
import cv2
import numpy as np

def default(pointsA,pointsB):
    return cv2.findFundamentalMat(pointsA, pointsB,param1=0.002)


def implementacionRansac(pointsA,pointsB):
    t = 0.002 #Distancia a la cual se considera outlier
    F, inliers = ransacfitfundmatrix(pointsA, pointsB, t )
    print "Inliners ",len(inliers)
    print 'Puntos totales ',len(pointsA)
    print 'Porcentaje de INLIERS ', len(inliers)*1.0 / len(pointsA)


    return F,inliers

def ransacfitfundmatrix(pA,pB,tolerancia):
    assert(pA.shape == pB.shape)

    #Normalizar de forma que el origen es cnetroide y distancia media del origen es sqrt(2)
    #Ademas se asegura que el parametro de escala es 1
    na,Ta = normalizeHomogeneous(pA)
    nb,Tb = normalizeHomogeneous(pB)

    #Puntos para realizar la estimacion de fundamental matrix
    s = 8

    #Mandar al algoritmo RANSAC (conseguir modelo con mas inliners)
    modeloF = fundamentalFit
    distFun = distanceModel
    isdegenerate = lambda x : False #Nada es degenerado

    #Agregar a la hstack en cada fila x1,x2 3+3 6 elementos por fila
    dataset = np.hstack([na,nb])
    inliners,M = ransac(dataset,modeloF,distFun,isdegenerate,s,tolerancia)

    F = fundamentalFit(np.hstack([na[inliners,:],nb[inliners,:]]))

    F = np.dot(np.dot(Tb, F), np.transpose(Ta))

    return F,inliners
def fundamentalFit(data):
    assert(data.shape[1] == 6 )

    p1,p2 = data[:,0:3],data[:,3:]
    n, d = p1.shape

    na,Ta = normalizeHomogeneous(p1)
    nb,Tb = normalizeHomogeneous(p2)

    p2x1p1x1 = nb[:,0] * na[:,0]
    p2x1p1x2 = nb[:,0] * na[:,1]
    p2x1 = nb[:, 0]
    p2x2p1x1 = nb[:,1] * na[:,0]
    p2x2p1x2 = nb[:,1] * na[:,1]
    p2x2 = nb[:,1]
    p1x1 = na[:,0]
    p1x2 = na[:,1]
    ones = np.ones((1,p1.shape[0]))

    A = np.vstack([p2x1p1x1,p2x1p1x2,p2x1,p2x2p1x1,p2x2p1x2,p2x2,p1x1,p1x2,ones])
    A = np.transpose(A)

    u, D, v = np.linalg.svd(A)
    vt = v.T

    F = vt[:, 8].reshape(3,3) #Conseguir el vector con menor valor propio y eso es F

    #Como la matriz fundamental es de rango 2 hay que volver a hacer svd y reconstruir
    #A partir de rango 2
    u, D, v = np.linalg.svd(F)
    F=np.dot(np.dot(u, np.diag([D[0], D[1], 0])), v)

    F= np.dot(np.dot(Tb,F),np.transpose(Ta))

    return F

    pass
def distanceModel(F, x, t):
    p1, p2 = x[:, 0:3], x[:, 3:]

    x2tFx1 = np.zeros((p1.shape[0],1))

    x2ftx1 = [np.dot(np.dot(p2[i], F), np.transpose(p1[i])) for i in range(p1.shape[0])]

    ft1 = np.dot(F,np.transpose(p1))
    ft2 = np.dot(F.T,np.transpose(p2))

    bestInliers = None
    bestF = None

    sumSquared = (np.power(ft1[0, :], 2) +
                  np.power(ft1[1, :], 2)) + \
                 (np.power(ft2[0, :], 2) +
                  np.power(ft2[1, :], 2))
    d34 = np.power(x2ftx1, 2) / sumSquared
    bestInliers = np.where(np.abs(d34) < t)[0]
    bestF = F
    return bestInliers,bestF

def ransac(x, fittingfn, distfn, degenfn, s, t):
    maxTrials = 2000
    maxDataTrials = 200
    p=0.99

    bestM = None
    trialCount = 0
    maxInlinersYet = 0
    N=1
    maxN = 120
    n, d = x.shape

    M = None
    bestInliners = None
    while N > trialCount:
        degenerate = 1
        degenerateCount = 1
        while degenerate:
            inds = np.random.choice(range(n),s,replace=False)
            sample = x[inds,:]
            degenerate = degenfn(sample)

            if not degenerate:
                M = fittingfn(sample)
                if M is None:
                    degenerate = 1
            degenerateCount +=1
            if degenerateCount > maxDataTrials:
                raise Exception("Error muchas sample degeneradas saliendo")
        #Evaluar modelo
        inliners,M = distfn(M,x,t)
        nInliners = len(inliners)

        if maxInlinersYet < nInliners:
            maxInlinersYet = nInliners
            bestM = M
            bestInliners = inliners

            #Estimacion de probabilidad trials hasta conseguri
            eps = 0.000001
            fractIn = nInliners*1.0/n
            pNoOut = 1 - fractIn*fractIn
            pNoOut = max(eps,pNoOut) #Evitar division por 0
            N = np.log(1-p) / np.log(pNoOut)
            N = max(N,maxN)

        trialCount +=1
        if trialCount > maxTrials:
            print("Se alcanzo maxima iteracion saliendo")
            break
    if M is None:
        raise Exception("Error no se encontro el modelo")
    print "Se realizacion ",trialCount,' itentos'
    return bestInliners,bestM




def normalizeHomogeneous(points):
    normPoints = []
    if points.shape[1] == 2:
        # Agrego factor de escala (concadenar columna con 1)
        points = np.hstack([points, np.ones((points.shape[0],1))])

    n = points.shape[0]
    d = points.shape[1]
    #Deja en escala 1
    factores = np.repeat((points[:, -1].reshape(n, 1)), d, axis=1)
    points = points / factores #NOTAR QUE ESTO ES POR ELEMENTO

    prom = np.mean(points[:,:-1],axis=0)
    newP = np.zeros(points.shape)
    #Dejar todas las dimensiones en promedio 0 (menos la de escala)
    newP[:,:-1] = points[:,:-1] - np.vstack([prom for i in range(n)])

    #Calcular distancia promedio
    dist = np.sqrt(np.sum(np.power(newP[:,:-1],2),axis=1))
    meanDis = np.mean(dist)
    scale = np.sqrt(2)*1.0/ meanDis

    T = [[scale,0,-scale*prom[0]],
         [0, scale, -scale * prom[1]],
         [0,    0,     1]
         ]
    #ESTA ES LA VERSION ORIGINAL QUE SE USABA T*points
    #Esta asume puntos DxN como se usan puntos en formato NxD
    #Se usa el transpuesto
    T = np.transpose(np.array(T))
    transformedPoints = np.dot(points,T)
    return transformedPoints,T

    pass