import numpy as np
import matplotlib.pyplot as plt
import pylab
from utils.getPose import vgg_X_from_xP_nonlin
from utils.parser import fullTest, checkIfPerm


class tempGraph:
    def __init__(self):
        self.frames=[]
        self.f = None
        self.mot = None
        self.str = None
        self.obsVal = None
        self.ObsIdx = None
        self.focal = np.array([1])
        self.denseMatch = None
        self.matches=None
        pass
    def closeEnought(self,other,tol):
        t1 = (self.frames == other.frames)
        t2 = (other.f == self.f)
        t3 = fullTest(self.mot, other.mot,tol)
        t4 = fullTest(self.str, other.str,tol)

        # Permite que los valores esten en otro orden, mientras sea un
        # Re ordenamiento de las filas
        t5 = checkIfPerm(self.obsVal, other.obsVal)

        # ObsIdx solo contiene indices , si se reordena estos indices cambian
        # Pero debe mantenerse que Ga.vals[gA.ObsIdx] = GB.vals[gB.ObsIdx]
        # Aun asi pueden venir desordenadas
        # TODO INEFINCIENTE A MEDIDA QUE AUMENTAN VALORES
        AllValuesFromIndexsA = self.obsVal[self.ObsIdx.astype(np.int), :]
        AllValuesFromIndexsB = other.obsVal[other.ObsIdx.astype(np.int), :]
        t6 = True
        for matchA in AllValuesFromIndexsA:
            fail = True
            for matchB in AllValuesFromIndexsB:
                if checkIfPerm(matchA, matchB):
                    fail = False
                    break
            if fail:
                t6 = False
        t7 = fullTest(self.focal, other.focal)
        return (t1 and t2 and t3 and t4 and t5 and t6 and t7)
    def __eq__(self, other):
        return self.closeEnought(other, 1e-02)
def createGraph(id1,id2,focal,pA,pB,Rt,f):
    graph = tempGraph()
    graph.frames = [id1,id2]
    graph.focal = focal
    graph.f = f
    graph.mot = np.zeros((3,4,2))
    n = pA.shape[0]

    graph.mot[:,:,0] = np.hstack([np.eye(3),np.zeros((3,1))])
    graph.mot[:,:,1] = Rt

    graph.str = np.zeros((n,3))
    graph.matches = np.hstack([pA, pB])
    graph.obsVal = np.vstack([pA,pB])

    graph.ObsIdx = np.zeros((n,2))
    graph.ObsIdx[:,0] = range(n)
    graph.ObsIdx[:,1] = range(n,2*n)

    return graph


def triangulateGraph(graph,imagesize):
    newGraph = graph
    n = newGraph.str.shape[0]
    X = np.zeros((n,4))
    colIms = np.array([imagesize[1], imagesize[0]]).reshape((2, 1))
    imsize = colIms.repeat(len(graph.frames), axis=1)
    for i in range(n):
        validCamera = np.where(graph.ObsIdx[i] != -1)[0]
        P = np.zeros((3,4,validCamera.shape[0]))
        x= np.zeros((validCamera.shape[0],2))
        cnt=0
        #Consigue los puntos en el plano de la camara y la matriz de proyeccion
        for ind in validCamera:
            x[cnt,:] = newGraph.obsVal[newGraph.ObsIdx[i][ind],:]
            P[:,:,cnt] = np.dot(newGraph.focal,newGraph.mot[:,:,ind])
            cnt+=1

        X[i,:] = vgg_X_from_xP_nonlin(x,P,imsize,X=None)
    allscales = X[:,3].reshape((n, 1))
    newGraph.str = X[:,0:3] / np.hstack([allscales,allscales,allscales])
    return newGraph


def visualizeDense(listG,merged,imsize):
    #plotear merge


    ax = showGraph(merged, imsize,True)
    allPoints = np.empty((3,0))
    #plotear dense
    for g in listG:
        goodPoint = g.denseRepError < 0.05;
        ax.scatter(g.denseX[0,goodPoint], g.denseX[1,goodPoint], g.denseX[2,goodPoint])
        allPoints = np.hstack([allPoints,g.denseX[:, goodPoint]])
    allPoints = np.hstack([allPoints, np.transpose(merged.str)])
    allPoints = np.transpose(allPoints)
    plt.show()
    return allPoints

def showGraph(graph,imsize,getAxis=False):
    from mpl_toolkits.mplot3d import Axes3D


    fig = pylab.figure()
    ax = fig.gca(projection='3d')

    #dibujar camaras
    for i in range(graph.mot.shape[2]):
        V = getCamera(graph.mot[:, :, i], imsize[1], imsize[0], graph.f, 0.001)
        xi,yi,zi = V[0, [0, 4]], V[1, [0, 4]], V[2, [0, 4]]
        ax.plot(xi,yi,zi)
        xi,yi,zi = V[0, [0, 5]], V[1, [0, 5]], V[2, [0, 5]]
        ax.plot(xi,yi,zi)
        xi,yi,zi = V[0, [0, 6]], V[1, [0, 6]], V[2, [0, 6]]
        ax.plot(xi,yi,zi)
        xi,yi,zi = V[0, [0, 7]], V[1, [0, 7]], V[2, [0, 7]]
        ax.plot(xi,yi,zi)
        ax.plot(V[0, [4, 5,6,7,4]], V[1, [4, 5,6,7,4]], V[2, [4, 5,6,7,4]])

    ax.scatter(graph.str[:,0], graph.str[:,1], graph.str[:,2])

    if getAxis:
        return ax
    else:
        plt.show()

def getCamera(Rt, w, h, f, scale):
    V = np.array([
        [0, 0, 0, f, -(w * 0.5), (w * 0.5), (w * 0.5), -(w * 0.5)],
        [0, 0, f, 0, -(h * 0.5), -(h * 0.5), (h * 0.5), (h * 0.5)],
        [0, f, 0, 0, f, f, f, f]
        ])
    V = scale * V
    V = transformPtsByRt(V, Rt, True)
    return V


def transformPtsByRt(X3D, Rt, isInverse=True):
    repMat = np.repeat(Rt[:, 3, np.newaxis], X3D.shape[1], axis=1)

    if isInverse:
        Y3D = np.dot( np.transpose(Rt[:,0:3]) , (X3D - repMat ) )
    else:
        Y3D = np.dot( Rt[:,0:3] , X3D) + repMat
    return Y3D