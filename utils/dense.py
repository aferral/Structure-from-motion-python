from test.dataMergeTest import *
from utils.getPose import vgg_X_from_xP_nonlin
from utils.graph import visualizeDense
from utils.mergeGraph import dictToGraph
from utils.parser import parseStruct, pst, fullTest
import cv2
import numpy as np
import pylab as plt
import heapq
from plyfile import PlyData, PlyElement

class dummyHeap:
    def __init__(self,n):
        self.list = []
    def push(self,idx,val):
        heapq.heappush(self.list,(-val,idx))
    def pop(self):
        val,idx = heapq.heappop(self.list)
        return (-val,idx)
    def size(self):
        return len(self.list)

def denseMatch(graph,imA,imB,imSA,imSB):

    if imA.shape != imSA:
        cv2.resize(imA,imSA)
    if imB.shape != imSB:
        cv2.resize(imB,imSB)

    halfsizeprop = 2
    tempIm = [imA,imB]
    matchs = []
    znccs = []
    for i in range(2):
        #Aqui deberian ser imagenes en rango 0 1 (cv2 deja en 0-256)
        imt = tempIm[i].astype(np.float64) / 256
        if imt.shape[2] == 3:
            imt = toGrayScale(imt)
        matchs.append(reliableArea(imt))
        znccs.append(ZNCCpath_all(imt,halfsizeprop))
    hh = imA.shape[0] * 0.5
    hw = imA.shape[1] * 0.5
    points = int(graph.matches.shape[0])
    x1p = graph.matches[:,0].reshape((points,1))
    y1p = graph.matches[:,1].reshape((points,1))
    x2p = graph.matches[:,2].reshape((points,1))
    y2p = graph.matches[:,3].reshape((points,1))

    initMatch = np.round(np.hstack([
        hh - y1p -1,
        hw - x1p -1,
        hh - y2p -1,
        hw - x2p -1,
        np.zeros((points,1))
    ]))
    matched_pair = propagate(initMatch,matchs[0],matchs[1],znccs[0],znccs[1],halfsizeprop)
    graph.denseMatch = matched_pair[0]
    graph.denseMatch[:,0] = (imA.shape[0] * 0.5) - graph.denseMatch[:,0]
    graph.denseMatch[:,1] = (imA.shape[1] * 0.5) - graph.denseMatch[:,1]
    graph.denseMatch[:,2] = (imB.shape[0] * 0.5) - graph.denseMatch[:,2]
    graph.denseMatch[:,3] = (imB.shape[1] * 0.5) - graph.denseMatch[:,3]

    graph.denseMatch = np.transpose(graph.denseMatch)
    graph.denseMatch = graph.denseMatch[[1,0,3,2,4],:]
    return graph

def propagate(i_m,mim_i,mim_j,zncc_i,zncc_j,winhalfsize):
    """
    Please cite this paper if you use this code.
    J. Xiao, J. Chen, D.-Y. Yeung, and L. Quan
    Learning Two-view Stereo Matching
    Proceedings of the 10th European Conference on Computer Vision (ECCV2008)
    Springer Lecture Notes in Computer Science (LNCS), Pages 15-27
    """
    # testMat = None
    # with open('test/varProp.txt','r') as f:
    #     testMat=pst(f.read())
    # testMat[:,0] = testMat[:,0] - 1
    # testMat[:, 1] = testMat[:, 1] - 1
    # testMat[:, 2] = testMat[:, 2] - 1
    # testMat[:, 3] = testMat[:, 3] - 1


    hi = mim_i.shape[0]
    wi = mim_i.shape[1]
    hj = mim_j.shape[0]
    wj = mim_j.shape[1]

    max_cost = 0.5
    outIm_i = np.zeros((mim_i.shape[0],mim_i.shape[1],2))
    outIm_j = np.zeros((mim_j.shape[0], mim_j.shape[1], 2))

    outIm_i[:,:,0] = mim_i - 2
    outIm_i[:, :, 1] = outIm_i[:,:,0]
    outIm_j[:,:,0] = mim_j - 2
    outIm_j[:, :, 1] = outIm_j[:,:,0]

    elementosI = mim_i.shape[0]*mim_i.shape[1]
    elementosJ = mim_j.shape[0]*mim_j.shape[1]
    maxMatchingNo = min(elementosI , elementosJ)
    maxIndexValid = max(elementosI,elementosJ)
    nbMaxStart = maxIndexValid + 5*5*9

    #Crear priority queue de tamano
    maxSizeHeap = nbMaxStart*25  + maxIndexValid
    heap = dummyHeap(maxSizeHeap)


    match_pair = i_m
    match_pair_size=0
    for match_pair_size in range(match_pair.shape[0]):
        e1 = zncc_i[match_pair[match_pair_size,0],match_pair[match_pair_size,1],:]
        e2 = zncc_j[match_pair[match_pair_size,2],match_pair[match_pair_size,3],:]
        val = np.sum(e1*e2)
        match_pair[match_pair_size,4] = val
        heap.push(match_pair_size,val)

    while (maxMatchingNo >= 0 and  heap.size() > 0):
        bestPri,bestInd = heap.pop()

        x0 = match_pair[bestInd,0]
        y0 = match_pair[bestInd,1]
        x1 = match_pair[bestInd,2]
        y1 = match_pair[bestInd,3]

        xMin0 = int(max(winhalfsize,x0-winhalfsize))
        xMax0 = int(min(hi-winhalfsize-1,x0+winhalfsize+1))
        yMin0 = int(max(winhalfsize,y0-winhalfsize))
        yMax0 = int(min(wi-winhalfsize-1,y0+winhalfsize+1))

        xMin1 = int(max(winhalfsize,x1-winhalfsize))
        xMax1 = int(min(hj-winhalfsize-1,x1+winhalfsize+1))
        yMin1 = int(max(winhalfsize,y1-winhalfsize))
        yMax1 = int(min(wj-winhalfsize-1,y1+winhalfsize+1))

        localH = []
        for yy0 in range(yMin0,yMax0+1):
            for xx0 in range(xMin0,xMax0+1):
                if outIm_i[xx0,yy0,0] == -1:
                    xx = int(xx0 + x1 - x0)
                    yy = int(yy0 + y1 - y0)
                    for yy1 in range(max(yMin1,yy-1), min(yMax1,yy+2)+1):
                        for xx1 in range(max(xMin1,xx-1),min(xMax1,xx+2)+1):
                            if outIm_j[xx1,yy1,0] == -1:
                                auxCost =np.sum(zncc_i[xx0,yy0,:] * zncc_j[xx1,yy1,:])
                                if (1-auxCost) <= max_cost:
                                    localH.append([xx0,yy0,xx1,yy1,auxCost])

        if len(localH) > 0:
            localH.sort(key=lambda x: x[4],reverse=True)
            for elem in localH:
                xx0, yy0, xx1, yy1, auxCost = elem
                if (outIm_i[xx0,yy0,0] < 0) and (outIm_j[xx1,yy1,0] < 0):
                    outIm_i[xx0,yy0,:] = [xx1,yy1]
                    outIm_j[xx1,yy1,:] = [xx0,yy0]
                    #probablemente necesitare vstack
                    match_pair_size+=1
                    match_pair = np.vstack([match_pair,elem])
                    # cond = np.allclose(testMat[match_pair_size, :],
                    #                       match_pair[match_pair_size, :])
                    # if not cond:
                    #     print "hi"
                    heap.push(match_pair_size,auxCost)
                    maxMatchingNo-=1
    match_pair = match_pair[i_m.shape[0]:, :]
    return match_pair,outIm_j,outIm_j

def ZNCCpath_all(im,half_size_prop):
    dim3=np.power(2*half_size_prop + 1,2)
    zncc = np.zeros((im.shape[0],im.shape[1],dim3))
    k=0
    for i in range(-half_size_prop,half_size_prop+1):
        for j in range(-half_size_prop, half_size_prop + 1):
            x0 = half_size_prop
            xf = zncc.shape[0]  - half_size_prop
            y0 = half_size_prop
            yf = zncc.shape[1]  - half_size_prop
            zncc[x0:xf,y0:yf,k] = im[x0+i:xf+i , y0+j:yf+j]
            k+=1
    zncc_m = np.mean(zncc,2)

    t=(2*half_size_prop+1) * zncc_m
    desv = np.sqrt( np.sum(zncc*zncc,axis=2) - (t*t) )
    extMean = np.repeat(zncc_m.reshape((zncc_m.shape[0],zncc_m.shape[1],1)),dim3,axis=2)
    extDesv = np.repeat(desv.reshape((desv.shape[0],desv.shape[1],1)),dim3,axis=2)
    zncc = (zncc-extMean) / extDesv
    return zncc

def reliableArea(im):
    permARow = range(1,im.shape[0]) + [0]
    permACol = range(1,im.shape[1]) + [0]
    permBRow = [im.shape[0] - 1] + range(0,im.shape[0]-1)
    permBCol = [im.shape[1] - 1] + range(0, im.shape[1] - 1)

    t1= np.maximum( np.abs( im - im[permARow,:] ) ,np.abs( im - im[permBRow,:] ))
    t2= np.maximum( np.abs( im - im[:,permACol] ) ,np.abs( im - im[:,permBCol] ))

    rim = np.maximum(t1, t2)
    rim[0,:] = 0
    rim[-1, :] = 0
    rim[:,0] = 0
    rim[:, -1] = 0
    rim = (rim < 0.01)
    return (1-rim).astype(np.float64)


def toGrayScale(im):
    return (30 * im[:,:,0] + 150 * im[:,:,1] + 76 * im[:,:,2]) / 256
def denseReconstruction(graph,merged,Kmat,ims):
    idFrame = graph.frames[0]
    p1 = np.dot(Kmat, merged.mot[:,:,idFrame])
    p2 = np.dot(Kmat, merged.mot[:,:,idFrame+1])
    p = np.zeros((p1.shape[0],p1.shape[1],2))
    p[:,:,0] = p1
    p[:,:,1] = p2
    X = np.zeros((4,graph.denseMatch.shape[1]))
    colIms = np.array([ims[1], ims[0]]).reshape((2, 1))
    imsize = colIms.repeat(2, axis=1)

    # import pickle
    # with open('testX.pkl','rb') as f:
    #     X = pickle.load(f)
    for i in range(graph.denseMatch.shape[1]):
        X[:,i] = vgg_X_from_xP_nonlin(graph.denseMatch[0:4,i].reshape((2,2)),p,imsize,X=None)
    X=X[0:3,:] / X[[3,3,3],:]
    x1 = np.dot(p1, np.vstack([X, np.ones((1,X.shape[1]))]))
    x2 = np.dot(p2, np.vstack([X, np.ones((1, X.shape[1]))]))
    x1 = x1[0:2,:] / x1[[2,2],:]
    x2 = x2[0:2,:] / x2[[2,2],:]
    graph.denseX = X
    temp = np.vstack([x1+1, x2+1]) - graph.denseMatch[0:4,:]
    graph.denseRepError = np.sum(temp*temp,axis=0)

    #Consigue las camaras segun merged
    rt1 = merged.mot[:,:,idFrame]
    rt2 = merged.mot[:,:,idFrame+1]
    c1 = - np.dot( np.transpose(rt1[0:3,0:3]) , rt1[:,3])
    c2 = - np.dot(np.transpose(rt2[0:3, 0:3]), rt2[:, 3])

    nPo = graph.denseMatch.shape[1]

    t1 = np.repeat(c1.reshape((3,1)), nPo, axis=1)
    t2 = np.repeat(c2.reshape((3, 1)), nPo, axis=1)
    view_dirs_1 = X - t1
    view_dirs_2 = X - t2
    temp =(1.0 / np.sqrt(np.sum(view_dirs_1 * view_dirs_1,axis = 0)))
    t1=  np.repeat(temp.reshape((1,nPo)), 3,axis=0)
    view_dirs_1 =  view_dirs_1 * t1

    temp = (1.0 / np.sqrt(np.sum(view_dirs_2 * view_dirs_2,axis = 0)))
    t2=  np.repeat(temp.reshape((1,nPo)), 3,axis=0)
    view_dirs_2 = view_dirs_2 * t2

    graph.cos_angles = np.sum(view_dirs_1 * view_dirs_2)
    c_dir1 = np.transpose(rt1[2,0:3])
    c_dir2 = np.transpose(rt2[2,0:3])

    t1=  np.repeat(c_dir1.reshape((3,1)), nPo,axis=1)
    t2 = np.repeat(c_dir2.reshape((3,1)), nPo,axis=1)
    bt1=(np.sum(view_dirs_1 * t1,axis=0) > 0)
    bt2=(np.sum(view_dirs_2 * t2,axis=0) > 0)
    graph.visible = np.bitwise_and(bt1,bt2)

    return graph



def outputPly(data,name):
    d2 = np.empty(data.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(data.shape[0]):
        d2[i] = tuple(data[i])
    el = PlyElement.describe(d2, 'vertex')
    PlyData([el]).write(name+'.ply')


def testDenseTriangulation():
    import pickle
    #Cargo grafos esperados del resultado
    listGAfter = ['test/d1.txt','test/d2.txt','test/d3.txt','test/d4.txt']
    for ind,filepat in enumerate(listGAfter):
        with open(filepat,'r') as f:
            listGAfter[ind] = f.read()
    listGAfter = [dictToGraph(parseStruct(st)) for st in listGAfter]
    #carga el grafo merged de prueba
    mer = dictToGraph(parseStruct(m34))
    imsize = (640, 480, 3)
    derp = "[719.5459 0 0;0 719.5459 0;0 0 1]"

    gRes = denseReconstruction(listGAfter[0], mer, pst(derp), (imsize[0],imsize[1]))

    data = visualizeDense([gRes], mer, imsize)

    outputPly(data,"ble")

#Agarrar la lista de grafos en dense match
#Agarrar lista de reesultados en dense match
def testFullDenseMatch():
    import pickle
    imsize = (640,480,3)

    #Cargo imagenes
    imlist=["B21.jpg","B22.jpg","B23.jpg","B24.jpg","B25.jpg"]
    imlist = map(lambda x : cv2.imread("test/"+x), imlist)

    #Cargo grafos originales antes de dense
    with open("test/graphsMergeStart.pkl",'rb') as f:
        listGbefore = pickle.load(f)
    for i in range(len(listGbefore)):
        listGbefore[i].matches = np.transpose(listGbefore[i].matches)
    #Cargo grafos esperados del resultado
    listGAfter = ['test/d1.txt','test/d2.txt','test/d3.txt','test/d4.txt']
    for ind,filepat in enumerate(listGAfter):
        with open(filepat,'r') as f:
            listGAfter[ind] = f.read()
    listGAfter = [dictToGraph(parseStruct(st)) for st in listGAfter]

    for ind,graph in enumerate(listGbefore):
        t=denseMatch(graph, imlist[ind], imlist[ind+1], imsize, imsize)
        assert(fullTest(t.denseMatch,listGAfter[ind].denseMatch,debug=True))
        print "Iteracion ",ind," OK"

if __name__ == "__main__":
    testDenseTriangulation()
    # testFullDenseMatch()