from copy import deepcopy
import numpy as np
from utils.graph import transformPtsByRt, tempGraph
from utils.parser import parseStruct, getIndexOfRow
from test.dataMergeTest import *

def mergeG(gA, gB):
    # Como ejemplo sean frames A 1 2 y B 2 3
    # los frarmes son las camaras o fotos

    # Primero se calculan frames que hacen overlap
    comFram = list(set(gA.frames).intersection(gB.frames))

    # Luego las que son propias de A y las propias de B (en ej A1 B3)
    propB = list(set(gB.frames).difference(gA.frames))
    indpB = [gB.frames.index(a) for a in propB]

    # Si no hay comunes retorna error
    # Si las propias de B son ninguna tira error
    if len(comFram) == 0:
        raise Exception("Comunes vacio ")
    if len(propB) == 0:
        raise Exception("No hay propias de B")

    # Crear grafo mezclca igual a grafo A
    merged = deepcopy(gA)

    # Para el primer overlap (pueden existir muchos)
    firstOv = comFram[0]

    # Transforma B.mot b.str al mismo sistema de cordenadas de A
    commonA = gA.frames.index(firstOv)
    commonB = gB.frames.index(firstOv)
    # Consigue transformada rtB

    transf = catRt(invertRt(gA.mot[:, :, commonA]), gB.mot[:, :, commonB])
    gB.str = transformPtsByRt(np.transpose(gB.str), transf, False)  # Aplicar a str B

    # Mot ahora es la concadenacion de mot y el inverso RtB
    for i in range(len(gB.frames)):
        gB.mot[:, :, i] = catRt(gB.mot[:, :, i], invertRt(transf))
    merged.frames = list(set(gA.frames).union(set(gB.frames)))
    newMot = np.zeros((3, 4, len(merged.frames)))
    newMot[:, :, np.array(range(len(gA.frames)))] = gA.mot
    newMot[:, :, np.array(range(len(gA.frames), len(merged.frames)))] = gB.mot[:, :, indpB]

    merged.mot = newMot
    # Agrega frames a grafico

    # Ahora caso common frames mas de una
    for fr in comFram:
        cA = gA.frames.index(fr)
        cB = gB.frames.index(fr)

        obsIndA = gA.ObsIdx[:, cA][gA.ObsIdx[:, cA] != -1]
        obsIndA = gA.ObsIdx[:, cA]
        valA = gA.obsVal[obsIndA.astype(np.int), :]

        obsIndB = gB.ObsIdx[:, cB][gB.ObsIdx[:, cB] != -1]
        obsIndB = gB.ObsIdx[:, cB]
        valB = gB.obsVal[obsIndB.astype(np.int), :]

        iA = findInterIndexA(valA, valB)[0]
        comunes = valA[iA]
        iB = np.array([getIndexOfRow(valB, row)[0][0] for row in comunes])

        iA,iB = deleteRepeated(iA.tolist(), iB.tolist(),valA,valB)
        iA = np.array(iA)
        iB = np.array(iB)

        for i in range(iA.shape[0]):
            # idA = obsIndA[iA[i]]
            # idB = obsIndB[iB[i]]
            for j in range(len(indpB)):
                bObbsIdx = gB.ObsIdx[iB[i], indpB[j]]
                # Agrego un elemento a obsVal y a ObsIdx
                merged.obsVal = np.vstack([merged.obsVal, gB.obsVal[bObbsIdx, :]])
                while merged.ObsIdx.shape[1] < (len(gA.frames) + j + 1):
                    merged.ObsIdx = np.hstack([merged.ObsIdx, minus1((merged.ObsIdx.shape[0], 1))])
                merged.ObsIdx[iA[i], len(gA.frames) + j] = merged.obsVal.shape[0]-1

        # Calcula set diference
        diferentesB = setDif(valB, valA)
        idB = np.array([getIndexOfRow(valB, row)[0][0] for row in diferentesB])

        for i in range(idB.shape[0]):
            bObbsIdx = gB.ObsIdx[idB[i], cB]
            merged.obsVal = np.vstack([merged.obsVal, gB.obsVal[bObbsIdx, :]])
            merged.ObsIdx = np.vstack([merged.ObsIdx, minus1((1,merged.ObsIdx.shape[1]))])
            merged.ObsIdx[merged.ObsIdx.shape[0]-1, cA] = merged.obsVal.shape[0]-1
            merged.str = np.vstack([ merged.str   ,  gB.str[:,idB[i]].reshape((1,3))  ])
            for j in range(len(indpB)):
                bObbsIdx = gB.ObsIdx[idB[i], indpB[j]]
                merged.obsVal = np.vstack([merged.obsVal, gB.obsVal[bObbsIdx, :]])
                while merged.ObsIdx.shape[1] < (len(gA.frames) + j + 1):
                    merged.ObsIdx = np.hstack([merged.ObsIdx, minus1((merged.ObsIdx.shape[0], 1))])
                merged.ObsIdx[-1, len(gA.frames) + j] = merged.obsVal.shape[0]-1


    #Revisa si quedo alguna columna sin algun valor
    #Selecciona en ObsIdx los frames comunes y dif en Gb
    #Asegurate que para ningun punto se cumpla A and B
    #Siendo A = En columnas comunes todas tienen el valor de -1
    #Siendo B = En columnas dif la suma de los valores mayores que -1 es mayor que 0
    newB = np.zeros((1,len(gB.frames)))
    newB[:,np.array(indpB)] = 1
    A= (np.sum( gB.ObsIdx[:,np.bitwise_not(newB.astype(np.bool)).astype(int)[0]], axis=1) < 0 )
    B= (np.sum( gB.ObsIdx[:,newB[0].astype(np.int)], axis=1) > 0 )
    assert(not np.any(np.bitwise_and(A,B)))

    return merged

def deleteRepeated(indexsA,indexsB,vA,vB):
    #Conseguir indices repetidos en vA,vB
    # filterRepetedA = lambda x: x not in uniqueRowsIndexs(vA)
    # repeteA = filter(filterRepetedA, range(vA.shape[0]))

    toDelete = []
    for i in range(len(indexsA)):
        for j in range(i+1,len(indexsA)):
            if np.array_equal(vA[indexsA[i]],vA[indexsA[j]]):
                toDelete.append(indexsA[j])
    toDelete = list(set(toDelete)) #Borra repetidos sino tira error.
    for e in toDelete:
        ind = indexsA.index(e)
        indexsA.remove(e)
        del indexsB[ind]
    assert(np.array_equal(vA[np.array(indexsA)],vB[np.array(indexsB)]))
    assert (len(set(indexsA)) == len(indexsA))
    assert (len(set(indexsB)) == len(indexsB))
    assert (len(indexsB) == len(indexsA))
    return indexsA,indexsB

def uniqueRows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]


def uniqueRowsIndexs(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx


def minus1(shape):
    return -1 * np.ones(shape)


def setDif(a1, a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])


def findInterIndexA(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    return np.nonzero(np.in1d(x.view('d,d').reshape(-1), y.view('d,d').reshape(-1)))


def catRt(rt1, rt2):
    temp = rt1[:, 0:3]
    temp2 = rt1[:, 3].reshape(3, 1)
    return np.hstack([np.dot(temp, rt2[:, 0:3]), np.dot(temp, rt2[:, 3]).reshape(3, 1) + temp2])


def invertRt(rt):
    temp = np.transpose(rt[0:3, 0:3])
    return np.hstack([temp, np.dot(-temp, rt[0:3, 3]).reshape(3, 1)])


def removeOutlierPts(g, th_pix=10):
    sq_th_pix = th_pix * th_pix
    td = 2
    tincos = np.cos(np.pi * td *1.0 / 180 )
    for i in range(g.ObsIdx.shape[1]):
        X = np.dot(g.focal,transformPtsByRt(np.transpose(g.str),g.mot[:,:,i],False))
        xy = X[0:2,:] / X[[2,2],:]
        selector = np.where(g.ObsIdx[:,i] != -1)[0]

        dif = xy[:,selector] - np.transpose(g.obsVal[g.ObsIdx[selector,i].astype(np.int)])
        outliers = np.sum(dif*dif,axis=0) > sq_th_pix
        cantB = np.sum(outliers)
        if cantB > 0:
            print "Se borraron ", cantB, " outliers de ", outliers.shape[0], \
                " puntos totales con sq_th_pix de ", sq_th_pix
        p2keep = np.ones((1,g.str.shape[0]))
        p2keep[:, selector[outliers]] = False
        p2keep = p2keep[0].astype(np.bool)
        g.str = g.str[p2keep,:]
        g.ObsIdx = g.ObsIdx[p2keep,:]

    nF = len(g.frames)
    pos = np.zeros((3,nF))

    for ii in range(nF):
        Rt = g.mot[:,:,ii]
        pos[:,ii] = - np.dot( np.transpose(Rt[0:3,0:3]), Rt[:,3])

    view_dirs = np.zeros((g.str.shape[0],3,nF))
    for c in range(g.ObsIdx.shape[1]):
        selector = np.where(g.ObsIdx[:,c] != -1)[0]
        t=np.repeat(pos[:,c].reshape((1,3)),g.str[selector,:].shape[0],axis=0)
        camera_v_d = g.str[selector,:] - t
        d_lengh = np.sqrt(np.sum(camera_v_d * camera_v_d,axis=1))
        dt = 1.0 / d_lengh
        camera_v_d = camera_v_d * np.transpose(np.vstack([dt,dt,dt]))
        view_dirs[selector,:,c] = camera_v_d
    for c1 in range(g.ObsIdx.shape[1]):
        for c2 in range(g.ObsIdx.shape[1]):
            if c1 == c2:
                continue
            selector = np.where( np.bitwise_and(g.ObsIdx[:,c1] != -1 , g.ObsIdx[:,c2] != -1 ) )[0]
            v_d1 = view_dirs[selector,:,c1]
            v_d2 = view_dirs[selector,:,c2]
            cos_a = np.sum(v_d1 * v_d2,axis=1)
            outliers = cos_a > tincos

            cantB = np.sum(outliers)
            if cantB > 0:
                print "Se borraron ",cantB," outliers de ",outliers.shape[0],\
                    " puntos totales con cost_thr de ",tincos
            p2keep = np.ones((1, g.str.shape[0]))
            p2keep[:,selector[outliers]] = False
            p2keep = p2keep[0].astype(np.bool)
            g.str = g.str[p2keep, :]
            g.ObsIdx = g.ObsIdx[p2keep, :]

    return g



def test3It():
    Originals = [parseStruct(or1), parseStruct(or2), parseStruct(or3), parseStruct(or4)]

    expectedMerged = [parseStruct(st1), parseStruct(st2), parseStruct(st3), parseStruct(st4)]

    orgGraphs = [dictToGraph(g) for g in Originals]
    expectedGraphs = [dictToGraph(g) for g in expectedMerged]

    assert (expectedGraphs[0] == orgGraphs[0])
    assert (orgGraphs[0] != orgGraphs[1])

    graphMerged = orgGraphs[0]
    # merge de vistas parciales
    for i in range(len(orgGraphs) - 1):
        graphMerged = mergeG(graphMerged, orgGraphs[i + 1])
        assert (graphMerged == expectedGraphs[i+1])
    print "TEST MERGE OK"
    return Originals
def testOutliers():
    gbefore = dictToGraph(parseStruct(beforeMerge))
    expectedG = dictToGraph(parseStruct(expectedMerge))
    result = removeOutlierPts(gbefore, 10)
    assert(result == expectedG)
    print "TEST OUTLIERS OK"
    pass


def dictToGraph(d):
    # Transform graph from matlab format to python
    graph = tempGraph()
    graph.frames = (d['frames'] - 1).tolist()[0]
    graph.f = d['f']
    graph.focal = np.eye(3)*graph.f
    graph.focal[2,2] = 1
    graph.mot = d['Mot']
    graph.str = np.transpose(d['Str'])
    graph.obsVal = np.transpose(d['ObsVal'])
    graph.ObsIdx = np.transpose(d['ObsIdx']) - 1

    if d.has_key('matches'):
        graph.matches = np.transpose(d['matches'])
    if d.has_key('denseMatch'):
        graph.denseMatch = d['denseMatch']
        graph.denseMatch[0,:] = graph.denseMatch[0,:] + 1
        graph.denseMatch[1,:] = graph.denseMatch[1,:] + 1
        graph.denseMatch[2,:] = graph.denseMatch[2,:] + 1
        graph.denseMatch[3,:] = graph.denseMatch[3,:] + 1

    return graph


if __name__ == '__main__':
    testOutliers()
    a = test3It()

    # import pickle
    #
    # with open("f.pkl", 'rb') as f:
    #     graphList = pickle.load(f)
    # graphMerged = graphList[0]
    # # merge de vistas parciales
    # for i in range(len(graphList) - 1):
    #     graphMerged = mergeG(graphMerged, graphList[i + 1])
    # pass
