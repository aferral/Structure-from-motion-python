import numpy as np
from scipy.optimize import least_squares
def bundleAdjustment(graph, adjustFocalLength=False):
    nCamaras = graph.mot.shape[2]
    mot = np.zeros((3,2,nCamaras))

    for i in range(nCamaras):
        mot[:,0,i] = rotationMatrix2angleaxis(graph.mot[:,0:3,i])
        mot[:,1,i] = graph.mot[:,3,i]

    stre = graph.str

    #Se asume px,py = 0

    px,py =0,0
    f = graph.f

    # t = packmotst(mot,stre)
    # unpackMotStr(t, nCamaras)

    res = reprojectionResidual(graph.ObsIdx,graph.obsVal,px,py,f,mot,stre)

    error = lambda x : 2* np.sqrt( np.sum(np.power(x,2)) / x.shape[0] )


    print "Error inicial de ",error(res)
    #Realizar optimizacion de valores
    #Quiero conseguir minimizar la norma del vector reprojectionResidual
    fun = lambda x : wrapperFuntionStrMot(x,nCamaras,graph.ObsIdx,graph.obsVal,px,py,f)
    sol = least_squares(fun, packmotst(mot,stre), method='lm',max_nfev=1000)
    resultM, resultS = unpackMotStr(sol.x,nCamaras,graph.ObsIdx.shape[0])
    print "Error despues de optimizar de ", error(sol.fun)

    if adjustFocalLength:
        # Realizar optimizacion de valores
        # Quiero conseguir minimizar la norma del vector reprojectionResidual
        fun = lambda x: wrapperFuntionStrMotF(x, nCamaras, graph.ObsIdx, graph.obsVal, px, py)
        sol = least_squares(fun, packMSF(mot, stre,f), method='lm')
        resultM, resultS,resultF = unpackMotStrf(sol.x, nCamaras)
        print "Error despues de optimizar de ", error(sol.fun)
        graph.focal = np.eye(3) * resultF
        graph.f = resultF

    for i in range(nCamaras):
        graph.mot[:,:, i] = np.hstack([AngleAxis2RotationMatrix(resultM[:, 0, i]) , resultM[:,1,i].reshape((3,1))])
    graph.str = resultS
    return graph





def packMSF(mot,st,f):
    a=mot.flatten(order='F')
    b=st.flatten(order='F')
    return np.concatenate((f,a,b))
def packmotst(mot,st):
    return np.concatenate((mot.flatten(order='F'), st.flatten(order='F')))

def wrapperFuntionStrMot(x,ncam,ObsIdx,ObsVal,px,py,f):
    mot, st = unpackMotStr(x,ncam,ObsIdx.shape[0])
    return reprojectionResidual(ObsIdx, ObsVal, px, py, f, mot,st)
def wrapperFuntionStrMotF(x,ncam,ObsIdx,ObsVal,px,py):
    mot, st,f = unpackMotStrf(x,ncam,ObsIdx.shape[0])
    return reprojectionResidual(ObsIdx, ObsVal, px, py, f, mot,st)

def rotationMatrix2angleaxis(R):
    #El problema ocurre que hay valores muy pequenos asi que se sigue el sigueinte proceso

    ax = [0,0,0]
    ax[0] = R[2,1] - R[1,2]
    ax[1] = R[0,2] - R[2,0]
    ax[2] = R[1,0] - R[0,1]
    ax = np.array(ax)


    costheta = max( (R[0,0] + R[1,1] + R[2,2] - 1.0) / 2.0 , -1.0)
    costheta = min(costheta, 1.0)

    sintheta = min(np.linalg.norm(ax) * 0.5 , 1.0)
    theta = np.arctan2(sintheta, costheta)

    #TODO (esto tenia problemas de precision en matlba nose si se mantienen)
    #por seguridad copie la version que arregla estos problemas

    kthreshold = 1e-12
    if (sintheta > kthreshold) or (sintheta < -kthreshold):
        r = theta / (2.0 *sintheta)
        ax = r * ax
        return ax
    else:
        if (costheta > 0.0):
            ax = ax *0.5
            return ax
        inv_one_minus_costheta = 1.0 / (1.0 - costheta)

        for i in range(3):
            ax[i] = theta * np.sqrt((R(i, i) - costheta) * inv_one_minus_costheta)
            cond1 = ((sintheta < 0.0) and (ax[i] > 0.0))
            cond2 = ((sintheta > 0.0) and (ax[i] < 0.0))
            if cond1 or cond2:
                ax[i] = -ax[i]
        return ax
    pass


# La funcion reprojectionResidual tiene 3 usos. 1 Evaluar unn modelo, 2. Optimizar dado [Mot[:] ; Str[:]]
# Y 3 optimizar daod [f; Mot(:); Str(:)] Por eso existen estas funciones unpack
def unpackMotStr(vect,ncam,n):
    cut = 3 * 2 * ncam
    mot = np.reshape(vect[0:cut], (3, 2,ncam),order='F')
    st = np.reshape(vect[cut:], (n,3),order='F' )
    return mot,st

def unpackMotStrf(vect,ncam,n):
    cut = 1+3*2*ncam
    f   = vect[0]
    mot = np.reshape(vect[1:cut], (3, 2,ncam),order='F')
    st = np.reshape(vect[cut:], (n,3),order='F' )
    return f,mot,st

def reprojectionResidual(ObsIdx,ObsVal,px,py,f,Mot,Str):
    nCam = len(ObsIdx[0])

    residuals = np.zeros((0,0))
    for i in range(nCam):

        validsPts = ObsIdx[:,i] != -1
        valIndexs = ObsIdx[validsPts,i]

        validMot = Mot[:,0,i]
        validStr = Str[validsPts,:]

        RP = AngleAxisRotatePts(validMot, validStr)

        TRX = RP[:,0] + Mot[0, 1, i]
        TRY = RP[:,1] + Mot[1, 1, i]
        TRZ = RP[:,2] + Mot[2, 1, i]

        TRXoZ = TRX / TRZ
        TRYoZ = TRY / TRZ

        x = f * TRXoZ + px
        y = f * TRYoZ + py

        ox = ObsVal[ valIndexs.astype('int'),0]
        oy = ObsVal[valIndexs.astype('int'),1]

        step = np.vstack([(x-ox),(y-oy)])

        if i ==0:
            residuals = step
        else:
            residuals = np.hstack([residuals, step])
    return residuals.flatten()

def AngleAxisRotatePts(validMot, validStr):
    validStr=np.transpose(validStr)
    angle_axis = np.reshape(validMot[0:3], (1,3))
    theta2 = np.inner(angle_axis, angle_axis)

    if (theta2 > 0.0):
        theta = np.sqrt(theta2)
        w = (1.0/theta) * angle_axis

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        w_cross_pt = np.dot(xprodmat(w),validStr)

        w_dot_pt = np.dot(w,validStr)
        t1= (validStr * costheta)
        t2 = (w_cross_pt * sintheta)
        t3 = np.dot( (1 - costheta) * np.transpose(w),w_dot_pt)
        result = t1 + t2 + t3

    else:
        w_cross_pt = np.dot(xprodmat(angle_axis),validStr)
        result = validStr + w_cross_pt
    return np.transpose(result)

def xprodmat(a):
    assert(a.shape[0] == 1  and a.shape[1] ==3)
    ax=a[0,0]
    ay=a[0,1]
    az=a[0,2]
    A=np.array([[0, -az,ay],[az,0,-ax],[-ay,ax,0]])
    return A


def AngleAxis2RotationMatrix(angle_axis):
    R= np.zeros((3,3))
    theta2 = np.inner(angle_axis, angle_axis)
    if (theta2 > 0.0):
        theta = np.sqrt(theta2)
        wx = angle_axis[0] / theta
        wy = angle_axis[1] / theta
        wz = angle_axis[2] / theta
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        R[0,0] = costheta + wx * wx * (1 - costheta)
        R[1,0] = wz * sintheta + wx * wy * (1 - costheta)
        R[2,0] = -wy * sintheta + wx * wz * (1 - costheta)
        R[0,1] = wx * wy * (1 - costheta) - wz * sintheta
        R[1,1] = costheta + wy * wy * (1 - costheta)
        R[ 2, 1] = wx * sintheta + wy * wz * (1 - costheta)
        R[0,2] = wy * sintheta + wx * wz * (1 - costheta)
        R[1,2] = -wx * sintheta + wy * wz * (1 - costheta)
        R[2,2] = costheta + wz * wz * (1 - costheta)
    else:
        R[0, 0] = 1
        R[1, 0] = -angle_axis[2]
        R[2, 0] = angle_axis[1]
        R[0, 1] = angle_axis[2]
        R[1,1] = 1
        R[2,1] = -angle_axis[0]
        R[0,2] = -angle_axis[1]
        R[1,2] = angle_axis[0]
        R[2,2] = 1
    return R
if __name__ == '__main__':
    pass


