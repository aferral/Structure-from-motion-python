import numpy as np

def getPose(E,K,matches,ims):
    n,d = matches.shape

    U, s, V = np.linalg.svd(E, full_matrices=True)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])

    S = np.dot(np.dot(U,Z),np.transpose(U))

    R1 = np.dot(np.dot(U,W),V)
    R2 = np.dot(np.dot(U, np.transpose(W)), V)

    t1 = U[:,2]
    t2 = -1*U[:,2]

    if np.linalg.det(R1) < 0:
        print "Determinante negativo F1 multiplico por -1"
        R1 = -1*R1

    if np.linalg.det(R2) < 0:
        print "Determinante negativo R2 multiplico por -1"
        R2 = -1 * R2

    #Esto genera 4 posibles soluciones
    t1t = t1.reshape(3,1)
    t2t = t2.reshape(3,1)
    sols=[np.hstack((R1, t1t)), np.hstack((R1, t2t)), np.hstack((R2, t1t)), np.hstack((R2, t2t))]

    Rt = np.zeros((3,4,4))
    Rt[:,:,0] = sols[0]
    Rt[:, :, 1] = sols[1]
    Rt[:, :, 2] = sols[2]
    Rt[:, :, 3] = sols[3]


    #Por cada solucion
    P0 = np.dot(K,np.hstack([np.eye(3),np.zeros((3,1))]))
    goodV = np.zeros((1,4))
    for i in range(4):
        outX = np.zeros((n, 4))
        P1 = np.dot(K,sols[i])
        #Por cada par de puntos 2D
        for j in range(n):
            # aplicar vgg para calcular puntos 3D
            colIms = np.array([ims[1],ims[0]]).reshape((2,1))
            imsize = colIms.repeat(2,axis=1)
            pt = np.zeros((P0.shape[0],P0.shape[1],2))
            pt[:,:,0] = P0
            pt[:,:,1] = P1
            formatedMatched = np.reshape(matches[j,:],(2,2))
            outX[j,:] = vgg_X_from_xP_nonlin(formatedMatched,pt,imsize)
        #Aplicar escala
        outX = outX[:,0:3] / outX[:,[3,3,3]]

        t = Rt[0:3, 3, i].reshape((3,1))
        aux = np.transpose(outX[:,:]) - np.repeat(t, outX.shape[0], axis=1)
        t2 = Rt[2, 0:3, i].reshape((1,3))
        dprd = np.dot(t2,aux)
        goodV[0,i] = np.sum([np.bitwise_and(outX[:,2] > 0,dprd > 0)])




    #Calcular cual es mejor
    bestIndex = np.argmax(goodV)
    return Rt[:,:, bestIndex]


def vgg_X_from_xP_nonlin(u,P,imsize=None,X=None):
    eps = 2.2204e-16

    K = P.shape[2]
    assert(K >= 2)

    #Primero consigo X si no fue proporcioando
    if (X is None):
        X = vgg_X_from_xP_lin(u, P, imsize)
    #Fixed el -1 ????? (en version antigua los indices diferenciaban de un numero)
    newu = u.copy()-1
    newP = P.copy()
    if not imsize is None:
        for i in range(K):
            H = np.array([[2.0/imsize[0,i],0 , -1],
                 [0, 2.0/imsize[1,i], -1],
                 [0,0,1]])
            newP[:,:,i] = np.dot(H,newP[:,:,i])
            newu[i, :] = np.dot(H[0:2,0:2], newu[i, :]) + H[0:2,2]

    T, s, U = np.linalg.svd(X.reshape((4,1)))
    lc = T.shape[1]
    T = T[:,[1,2,3,0]]
    Q = newP.copy()
    for i in range(K):
        Q[:, :, i] = np.dot(newP[:, :, i], T )

    #DO THE NEWTON
    Y = np.zeros((3,1))
    eprev = np.inf
    for i in range(10):
        e,j = resid(Y,newu,Q)
        if (1- (np.linalg.norm(e) / np.linalg.norm(eprev))) < 1000*eps:
            break
        eprev = e
        jj = np.dot(np.transpose(j),j)
        je = np.dot(np.transpose(j),e)
        Y  = Y - np.linalg.solve(jj,je)
    X =np.dot(T,np.vstack([Y,1]))
    return X.flatten()

def resid(Y,u,Q):
    K = Q.shape[2]

    q = Q[:, 0:3, 0]
    x0 = Q[:, 3, 0]
    x0 = x0.reshape((3, 1))
    x = np.dot(q, Y) + x0

    tu = u[0,:].reshape((2,1))
    e = x[0:2]/x[2]-tu

    t1 = x[2]* q[0, :]
    t2 = x[0]* q[2, :]
    t3 = x[2]* q[1, :]
    t4 = x[1]* q[2, :]
    aux = np.vstack([t1 - t2, t3 - t4])
    J = (aux / (x[2] * x[2]))

    for i in range(1,K):
        q = Q[:,0:3,i]
        x0 = Q[:,3,i]
        x0 = x0.reshape((3,1))
        x = np.dot(q,Y) + x0
        tu = u[i, :].reshape((2, 1))
        e = np.vstack([e, x[0:2]/x[2]-tu  ])

        t1 = x[2] * q[0, :]
        t2 = x[0] * q[2, :]
        t3 = x[2] * q[1, :]
        t4 = x[1] * q[2, :]
        aux = np.vstack([t1-t2 , t3-t4 ])
        J = np.vstack([J, (aux/(x[2]*x[2]))])
    return e,J


def vgg_X_from_xP_lin(u,P,imsize):
    K = P.shape[2]
    newu = u.copy()
    newP = P.copy()
    if not imsize is None:
        for i in range(K):
            H = np.array([[2.0/imsize[0,i],0 , -1],
                 [0, 2.0/imsize[1,i], -1],
                 [0,0,1]])
            newP[:,:,i] = np.dot(H,newP[:,:,i])
            newu[i, :] = np.dot(H[0:2,0:2], newu[i, :]) + H[0:2,2]

    A= np.dot( formatVgg(np.hstack([newu[0,:],1])),  newP[:,:,0])
    for i in range(1,K):
        newRow = np.dot( formatVgg(np.hstack([newu[i,:],1])),  newP[:,:,i])
        A = np.vstack([A, newRow  ])

    U, s, out = np.linalg.svd(A)
    out = out.T
    out = out[:,-1]
    s = np.dot( np.reshape(newP[2,:,:], (4,K)).T ,out)
    if np.any(s < 0):
        out = -out

    return out

def formatVgg(X): #Debe ser vector de 3 elementos
    row = [0,X[2],-X[1]]
    row2 = [-X[2],0,X[0]]
    row3 = [X[1],-X[0],0]
    return np.array([row,row2,row3])

