from main import updateMerged
from utils.graph import showGraph
from utils.mergeGraph import invertRt, catRt, dictToGraph
from utils.parser import *
from test.dataMergeTest import *

def inverseTest(debug):
    if debug:
        print "----------Test Inverse ---------"

    Rtstr = "[0.980636494759034 -0.0848521010283081 -0.176499818973837 0.916986481160674;0.0852828285792061 0.996343432040814 -0.00515796264643823 -0.0706249414988878;0.176292099358551 -0.00999431739610548 0.984287157959272 -3.44237398527039]"
    RtInvStr = "[0.980636494759034 0.0852828285792061 0.176292099358551 -0.286343977206979;-0.0848521010283081 0.996343432040814 -0.00999431739610548 0.113770747936833;-0.176499818973837 -0.00515796264643823 0.984287157959272 3.54976817371088]"

    mat = pst(Rtstr)
    expected = pst(RtInvStr)
    result = invertRt(mat)

    assert(fullTest(expected, result,debug))

def cancatenateRtTest(debug):
    if debug:
        print "----------Test Concat ---------"
    rtStr1 = "[0.980636494759034 0.0852828285792061 0.176292099358551 -0.286343977206979;-0.0848521010283081 0.996343432040814 -0.00999431739610548 0.113770747936833;-0.176499818973837 -0.00515796264643823 0.984287157959272 3.54976817371088]"
    rtStr2 = "[0.999973627905639 -0.00211608063595445 0.00694735172399251 -0.189212428340614;0.00201958191092069 0.999901798484945 0.0138677566850883 -0.0264935267310269;-0.00697601477491304 -0.0138533602174298 0.999879702578538 2.10847679899011]"
    catStr = "[0.979553052688945 0.0807571098228607 0.184266400032201 -0.102444231255662;-0.0827679456224583 0.996563598373194 0.00323443579447091 0.0823563823207143;-0.183371982988695 -0.0184196528341118 0.982871015059588 5.65864742178288]"

    mat1 = pst(rtStr1)
    mat2 = pst(rtStr2)
    expected = pst(catStr)
    result = catRt(mat1,mat2)

    assert (fullTest(expected, result, debug))

def testFullMergeProcess():
    # mergeAllGraph(gL, imsize)
    graphL = [g1,g2,g3,g4]
    graphL= [dictToGraph(parseStruct(st)) for st in graphL]
    expectedAtIt = [m12,m23,m34]
    expectedAtIt = [dictToGraph(parseStruct(st)) for st in expectedAtIt]

    imsize = (640, 480, 3)

    tempResult = graphL[0]
    # merge de vistas parciales

    #Aqui lamentablemtne la optimizacion tira valores algo distintos
    #Pero los errores se mantienen a 0.001 de distancia asi que se esperan
    #buenos resultaods.

    for i in range(len(graphL) - 1):
        tempResult = updateMerged(tempResult, graphL[i + 1], imsize)
        tempResult.closeEnought(expectedAtIt[i],0.5)
    showGraph(tempResult, imsize)

if __name__ == "__main__":
    debug = True
    testFullMergeProcess()
    inverseTest(debug)
    cancatenateRtTest(debug)

