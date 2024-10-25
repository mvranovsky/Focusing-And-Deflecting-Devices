#!/usr/bin/python3


from AstraWrapper.SettingsFile import SettingsFile
from AstraWrapper.Astra import Astra
import scipy as sc
import subprocess
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
#import ROOT 

Qlength = 0.08
Qbore = 0.017


def wait():
    while True:
        line = astra.process.stdout.readline()
        #print(line)
        if 'Goodbye' in line:
            return

def func(D, Pz):

    data = astra.runRef(*D, None, astra.setupLength,Pz, False)
    if data == 1:
        print(D,"1")
        return 1E+9

    Sum = astra.parallelFocusing(data)
    print(D, Sum)
    return Sum

def tripletFocusing(Pz):

    Dmin = [0.0, 0.0, 0.0]
    Dmax = [0.4, 0.4, 0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    method = "COBYLA"
    tolerance = 1e-5
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    # settings
    astra.setupLength = 0.9  #m

    astra.quadType(1)
    
    res = sc.optimize.minimize(func, (0.1,0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(Pz) )
    
    funcVal = func(res.x, Pz)

    #if astra.plotRefXY(*res.x, None,astra.setupLength, Pz, f"parallel Focusing, setup:{[math.ceil(num*10000)/100 for num in res.x]}\nminimum found, Pz={math.ceil(Pz*1e-6)} MeV,\nf(x',y')={math.ceil(funcVal*1e+6)/1000000} mrad^2", f"specialAssignment/solution{Pz*1e-6}MeV") == False:
    #    print("sad times")
    #    return

    #get beam ratio
    beamRatio = astra.beamRatio(*res.x, None, astra.setupLength, Pz)

    #get ang acceptance
    acc = astra.checkAngleAcceptance(*res.x, None, astra.setupLength, Pz)


    return [*[math.ceil(num*10000)/100 for num in res.x],None, astra.setupLength,Pz, funcVal,*acc, beamRatio]


def checkAng(data):

    Qpos5 = float(setFile.readOption("Q_pos(5)"))
    Qpos6 = float(setFile.readOption("Q_pos(6)"))

    Q4start = 0.9
    Q4end = 0.9 + Qlength
    Q5start = Qpos5 - Qlength/2
    Q5end = Qpos5 + Qlength/2
    Q6start = Qpos6 - Qlength/2
    Q6end = Qpos6 + Qlength/2

    maxOffsetX = [0,0,0]
    maxOffsetY = [0,0,0]
    maxOffsetXzpos = [0,0,0]
    maxOffsetYzpos = [0,0,0]
  


    for line in data[1]:
        if line[0] > Q4start and line[0] < Q4end:
            if math.fabs(line[5]) > maxOffsetX[0]:
                maxOffsetX[0] = math.fabs(line[5])
                maxOffsetXzpos[0] = line[0]

        if line[0] > Q5start and line[0] < Q5end:
            if math.fabs(line[5]) > maxOffsetX[1]:
                maxOffsetX[1] = math.fabs(line[5])
                maxOffsetXzpos[1] = line[0]

        if line[0] > Q6start and line[0] < Q6end:
            if math.fabs(line[5]) > maxOffsetX[2]:
                maxOffsetX[2] = math.fabs(line[5])
                maxOffsetXzpos[2] = line[0]

    for line in data[2]:

        if line[0] > Q4start and line[0] < Q4end:
            if math.fabs(line[6]) > maxOffsetY[0]:
                maxOffsetY[0] = math.fabs(line[6])
                maxOffsetYzpos[0] = line[0]

        if line[0] > Q5start and line[0] < Q5end:
            if math.fabs(line[6]) > maxOffsetY[1]:
                maxOffsetY[1] = math.fabs(line[6])
                maxOffsetYzpos[1] = line[0]

        if line[0] > Q6start and line[0] < Q6end:
            if math.fabs(line[6]) > maxOffsetY[2]:
                maxOffsetY[2] = math.fabs(line[6])
                maxOffsetYzpos[2] = line[0]


    #angular acceptance separately for x and y and for quads
    maxValsX = [xAngTriplet, (astra.xAngle*Qbore*1e+3)/(2*maxOffsetX[0]), (astra.xAngle*Qbore*1e+3)/(2*maxOffsetX[1]), (astra.xAngle*Qbore*1e+3)/(2*maxOffsetX[2])  ]
    maxValsY = [yAngTriplet, (astra.yAngle*Qbore*1e+3)/(2*maxOffsetY[0]), (astra.yAngle*Qbore*1e+3)/(2*maxOffsetY[1]), (astra.yAngle*Qbore*1e+3)/(2*maxOffsetY[2])  ]
    

    #get the minimal value
    astra.xAngularAcceptance = min(maxValsX)
    astra.yAngularAcceptance = min(maxValsY)

    return [astra.xAngularAcceptance, astra.yAngularAcceptance]     


def plotRef(arg,Ds, Pz):

    setFile.changeInputData("Q_grad(4)", arg[0])
    setFile.changeInputData("Q_grad(5)", arg[1])
    setFile.changeInputData("Q_grad(6)", arg[2])


    files = ["test0.ini","test1.ini","test2.ini"]
    data = []
    for file in files:
        setFile.changeInputData("Distribution", file)

        astra.process.stdin.write("./Astra " + astra.fileName + "\n")
        astra.process.stdin.flush()

        wait()

        data.append( astra.loadData("ref") )

    ang = checkAng(data)
    funkVal = astra.pointFocusing( [astra.getClosest(data[0]), astra.getClosest(data[1]) , astra.getClosest(data[2] ) ] )
    


    plt.figure(figsize=(10,5))
    plt.plot( [line[0] for line in data[0] ], [line[5] for line in data[0] ], color="black", label="0 offset, 0 angle" )
    plt.plot( [line[0] for line in data[1] ], [line[5] for line in data[1] ], color="blue", label="0 offset, x angle" )
    plt.plot( [line[0] for line in data[2] ], [line[6] for line in data[2] ], color="green", label="0 offset, y angle" )
    plt.plot([astra.setupLength, astra.setupLength], [-0.5,0.5], color='red', label=f"screen = {math.ceil(astra.setupLength*10000)/100} cm")
    plt.legend()
    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")
    plt.title(f"point to point focusing using triplet and 3 EMagnets for Pz={math.ceil(Pz*1e-6)} MeV, ang accept: {[math.ceil(num) for num in ang]} mrad,\n size of triplet= {math.ceil( 10*(0.9 - Ds[3]/100 + astra.bores[2]*4) )/10 } m, gradients of the EM: {[math.ceil(num) for num in arg[0:3]]} T/m\n D = {[Ds]} cm")
    plt.savefig(f"specialAssignment/finalF/solution{math.ceil(Pz*1e-6)}MeV.png", format='png', dpi=300)
    plt.close()



def funcSextet(arg):

    setFile.changeInputData("Q_grad(4)", arg[0])
    setFile.changeInputData("Q_grad(5)", arg[1])
    setFile.changeInputData("Q_grad(6)", arg[2])

    files = ["test1.ini","test2.ini"]
    data = []
    data.append([0,0,0,0,0,0])
    for file in files:
        setFile.changeInputData("Distribution", file)

        astra.process.stdin.write("./Astra " + astra.fileName + "\n")
        astra.process.stdin.flush()

        wait()
        
        datCurrent =  astra.loadData("ref") 
        bestLine = astra.getClosest(datCurrent)

        #print(f"setup length and closest:{bestLine[0]}, {astra.setupLength}")
        data.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )

    Sum = astra.pointFocusing(data)

    print(Sum)
    return Sum


def sextetFocusing(tripletData,D5, D6, D7):

    astra.changePositions(*tripletData[0:5])
    astra.changeMom(tripletData[5])

    Dmin = [-60, -60, -60]   #G4, G5, G6
    Dmax = [60, 60, 60]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    method = "Powell"
    tolerance = 1e-5

    # settings
    astra.setupLength = 0.9 +Qlength + D5 + Qlength + D6 + Qlength + D7
    

    astra.quadType(1)
    setFile.changeInputData("Q_pos(4)", str( 0.9 + Qlength/2 ))
    setFile.changeInputData("Q_pos(5)", str( 0.9 +Qlength + D5 + Qlength/2))
    setFile.changeInputData("Q_pos(6)", str( 0.9 +Qlength + D5 + Qlength + D6 + Qlength/2))
    setFile.changeInputData("ZSTOP", str( math.ceil( 10*astra.setupLength )/10  ) )


    res = sc.optimize.minimize(funcSextet, ( 30, -55, 30), method=method, tol=tolerance, bounds=bounds)

    if not res.success:
        print("Did not have success with finding solution for this setup")
        return 1

    print(res.x)

    plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,D7], tripletData[5])




if __name__ == "__main__":

    setFile = SettingsFile("parallelBeam")
    astra = Astra(setFile)


    # first find solution for as high momentum as possible with the triplet
    PZ = [2.5E+8, 3.0E+8, 3.5E+8, 4.0E+8, 4.5E+8, 5.0E+8, 5.5E+8,6.0E+8, 6.5E+8, 7.0E+8, 7.5E+8]
    data = []
    for Pz in PZ:
        data.append(tripletFocusing(Pz))


    #plt.scatter([math.ceil(num*1e-6) for num in PZ], [line[6] for line in data], color="blue")
    #plt.xlabel("Pz [MeV]")
    #plt.ylabel("f(x',y') [mrad^2]")
    #plt.title("Function values for point-point to parallel-parallel focusing\nfor quadrupole triplet with length of setup=0.9 m")
    #plt.savefig("specialAssignment/funkVals.png", format="png", dpi=300)
    #plt.show()

    df = pd.DataFrame(data)


    custom_header = ["D1 [cm]", "D2 [cm]", "D3 [cm]", "D4 [cm]", "setup length [m]", "Pz [eV/c]", "f(x',y') [mrad^2]", "x acceptance [mrad]", "y acceptance [mrad]", "beam ratio [-]"]
    #df.to_csv("output.csv", header=custom_header, index=False)


    # now switch on EMagnets
    setFile = SettingsFile("specAssign")
    astra = Astra(setFile)
    
    for dataForSextet in data:
        xAngTriplet = float(dataForSextet[-3])
        yAngTriplet = float(dataForSextet[-2])
        dataForSextet = [num/100 for num in dataForSextet[0:3] ] + dataForSextet[3:-1] 
        sextetFocusing(dataForSextet, 0.2,0.1,1.0)

    #plotRef([ 1.71413093e-01,  1.73534717e-02,  9.92670911e-01, -2.72794321e+01,  5.98140402e+01, -3.31527962e+01],
    #    [math.ceil(num*10000)/100 for num in dataForSextet[0:3]] + [dataForSextet[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2 ],
    #    dataForSextet[5]
    #)



