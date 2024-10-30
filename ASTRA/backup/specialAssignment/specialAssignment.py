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



Qlength = 0.08
Qbore = 0.017



def func(D, Pz):
    data = []
    try:
        data = astra.runRef(*D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return 1E+9
    else:
        Sum = astra.parallelFocusing(data)
        print(D, Sum)
        return Sum

def func1(D,D1, Pz):

    data = []
    try:
        data = astra.runRef(D1, *D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return 1E+9
    else:
        Sum = astra.parallelFocusing(data)
        print(D, Sum)
        return Sum

def func2(D,Pz):

    data = []
    try:
        data = astra.runRef(*D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return 1E+9
    else:
        Sum = (data[1][3]*1e+3/data[1][5])**2 + (data[2][4]*1e+3/data[2][5])**2 # + (data[1][0]/data[2][1] - 1)
        print(D, Sum)
        return Sum


def tripletFocusing(Pz, D1 = None , beamRatio = False, limitValue = 0.0001, FFFactor = 1 ):
    method = "Powell"
    method = "COBYLA"
    tolerance = 1e-4
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    astra.quadType(1)
    astra.setupLength = 0.9  #m

    Dmin = [ FFFactor*astra.bores[0], FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
    Dmax = [0.6, 0.6, 0.6]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    result = []
    
    if D1 != None:
        Dmin = [FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
        Dmax = [0.6, 0.6]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(func1, (0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(D1,Pz) )

        funcVal = func1(res.x,D1, Pz)
        if funcVal > limitValue:
            astra.setupLength = 1.2
            res = sc.optimize.minimize(func1, (0.1,0.1), method="Powell", tol=tolerance, bounds=bounds, args=(D1, Pz) )

        result = list(res.x)
        result.insert(0,D1)
    elif beamRatio == True:
        res = sc.optimize.minimize(func2, (0.1,0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(Pz) )
        result = list(res.x)
    else:
        res = sc.optimize.minimize(func, (0.1,0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(Pz) )
        result = list(res.x)


    print(result)
    funcVal = func(result, Pz)

    beamRatio = astra.beamRatio(*result, None, astra.setupLength, Pz)
    acc = astra.checkAngleAcceptance(*result, None, astra.setupLength, Pz)

    print(f"Now about to plot the result...")
    astra.plotRefXY(*result, None, astra.setupLength, Pz, f"Solution triplet point to parallel focusing with Pz = {Pz/1000000} MeV\n Ds = {[math.ceil(d*10000)/100 for d in result]} cm", f"specialAssignment/tripletFocusing/solution{math.ceil(Pz*1e-6)}MeV")

    return [*[math.ceil(num*10000)/100 for num in result],None, astra.setupLength,Pz, funcVal,*acc, beamRatio]


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

        astra.runAstra()

        data.append( astra.loadData("ref") )

    ang = checkAng(data)
    funkVal = astra.pointFocusing( [astra.getClosest(data[0]), astra.getClosest(data[1]) , astra.getClosest(data[2] ) ] )
    
    subprocess.run(["mkdir", "-p", f"specialAssignment/sextetFocusing/D5:{math.ceil(Ds[4]*100)}cm,D6:{math.ceil(Ds[5]*100)}cm"])
    

    plt.figure(figsize=(10,5))
    plt.plot( [line[0] for line in data[0] ], [line[5] for line in data[0] ], color="black", label="0 offset, 0 angle" )
    plt.plot( [line[0] for line in data[1] ], [line[5] for line in data[1] ], color="blue", label="0 offset, x angle" )
    plt.plot( [line[0] for line in data[2] ], [line[6] for line in data[2] ], color="green", label="0 offset, y angle" )
    plt.plot([astra.setupLength, astra.setupLength], [-0.5,0.5], color='red', label=f"screen = {math.ceil(astra.setupLength*10000)/100} cm")
    plt.legend()
    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")
    plt.title(f"point to point focusing using triplet and 3 EMagnets for Pz={math.ceil(Pz*1e-6)} MeV, ang accept: {[math.ceil(num) for num in ang]} mrad,\n size of triplet= {math.ceil( 100*(0.9 - Ds[3]/100 + astra.bores[2]*4) )/100 } m, gradients of the EM: {[math.ceil(num) for num in arg[0:3]]} T/m\n D = {Ds[0:4] + [num*100 for num in Ds[4:]]} cm")
    plt.savefig(f"specialAssignment/sextetFocusing/D5:{math.ceil(Ds[4]*100)}cm,D6:{math.ceil(Ds[5]*100)}cm/solution{math.ceil(Pz*1e-6)}MeV.png", format='png', dpi=300)
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

def funcSextet2(arg, D5, D6):

    setFile.changeInputData("Q_grad(4)", arg[0])
    setFile.changeInputData("Q_grad(5)", arg[1])
    setFile.changeInputData("Q_grad(6)", arg[2])
    astra.setupLength = 0.9 +Qlength + D5 + Qlength + D6 + Qlength + arg[3]

    setFile.changeInputData("ZSTOP", str(astra.setupLength ) )

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



def sextetFocusing(tripletData,D5, D6, D7, limitValue = 0.0001, moveD7 = False):

    astra.changePositions(*tripletData[0:5])
    astra.changeMom(tripletData[5])


    # settings
    astra.setupLength = 0.9 +Qlength + D5 + Qlength + D6 + Qlength + D7
    

    astra.quadType(1)
    setFile.changeInputData("Q_pos(4)", str( 0.9 + Qlength/2 ))
    setFile.changeInputData("Q_pos(5)", str( 0.9 +Qlength + D5 + Qlength/2))
    setFile.changeInputData("Q_pos(6)", str( 0.9 +Qlength + D5 + Qlength + D6 + Qlength/2))
    setFile.changeInputData("ZSTOP", str( math.ceil( 10*astra.setupLength )/10  ) )


    method = "COBYLA"
    tolerance = 1e-5

    Dmin = [0, -60, 0]   #G4, G5, G6
    Dmax = [60, 0, 60]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]


    res = sc.optimize.minimize(funcSextet, ( 30, -55, 30), method=method, tol=tolerance, bounds=bounds)
    funcValue = funcSextet(res.x)

    plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,D7], tripletData[5])
    print(res.x, funcValue)

    if funcValue < limitValue:
        print(f"Found a solution, the gradients of the resistive quadrupoles: {res.x} T/m")
        #plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,D7], tripletData[5])
        return


    print(f"Could not find a solution which maximizes angular acceptance at the same as keeping D7 constant. Now looking for solution with gradients being (-60, 60).")
    

    if moveD7:
        res = sc.optimize.minimize(funcSextet2, ( 30, -55, 30, 1.2), method=method, tol=tolerance, bounds=bounds, args=(D5,D6))
        funcValue = funcSextet2(res.x, D5,D6)

        plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,math.ceil(100*res.x[3])/100 ], tripletData[5])
        
        if funcValue < limitValue:
            print(f"Found a solution with varied D7, the gradients turned out: {res.x[0:3]} T/m, D7 = {math.ceil(res.x[3]*100000)/1000} cm.")
            return

        Dmin = [-60, -60, -60]   #G4, G5, G6
        Dmax = [60, 60, 60]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(funcSextet, ( -30, 55, -30), method=method, tol=tolerance, bounds=bounds)
        funcValue = funcSextet(res.x)

        plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,D7], tripletData[5])

        if funcValue < limitValue:
            print(f"Found a solution with reversed gradients. Found solution: {res.x} T/m")
            return
    else:
        Dmin = [-60, -60, -60]   #G4, G5, G6
        Dmax = [60, 60, 60]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(funcSextet, ( -30, 55, -30), method=method, tol=tolerance, bounds=bounds)
        funcValue = funcSextet(res.x)

        plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,D7], tripletData[5])

        if funcValue < limitValue:
            print(f"Found a solution with reversed gradients. Found solution: {res.x} T/m")
            return


        print(f"Could not find a solution with gradients being less than 60 in absolute value. Now looking for solution with D7 being varied.")
        Dmin = [0, -60, 0, 0]   #G4, G5, G6
        Dmax = [60, 0, 60, 3.0]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(funcSextet2, ( 30, -55, 30, 1.2), method=method, tol=tolerance, bounds=bounds, args=(D5,D6))
        funcValue = funcSextet2(res.x, D5,D6)

        plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,math.ceil(100*res.x[3])/100 ], tripletData[5])
        
        if funcValue < limitValue:
            print(f"Found a solution with varied D7, the gradients turned out: {res.x[0:3]} T/m, D7 = {math.ceil(res.x[3]*100000)/1000} cm.")
            return



    print(f"Could not find a solution with gradients being less than 60 in absolute value. Now looking for solution with D7 being varied.")
    Dmin = [-60, -60, -60, 0]   #G4, G5, G6
    Dmax = [60, 60, 60, 3.0]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    res = sc.optimize.minimize(funcSextet2, ( -30, 55, -30, 1.2), method=method, tol=tolerance, bounds=bounds, args=(D5,D6))
    funcValue = funcSextet2(res.x, D5,D6)
  
    plotRef(res.x, [math.ceil(num*10000)/100 for num in tripletData[0:3]] + [math.ceil( (tripletData[4] - float(setFile.readOption("Q_pos(3)")) -astra.AstraLengths[2]/2)*10000 ) /100] + [D5,D6,res.x[3]], tripletData[5])
    
    if funcValue < limitValue:
        print(f"Finally found a solution with varying gradients and D7. The gradients: {res.x[0:3]} T/m, D7 = {math.ceil(res.x[3]*100000)/1000} cm")
        return
    else:
        print(f"Could not find a solution for Pz = {Pz} MeV. Leaving...")
        return



if __name__ == "__main__":

    setFile = SettingsFile("parallelBeam")
    astra = Astra(setFile)

    
    D1s = []
    for k in range(66): #66
        D1s.append( astra.bores[0] + k*0.002 )

    PZ = []
    for k in range(80): #80
        PZ.append(2.0E+8 + k*1.0E+7)


    # first find solution for as high momentum as possible with the triplet
    #PZ = [ 2.5E+8, 3.0E+8, 3.5E+8, 4.0E+8, 4.5E+8, 5.0E+8, 5.5E+8,6.0E+8, 6.5E+8, 7.0E+8, 7.5E+8, 8.0E+8, 8.5E+8]
    #PZ = [5.0E+8]
    data = []
    for D1 in D1s:
        for Pz in PZ:
            print(f"Now running D1 = {D1*100} cm and Pz = {Pz*1e-6} MeV")
            data.append(tripletFocusing(Pz, D1 = D1, FFFactor = 1))



    df = pd.DataFrame(data)


    custom_header = ["D1 [cm]", "D2 [cm]", "D3 [cm]", "D4 [cm]", "setup length [m]", "Pz [eV/c]", "f(x',y') [mrad^2]", "x acceptance [mrad]", "y acceptance [mrad]", "beam ratio [-]"]
    df.to_csv("output.csv", header=custom_header, index=False)
    '''

    # now switch on EMagnets
    setFile = SettingsFile("specAssign")
    astra = Astra(setFile)
    
    pairs = [ [0.05, 0.05],[0.1, 0.1], [0.15,0.15] , [0.2,0.2], [0.25,0.25], [0.3, 0.3] , [0.1,0.2], [0.15, 0.1],
    [0.1, 0.15], [0.15, 0.2], [0.2, 0.15], [0.2, 0.25], [0.25,0.2], [0.1, 0.3], [0.3 , 0.1]
    ]



    for pair in pairs:
        for dataForSextet in data:
            xAngTriplet = float(dataForSextet[-3])
            yAngTriplet = float(dataForSextet[-2])
            dataForSextet = [num/100 for num in dataForSextet[0:3] ] + dataForSextet[3:-1] 
            
            sextetFocusing(dataForSextet, pair[0], pair[1],1.0)


    print(f"All plots are finished, goodbye.")
    '''
            
