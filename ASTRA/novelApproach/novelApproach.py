#!/usr/bin/python3


from AstraWrapper import SettingsFile
from AstraWrapper import Astra
from AstraWrapper import Generator 
import scipy as sc
import subprocess
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
#import ROOT 
import sys

def changeMom(Pz,name):

    with open(name, "r") as file:
        line = file.readlines()[0].split()

    if Pz >= 100 and Pz <= 1000:
        Pz = Pz*1000000



    xAngle = 1  #mrad
    yAngle = 1
    #momentum update
    if name == "test1.ini":
        line[3] = str(xAngle*Pz*1e-3)

    if name == "test2.ini":
        line[4] = str(yAngle*Pz*1e-3)

    line[5] = str(Pz)

    lineTogether = ''
    for num in line:
        lineTogether += num + " "

    with open(name, "w") as file:
        file.write(lineTogether)


def plotRefXY(data):



    plt.plot([line[0] for line in data[0] ], [line[5] for line in data[0] ] , label='x offset, initial x angle', color='blue')
    plt.plot([line[0] for line in data[1] ], [line[6] for line in data[1] ] , label='y offset, initial y angle', color='red')

    plt.legend()
    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")

    plt.savefig('figure.png', format="png", dpi=300)
    plt.show()


def func(Pz,L1, L2, showPlot = False, returnAllData = False):

    inputDataName = ["test1.ini", "test2.ini"]

    allData = []
    outputMoreData = [[0,0,0,0,0,0]]
    for i in range(len(inputDataName)):
        changeMom(Pz[0]*1000000, inputDataName[i])
        setFile.changeInputData("Distribution", inputDataName[i] )
        setFile.changeInputData("RUN", str(i+1))

        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
            return 1E+9

        currentData = astra.loadData("ref", str(i+1) )

        allData.append( currentData )
        bestLine = astra.getClosest(currentData)
        if bestLine == 1:
            return 1E+9

        outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )


    Sum = astra.parallelFocusing(outputMoreData)
    print(Pz, Sum)

    if showPlot:
        plotRefXY(allData)

    if returnAllData:
        return allData

    return Sum

def funcFindD(args,Pz,L1, L2, showPlot = False):


    inputDataName = ["test1.ini", "test2.ini"]
    changeMom(Pz*1000000, inputDataName[0])
    changeMom(Pz*1000000, inputDataName[1])

    # change positions accordingly
    setFile.changeInputData("C_pos(1)",str(args[0]))
    setFile.changeInputData("A_pos(1)",str(args[0]))
    setFile.changeInputData("C_pos(2)",str(args[0] + L1 + args[1] ))
    setFile.changeInputData("A_pos(2)",str(args[0] + L2 + args[1] ))

    allData = []
    outputMoreData = [[0,0,0,0,0,0]]
    for i in range(len(inputDataName)):
        setFile.changeInputData("Distribution", inputDataName[i] )
        setFile.changeInputData("RUN", str(i+1))

        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
            return 1E+9

        currentData = astra.loadData("ref", str(i+1) )
        allData.append(currentData)
        bestLine = astra.getClosest(currentData)
        if bestLine == 1:
            return 1E+9

        outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )

    if showPlot:
        plotRefXY(allData)

    Sum = astra.parallelFocusing(outputMoreData)
    print(args, Sum)
    return Sum


def doubletFocusing( D1, D2, L1,L2, limitValue = 0.0001, FFFactor = 1, findD = False , Pz = 500):
    method = "COBYLA"
    method = "Powell"
    tolerance = 1e-8

    astra.setupLength = 1.2
    setFile.changeInputData("ZSTOP", str(1.2) )

    # change positions accordingly
    setFile.changeInputData("C_pos(1)",str(D1))
    setFile.changeInputData("A_pos(1)",str(D1))
    setFile.changeInputData("C_pos(2)",str(D1 + L1 + D2 ))
    setFile.changeInputData("A_pos(2)",str(D1 + L2 + D2 ))
    #changeMom(Pz, "test1.ini")
    #changeMom(Pz, "test2.ini")

    funkVal = 0
    if findD:
        Dmin = [ FFFactor*(r1),FFFactor*(r1+r2) ] 
        Dmax = [0.5, 0.5]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(funcFindD, (0.1,0.2),method=method, bounds=bounds, args=(Pz,L1,L2) )

        funkVal = funcFindD(res.x,Pz,L1,L2,showPlot = False)

        if funkVal > limitValue:
            raise ValueError(f"Did not obtain actual minimum, reached only function value {funkVal} for D = {res.x} m")

        return [*res.x, 1.2, Pz , funkVal ]

    else:
        Pzmin = [100 ]  #MeV
        Pzmax = [1500]
        bounds = [(low, high) for low, high in zip(Pzmin, Pzmax)]

        # change positions accordingly
        setFile.changeInputData("C_pos(1)",str(D1))
        setFile.changeInputData("A_pos(1)",str(D1))
        setFile.changeInputData("C_pos(2)",str(D1 + L1 + D2 ))
        setFile.changeInputData("A_pos(2)",str(D1 + L2 + D2 ))

        res = sc.optimize.minimize(func, (300),method=method, bounds=bounds, args=(L1,L2,False))
        funkVal = func(res.x,L1,L2, False)

        if funkVal > limitValue:
            raise ValueError(f"Did not obtain actual minimum, reached only function value {funkVal} for Pz = {res.x[0]} MeV")



        return [D1,D2, 1.2, res.x[0] , funkVal ]




if __name__ == "__main__":

    setFile = SettingsFile("novelApproach")
    astra = Astra(setFile)
    generator = Generator('novelApproach')

    astra.setupLength = 1.2


    l1 = 0.08
    l2 = 0.15
    
    r1 = 0.004
    r2 = 0.012

    #PZs = [ 850, 900, 950, 1000]
    PZs = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    data = []
    for Pz in PZs:

        generator.generateFieldMap(l1, 0.8, Qbore1 = r1,Qbore2 = r1 ,fieldType = 0, fileOutputName='quad1', nFMPoints = 21, showPlot = False)
        intGrad1 = generator.integrateGradProfile(showPlot = False)
        
        generator.generateFieldMap(l2, 0.8, Qbore1 = r2, Qbore2 = r2,fieldType = 0, fileOutputName='quad2', xFocusing=False,nFMPoints = 21, showPlot = False)
        intGrad2 = generator.integrateGradProfile(showPlot = False)
        print("Integrated gradients of untapered quads:",intGrad1, intGrad2)

        setFile.changeInputData("File_Efield(1)", "'cavity/3Dquad1'")
        setFile.changeInputData("File_Efield(2)", "'cavity/3Dquad2'")

        sol = []
        try:
            sol = doubletFocusing( 0.1, 0.3,l1, l2, Pz = Pz,findD = True)
            #sol = [0.24460857727063723, 0.0002194907587622068, 1.2, 650, 7.261961855621302e-17]
        except Exception as e:
            print(f"exception: {e}")
            continue

        print("Found solution of D with untapered quads: ",sol)

        D1 = sol[0]
        D2 = sol[1]

        sol.append(D1 + l1 + D2 + l2)
        sol.append(l1)
        sol.append(l2)
        sol.append(intGrad1)
        sol.append(intGrad2)

        # first quad vals
        r11 = r1*D1/(D1 + l1)
        l1Prime = D1*( math.exp(l1/(D1+l1)) - 1)
        r12 = r1*(D1 + l1Prime)/(D1 + l1)


        # second quad vals
        r21 = r2*(D1+ l1Prime + D2)/(D1 + l1 + D2 + l2)
        l2Prime = (D1 + l1Prime + D2)*(math.exp(l2/(D1 + l1 + D2 + l2)) - 1)
        r22 = r2*(D1 + l1Prime + D2 + l2Prime)/(D1 + l1 + D2 + l2)



        generator.generateFieldMap(l1Prime, 0.8, Qbore1 = r11,Qbore2 = r12 ,fieldType = 0, fileOutputName='quad1Tapered', nFMPoints = 21, showPlot = False)
        intGrad1 = generator.integrateGradProfile(showPlot = False)
        
        generator.generateFieldMap(l2Prime, 0.8, Qbore1 = r21, Qbore2 = r22,fieldType = 0, fileOutputName='quad2Tapered', xFocusing=False,nFMPoints = 21, showPlot = False)
        intGrad2 = generator.integrateGradProfile(showPlot = False)
        setFile.changeInputData("File_Efield(1)", "'cavity/3Dquad1Tapered'")
        setFile.changeInputData("File_Efield(2)", "'cavity/3Dquad2Tapered'")

        print("Integrated gradients of tapered quads:",intGrad1, intGrad2)

        sol3 = []
        try: 
            sol3 = doubletFocusing( 0.1, 0.3,l1Prime, l2Prime, Pz = Pz,findD = True)
        except Exception as e:
            print("exception: ", e)
        
        sol3.append(D1 + l1Prime + D2 + l2Prime)
        sol3.append(l1Prime)
        sol3.append(l2Prime)
        sol3.append(intGrad1)
        sol3.append(intGrad2)
        print("sol: ", sol) 
        print("sol3:", sol3)
        data.append( list(sol) )
        data.append(list(sol3) )


    df = pd.DataFrame(data)
    header = ["D1 [m]", "D2 [m]", "setup length [m]", "Pz [MeV]", "f(x',y') [mrad^2]", "D1 + l1 + D2 + l2 [m]", "l1 [m]", "l2 [m]", "int Grad1 [T]", "int Grad2 [m]"]

    df.to_csv(f"outputNovelApproach.csv", header=header, index=False)




    #sol:  [0.3141409247274675, 0.03684367018821179, 1.2, 850, 1.0986445633217993e-20, 0.6509845949156794]
    #sol3: [0.3128260516242866, 0.06141230057748473, 1.2, 850, 1.6877110001384084e-14, 0.5938677173583706]

    #sol:  [0.31566391878793365, 0.0381584279038233, 1.2, 850, 2.6477957370242217e-06, 0.653822346691757]
    #sol3: [0.3144820776078279, 0.062476468678611854, 1.2, 850, 6.821411275294118e-06, 0.5970117879651433]
