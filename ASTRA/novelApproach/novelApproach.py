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
    #function that changes the momenta of rays
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
    #function which plots the rays 
    plt.plot([line[0] for line in data[0] ], [line[5] for line in data[0] ] , label='x offset, initial x angle', color='blue')
    plt.plot([line[0] for line in data[1] ], [line[6] for line in data[1] ] , label='y offset, initial y angle', color='red')

    plt.legend()
    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")

    plt.savefig('figure.png', format="png", dpi=300)
    plt.show()


def func(Pz,L1, L2, showPlot = False, returnAllData = False):
    #function that varies the Pz of rays to find the solution 

    inputDataName = ["test1.ini", "test2.ini"]

    allData = []
    outputMoreData = [[0,0,0,0,0,0]]
    #run both rays and get their trajectories
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

    # calculate their divergences at the final position
    Sum = astra.parallelFocusing(outputMoreData)
    print(Pz, Sum)

    if showPlot:
        plotRefXY(allData)

    # if returnAllData is set to True, return the entire trajectories for plotting
    if returnAllData:
        return allData

    return Sum

def funcFindD2(args,D1,Pz,L1, L2, showPlot = False):
    # very similar process as runRef() in Astra. Run 2 rays and calculate their sum of squares of divergences

    # change momentum of the rays
    inputDataName = ["test1.ini", "test2.ini"]
    changeMom(Pz*1000000, inputDataName[0])
    changeMom(Pz*1000000, inputDataName[1])

    # change positions according to arguments
    setFile.changeInputData("C_pos(1)",str(D1))
    setFile.changeInputData("A_pos(1)",str(D1))
    setFile.changeInputData("C_pos(2)",str(D1 + L1 + args[0] ))
    setFile.changeInputData("A_pos(2)",str(D1 + L2 + args[0] ))

    allData = []
    outputMoreData = [[0,0,0,0,0,0]]
    for i in range(len(inputDataName)):
        setFile.changeInputData("Distribution", inputDataName[i] )
        setFile.changeInputData("RUN", str(i+1))

        #run Astra with a ray
        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
            return 1E+9

        # get the output
        currentData = astra.loadData("ref", str(i+1) )
        allData.append(currentData)
        bestLine = astra.getClosest(currentData)
        if bestLine == 1:
            return 1E+9

        outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )
    #show plot if it is set to True
    if showPlot:
        plotRefXY(allData)

    # calculate the square root of sum of divergences
    Sum = astra.parallelFocusing(outputMoreData)
    print(args, Sum)
    return Sum





def funcFindD(args,Pz,L1, L2, showPlot = False):
    # very similar process as runRef() in Astra. Run 2 rays and calculate their sum of squares of divergences

    # change momentum of the rays
    inputDataName = ["test1.ini", "test2.ini"]
    changeMom(Pz*1000000, inputDataName[0])
    changeMom(Pz*1000000, inputDataName[1])

    # change positions according to arguments
    setFile.changeInputData("C_pos(1)",str(args[0]))
    setFile.changeInputData("A_pos(1)",str(args[0]))
    setFile.changeInputData("C_pos(2)",str(args[0] + L1 + args[1] ))
    setFile.changeInputData("A_pos(2)",str(args[0] + L2 + args[1] ))

    allData = []
    outputMoreData = [[0,0,0,0,0,0]]
    for i in range(len(inputDataName)):
        setFile.changeInputData("Distribution", inputDataName[i] )
        setFile.changeInputData("RUN", str(i+1))

        #run Astra with a ray
        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
            return 1E+9

        # get the output
        currentData = astra.loadData("ref", str(i+1) )
        allData.append(currentData)
        bestLine = astra.getClosest(currentData)
        if bestLine == 1:
            return 1E+9

        outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )
    #show plot if it is set to True
    if showPlot:
        plotRefXY(allData)

    # calculate the square root of sum of divergences
    Sum = astra.parallelFocusing(outputMoreData)
    print(args, Sum)
    return Sum


def doubletFocusing( D1, D2, L1,L2, limitValue = 0.001, FFFactor = 1, findD = False , findD2 = True, Pz = 500):
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
    #if findD is true, the minimization process will look for a solution by varying D1,D2
    if findD:

        Dmin = [ FFFactor*(r1),FFFactor*(r1+r2) ] 
        Dmax = [0.5, 0.5]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        # scipy method of minimization
        res = sc.optimize.minimize(funcFindD, (0.1,0.2),method=method, bounds=bounds, args=(Pz,L1,L2) )

        funkVal = funcFindD(res.x,Pz,L1,L2,showPlot = False)

        if funkVal > limitValue:
            raise ValueError(f"Did not obtain actual minimum, reached only function value {funkVal} for D = {res.x} m")

        return [*res.x, 1.2, Pz , funkVal ]

    elif findD2:

        Dmin = [ FFFactor*(r1+r2) ] 
        Dmax = [ 0.5]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        # scipy method of minimization
        res = sc.optimize.minimize(funcFindD2, (0.1,0.2),method=method, bounds=bounds, args=(D1,Pz,L1,L2) )

        funkVal = funcFindD2(res.x,D1,Pz,L1,L2,showPlot = False)

        if funkVal > limitValue:
            raise ValueError(f"Did not obtain actual minimum, reached only function value {funkVal} for D = {res.x} m")

        return [D1,*res.x, 1.2, Pz , funkVal ]

    else: #if findD is False, look for a solution for a specific D1,D2 by varying Pz (almost never found the solution)
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

    # define the position at which the focusing will be done. Because it is parallel focusing, it is only the maximum length
    astra.setupLength = 1.2

    # define the lengths and radii of the doublet
    l1 = 0.08
    l2 = 0.15
    
    r1 = 0.004
    r2 = 0.012

    #define which Pz to use
    #PZs = [ 850, 900, 950, 1000]
    PZs = [850, 500, 450, 400]
    PZs = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    data = []
    for Pz in PZs:

        # generate untapered quadrupoles
        generator.generateFieldMap(l1, 0.8, Qbore1 = r1,Qbore2 = r1 ,fieldType = 0, fileOutputName='quad1', nFMPoints = 21, showPlot = False)
        intGrad1 = generator.integrateGradProfile(showPlot = False)
        
        generator.generateFieldMap(l2, 0.8, Qbore1 = r2, Qbore2 = r2,fieldType = 0, fileOutputName='quad2', xFocusing=False,nFMPoints = 21, showPlot = False)
        intGrad2 = generator.integrateGradProfile(showPlot = False)
        print("Integrated gradients of untapered quads:",intGrad1, intGrad2)

        setFile.changeInputData("File_Efield(1)", "'cavity/3Dquad1'")
        setFile.changeInputData("File_Efield(2)", "'cavity/3Dquad2'")

        sol = []
        # look for a specific D1,D2 for parallel focusing of the untapered quads
        try:
            sol = doubletFocusing( 0.1, 0.3,l1, l2, Pz = Pz,findD = True)
        except Exception as e:
            print(f"exception: {e}")
            continue

        print("Found solution of D with untapered quads: ",sol)

        # save the output if there was a solution
        D1 = sol[0]
        D2 = sol[1]

        sol.append(D1 + l1 + D2 + l2)
        sol.append(l1)
        sol.append(l2)
        sol.append(intGrad1)
        sol.append(intGrad2)

        #taper the quadrupoles
        # first quad vals
        r11 = r1*D1/(D1 + l1)
        l1Prime = D1*( math.exp(l1/(D1+l1)) - 1)
        r12 = r1*(D1 + l1Prime)/(D1 + l1)


        # second quad vals
        r21 = r2*(D1+ l1Prime + D2)/(D1 + l1 + D2 + l2)
        l2Prime = (D1 + l1Prime + D2)*(math.exp(l2/(D1 + l1 + D2 + l2)) - 1)
        r22 = r2*(D1 + l1Prime + D2 + l2Prime)/(D1 + l1 + D2 + l2)


        # generate field maps with new radii of the tapered quads
        generator.generateFieldMap(l1Prime, 0.8, Qbore1 = r11,Qbore2 = r12 ,fieldType = 0, fileOutputName='quad1Tapered', nFMPoints = 21, showPlot = False)
        intGrad1 = generator.integrateGradProfile(showPlot = False)
        
        generator.generateFieldMap(l2Prime, 0.8, Qbore1 = r21, Qbore2 = r22,fieldType = 0, fileOutputName='quad2Tapered', xFocusing=False,nFMPoints = 21, showPlot = False)
        intGrad2 = generator.integrateGradProfile(showPlot = False)
        setFile.changeInputData("File_Efield(1)", "'cavity/3Dquad1Tapered'")
        setFile.changeInputData("File_Efield(2)", "'cavity/3Dquad2Tapered'")

        print("Integrated gradients of tapered quads:",intGrad1, intGrad2)

        #find a solution for the tapered quadrupoles. If no solution can be found, continue for different Pz
        sol3 = []
        try: 
            sol3 = doubletFocusing( 0.1, 0.3,l1Prime, l2Prime, Pz = Pz,findD2 = True)
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

    # save the output data to a .csv file with a specific name
    # the output data contain the values which are in header
    df = pd.DataFrame(data)
    header = ["D1 [m]", "D2 [m]", "setup length [m]", "Pz [MeV]", "f(x',y') [mrad^2]", "D1 + l1 + D2 + l2 [m]", "l1 [m]", "l2 [m]", "int Grad1 [T]", "int Grad2 [m]"]
    df.to_csv(f"outputNovelApproach2.csv", header=header, index=False)


