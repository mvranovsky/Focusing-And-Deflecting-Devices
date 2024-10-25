#!/usr/bin/python3


from AstraWrapper.SettingsFile import SettingsFile
from AstraWrapper.Astra import Astra
import scipy as sc
import subprocess
import time
#import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import ROOT 


def wait():
    while True:
        line = astra.process.stdout.readline()
        #print(line)
        if 'Goodbye' in line:
            return

def func(D, qType):

    quadType1(qType)
    astra.lengthQ1 = astra.AstraLengths[0]

    if qType == 2:
        setFile.changeInputData("Q_pos(1)",str(D[0]))
    else:
        setFile.changeInputData("Q_pos(1)",str(D[0] + astra.lengthQ1/2))

    ap1 = str(D[0]) + " " + str(astra.bores[0]*1E+3/2) + "\n" + str(D[0] + astra.lengthQ1 ) + " " + str(astra.bores[0]*1E+3/2)
    with open("aperture/aperture1.dat", "w") as file:
        file.write(ap1)    

    astra.runCommand("./Astra " + astra.fileName)

    wait()


    data = astra.loadData("ref")
    print(D[0], data[-1][5])

    return math.fabs(data[-1][5])

def checkXAcceptance(data, D1):

    maxX = 0.1
    bestLine = []
    for line in data:
        lowerLim = D1 - astra.lengthQ1/2
        upperLim = D1 + astra.lengthQ1/2

        if line[0] > lowerLim and line[0] < upperLim:
            if line[5] > maxX:
                maxX = float(line[5])
                bestLine = list(line)
    

    return astra.xAngle*astra.bores[0]*1e+3/maxX

def quadType1(i):

    if i == 0:
        setFile.changeInputData("Q_bore(1)", "1.0E-9")
        setFile.enable("Q_grad(1)")
        setFile.disable("Q_type(1)")
    elif i ==1:
        setFile.changeInputData("Q_bore(1)", str(astra.bores[0]))
        setFile.enable("Q_grad(1)")
        setFile.disable("Q_type(1)")
    elif i == 2:
        setFile.disable("Q_grad(1)")
        setFile.enable("Q_type(1)")

def findSolution(fields):

    Dmin = [0.0]
    Dmax = [0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    #method = "COBYLA"
    method = "Powell"
    tolerance = 1e-9
    #fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    # settings
    astra.setupLength = 1.0  #m
    Pz = 4.0E+8 #eV
    astra.lengthQ1 = astra.AstraLengths[0]
    
    
    setFile.changeInputData("ZSTOP",str(astra.setupLength))

    #first, i look only at the first quadrupole
    setFile.disable("Q_grad(2)")
    setFile.disable("File_Aperture(2)")
    setFile.disable("Q_grad(3)")
    setFile.disable("File_Aperture(3)")

    setFile.changeInputData("Distribution","test1.ini")
    astra.changeMom(Pz)  
    setFile.changeInputData("RUN", str(1))

    res = sc.optimize.minimize(func, (0.1), method=method, tol=tolerance, bounds=bounds, args = (fields) )

    #x = [0.202539]
    func([0.201734], 1)
    #func(res.x, 1)    
    data1 = astra.loadData("ref")

    #func(res.x, 2)
    func([0.201734], 2)
    data2 = astra.loadData("ref")

    #func(res.x, 0)
    func([0.201734], 0)
    data = astra.loadData("ref")


    plt.plot([row[0] for row in data], [row[5] for row in data], color="blue", label='laser ray trajectory(top hat fields) ')
    plt.plot([row[0] for row in data1], [row[5] for row in data1], color="red", label='laser ray trajectory(astra fringe fields)')
    plt.plot([row[0] for row in data2], [row[5] for row in data2], color="green", label='laser ray trajectory(field profiles)')
    plt.plot([0, astra.setupLength], [0,0], color='black', label='beamline')
    plt.xlabel("z [m]")
    plt.ylabel("x [mm]")
    plt.legend()
    plt.savefig("quad1Study/trajectoriesForQuadTypes.png", format='png', dpi=300)
    plt.show()
    #plt.close()

    print("solution: ", res.x[0])
    return res.x[0]

def checkMomentum():

    data = astra.loadData("0000")

    Pz = data[0][5]

    for line in data[1:-1]:
        num = math.sqrt( line[3]**2 + line[4]**2 + (Pz + line[5])**2 )/1000000
        print(num)



def study(fields , key , Vals, description):
    # D1 is a list of found positions for Q1 for different types of fields
    D1 = [0.201734, 0.203137 , 0.204737]


    
    # here define the ranges and number of bins
    histRange = [-0.5,0.5]  # mm
    nBins = 51

    hist = []

    histPx = ROOT.TH1D("histPx", "Initial distribution of p_{x} of particles in a beam; p_{x} [MeV/c];counts",40, -10, 10)
    histX = ROOT.TH1D("histX", "Initial distribution of x coordinate of particles in a beam; x [mm]; counts", 40, -0.0003, 0.0003)

    setFile.changeInputData("Distribution", astra.fileName + ".ini")
    setFile.changeInputData("sig_px", str(4000000))  

    for i in range(len(Vals)):
        hist.append( ROOT.TH1D("hist" + str(i),description[0] + " = " + Vals[i] + description[1] + "; x [mm]; counts", nBins, histRange[0], histRange[1]) )

        if key == "Q_grad(1)":
            setFile.changeInputData(key,str( float(Vals[i]) + 222  ) )
        else:
            setFile.changeInputData(key, Vals[i])

        subprocess.run("./generator " + astra.fileName, stderr=subprocess.PIPE,stdout=subprocess.PIPE, shell=True)
        
        if key == "Q_pos(1)":
            func([ D1[fields] ], fields)
        else:
            func([ D1[fields] ], fields)

        data = astra.loadData("0100")
        dataBegin = astra.loadData("0000")

        if len(data) != len(dataBegin):
            raise ValueError("Somethings wrong, input data length not equal to output.")


        Pz = data[0][5]
        for j, row in enumerate(data):

            if row[9] == 5:
                num = 1e+3*(row[0] + row[3]*row[2]/(Pz + row[5]))
                #num = dataBegin[j][0]
                hist[i].Fill(float(num))
                if i ==0:
                    histPx.Fill(float(dataBegin[j][3]*1e-6))
                    histX.Fill(float(dataBegin[j][0]*1e+3))




    file = ROOT.TFile("file.root", "recreate")
    file.cd()
    for h in hist:
        h.Write()
    histPx.Write()
    histX.Write()
    file.Close()


def surveyBottom(fields, D):


    step = 1E-6
    lowerRange = D - 50*step


    fVals0, fVals1, fVals2, dVals = [],[], [],[]
    for i in range(100):
        D_current = lowerRange + i*step
        dVals.append(D_current*100)
        setFile.changeInputData("H_max", "0.0001")
        setFile.changeInputData("Q_grad(1)", "222")        
        fVals0.append( func([D_current], fields ) )

        setFile.changeInputData("H_max", "0.001")
        fVals1.append( func([D_current], fields ) )

        setFile.changeInputData("Q_grad(1)", "223")
        setFile.changeInputData("H_max", "0.001")
        fVals2.append( func([D_current], fields ) )

    plt.scatter(dVals, fVals0, color='red', label='fields '+ str(fields)+ ", H_max = 0.0001,G=222" )
    plt.scatter(dVals, fVals1, color='blue', label='fields '+ str(fields) + ", H_max = 0.001, G=222" )
    plt.scatter(dVals, fVals2, color='green', label='fields '+ str(fields) + ", H_max = 0.001, G=223" )
    plt.legend()
    plt.xlabel("D1 [cm]")
    plt.ylabel("f(D1) [mm]")
    plt.show()




if __name__ == "__main__":

    setFile = SettingsFile("parallelBeam")
    astra = Astra(setFile)

    # default value 
    setFile.changeInputData("H_max", "0.0001")
    
    #surveyBottom(0,0.201734 )
    surveyBottom(1,0.203137 )

    
    lines = []
    with open("quad1Study/study.txt", "r") as file:
        lines = file.readlines()
    
    for line in lines:
        if line[0] == "#":
            continue
        line = line.split(" ")
        variable = line[0]
        key = line[1]
        rangeVal = [j for j in line[2:6] ]
        descript = [line[6], line[7]]

        for i in range(3):
            if i ==2:
                continue
            try:
                study(i, key, rangeVal, descript)
            except ValueError as e:
                print(f"An error occurred when trying to run study: {e}")

            # revert back to original settings
            setFile.changeInputData(key, 222)

            new_variable = ""
            if "1" in variable or "2" in variable:
                new_variable = variable[:-1]

            cmd = "mkdir -p quad1Study/" + new_variable

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error creating a directory {variable}: {e}")


            cmd = f'root -l -b -q \'drawHistsQuad1Study.C("{variable}",{i})\''

            try:
                subprocess.run(cmd,shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing ROOT function: {e}")

    
    #quadGradient1 Q_grad(1) 0 0.1 1 5 T/m
    #quadGradient2 Q_grad(1) 0 -0.1 -1 -5 T/m
    #spaceCharge IPart 1000 1001 5000 10000 particles


    


