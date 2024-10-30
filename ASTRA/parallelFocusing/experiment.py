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


def getClosest( currentData , hardEnd):

    bestLine = []
    closest = 0.1
    for j in range(len(currentData)):
        dist = math.fabs(currentData[j][0] - hardEnd)
        if dist < closest:
            bestLine = list(currentData[j])
            closest = float(dist)

    if closest >0.1:
        print(f"Reference particle {i} did not get to the end of setup.")
        return 1

    return bestLine



def loadData(arg, fillnum):
    data = []
    fillNumber = "00" + str(fillnum)
    #assuming setup length
    with open(setFile.fileName + "." + arg + ".00" + str(fillnum),"r") as file:
        for line in file:
            lineSplitted = line.split()
            data.append([float(num) for num in lineSplitted])

    return data
    

def runRef( D1, D2, D3, D4, hardEnd,momZ, moreData):

    inputDataName = ["test1.ini", "test2.ini"]
    outputMoreData = [[0,0,0,0,0,0]]
    for i in range(len(inputDataName)):
        setFile.changeInputData("Distribution", inputDataName[i] )
        setFile.changeInputData("RUN", str(i+ 1))

        subprocess.run(["./Astra", setFile.fileName + ".in"], capture_output=True, text=True)
        
        bestLine = getClosest( loadData("ref", 1), hardEnd )

        outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )

    return outputMoreData


def func(D, hardEnd, Pz):

    data = runRef(*D, None, hardEnd, Pz, False)
    if data == 1:
        return 1E+9

    summ = (data[1][3]*1e+3/data[1][5])**2 + (data[2][4]*1e+3/data[2][5])**2 

    print(D, summ)

    return summ



def study():

    Dmin = [ 0,0,0 ]
    Dmax = [0.4, 0.4, 0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]


    Pz = 4.0E+8
    hardEnd = 1.0

    res = sc.optimize.minimize(func, (0.1,0.1,0.1), method="COBYLA", bounds=bounds, args=(hardEnd, Pz) )

    funcVal = func(res.x, hardEnd, Pz)

    if funcVal < 0.0001:
        return True
    else:
        return False



if __name__ == "__main__":


    setFile = SettingsFile("parallelBeam")
    
    for i in range(100):
        study()

