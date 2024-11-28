#!/usr/bin/python3


from AstraWrapper.SettingsFile import SettingsFile
from AstraWrapper.Astra import Astra
from AstraWrapper.Generator import Generator 
import scipy as sc
import subprocess
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
#import ROOT 
import sys


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


def tripletFocusing(Pz, D1 = None , defaultFieldType=1, limitValue = 0.0001, FFFactor = 1 ):
    method = "COBYLA"
    method = "Powell"
    tolerance = 1e-4
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    astra.quadType(defaultFieldType)
    astra.setupLength = 1.2  #m

    Dmin = [ FFFactor*astra.bores[0], FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
    Dmax = [0.7, 0.7, 0.7]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    result = []
    
    if D1 != None:
        Dmin = [FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
        Dmax = [0.9, 0.9]
        bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

        res = sc.optimize.minimize(func1, (0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(D1,Pz) )
        
        funcVal = func1(res.x,D1, Pz)
        if funcVal > limitValue:
            astra.setupLength = 3.0
            res = sc.optimize.minimize(func1, (0.1,0.1), method="Powell", tol=tolerance, bounds=bounds, args=(D1, Pz) )
        
        result = list(res.x)
        result.insert(0,D1)
    else:
        res = sc.optimize.minimize(func, (0.1,0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(Pz) )
        result = list(res.x)
    

    try:
        beamRatio = astra.beamRatio(*result, None, astra.setupLength, Pz)
        acc = astra.checkAngleAcceptance(*result, None, astra.setupLength, Pz)
        astra.quadType(defaultFieldType)
        funcVal = astra.parallelFocusing( astra.runRef(*result, None, astra.setupLength,Pz, False ) )
        astra.plotRefXY(*result, None, astra.setupLength, Pz)

        return [*[math.ceil(num*10000000)/10000000 for num in result],None, astra.setupLength,Pz, funcVal, *acc, beamRatio]
    except:
        raise ValueError("Did not find solution.")




if __name__ == "__main__":

    setFile = SettingsFile("novelApproach")
    astra = Astra(setFile)
    generator = Generator('novelApproach')


    generator.generateFieldMap(0.036, 0.777, gradAtStartP=222, gradAtEndP=222, fileOutputName='quad1', nFMPoints = 20)
    generator.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94, fileOutputName='quad2', nFMPoints = 20)
    generator.generateFieldMap(0.1, 0.855, gradAtStartP=57, gradAtEndP=57, fileOutputName='quad3', nFMPoints = 20)


    # default value 
    setFile.changeInputData("H_max", "0.001")

    astra.quadType(3)

    sol = tripletFocusing( Pz = 5E+8, D1 = 0.15,defaultFieldType = 3 )
    print(sol)


    generator.generateFieldMap(0.036, 0.777, gradAtStartP=244, gradAtEndP=200, fileOutputName='quad1', nFMPoints = 20)
    generator.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94, fileOutputName='quad2', nFMPoints = 20)
    generator.generateFieldMap(0.1, 0.855, gradAtStartP=57, gradAtEndP=57, fileOutputName='quad3', nFMPoints = 20)




    
