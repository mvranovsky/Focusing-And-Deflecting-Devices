#!/usr/bin/python3


from SettingsFile import SettingsFile
from Astra import Astra
import scipy as sc
import subprocess
import time
import pandas as pd
import math





def function(D, D1, hardEnd,momZ, switch):

    #print(f"input D2, D3 = {D}")
    
    dataCurrent = astra.runRef(D1,D[0], D[1],None, hardEnd, momZ , False)
    if dataCurrent == 1:
        print(f"current data == 1")
        return 1E+9
    
    if switch == "parallel":    
        sum = astra.parallelFocusing(dataCurrent)
    elif switch == "point":
        sum = astra.pointFocusing(dataCurrent)
    elif switch == "lineX":
        sum = astra.xLineFocusing(dataCurrent)
    elif switch == "lineY":
        sum = astra.yLineFocusing(dataCurrent)

    print(D, sum)
        
    return sum

def runStudy(inputFile):

    Dmin = [0.0,0.0]
    Dmax = [1.0,1.0]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    method = "Powell"
    tolerance = 1e-5
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]
    
    with open(inputFile, "r") as file:
        input = file.readlines()

    setups = []
    for i,line in enumerate(input):
        line = line.replace("\n","")
        line = line.split(" ")  
        num = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        setups.append(num)
        break

    results = [[], [], []]
    #print(setups)
    for i in range(3): 
        astra.quadType(i)
        for setup in setups:
            '''
            #parallel
            time1 = time.time()
            res1 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0],setup[3],  setup[4], "parallel") )
            time2 = time.time()
            if not res1.success:
                print(f"Did not obtain result for parallel focusing. The result turned out {res1.x}")
                continue 
            astra.plotRefXY(setup[0], *res1.x, setup[3],setup[4], f"Found solution for {fields[i]}, parallel focusing,\nPz= {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s:\nD2 = {math.ceil(res1.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res1.x[1]*1e+5)/1000} cm", f"parallel_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, 0 ,res1.x[0]*1e+2, res1.x[1]*1e+2] + findInfo(setup[0], *res1.x,None, setup[3], setup[4]))
            '''
            print(f"Working on setup with {fields[i]}.")
            #point
            time1 = time.time()
            res2 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "point"))
            time2 = time.time()
            if not res2.success:
                print(f"Did not obtain result for point focusing. The result turned out {res2.x}")
                continue
            astra.plotRefXY(setup[0], *res2.x, None, setup[3], setup[4], f"Found solution for {fields[i]}, point-point focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s: D2 = {math.ceil(res2.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res2.x[1]*1e+5)/1000} cm", f"pointpoint_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res2.x[0]*1e+2, res2.x[1]*1e+2] + astra.findInfo(setup[0], *res2.x,None, setup[3], setup[4]))
            
            #x line
            time1 = time.time()
            res3 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "lineX"))
            time2 = time.time()
            if not res3.success:
                print(f"Did not obtain result for line X focusing. The result turned out {res3.x}")
                continue
            astra.plotRefXY(setup[0], *res3.x, None, setup[3], setup[4], f"Found solution for {fields[i]}, line X focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s:\nD2 = {math.ceil(res3.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res3.x[1]*1e+5)/1000} cm", f"lineX_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res3.x[0]*1e+2, res3.x[1]*1e+2] + astra.findInfo(setup[0], *res3.x, None, setup[3], setup[4]))

            #y line
            time1 = time.time()
            res4 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "lineY"))
            time2 = time.time()
            if not res4.success:
                print(f"Did not obtain result for line Y focusing. The result turned out {res4.x}")
                continue
            plotRefXY(setup[0], *res4.x, None, setup[3], setup[4], f"Found solution for {fields[i]}, line Y focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s: D2 = {math.ceil(res4.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res4.x[1]*1e+5)/1000} cm", f"lineY_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res4.x[0]*1e+2, res4.x[1]*1e+2] + astra.findInfo(setup[0], *res4.x, None, setup[3], setup[4]))
            print(f"Timing with fields {fields[i]}: the entire process for setup: {setup} took {time2 - time1} s")
        
    df = pd.DataFrame(results[0])
    df.to_csv("topHatFields.csv", index=False)
    
    df = pd.DataFrame(results[1])
    df.to_csv("AstraFringeFields.csv", index=False)

    df = pd.DataFrame(results[2])
    df.to_csv("fieldProfiles.csv", index=False)
    return 

if __name__ == "__main__":

    myFile = SettingsFile("parallelBeam")
    astra = Astra(myFile)

    #astra.aperture(True)

    runStudy("../../MAXIMA/analyticalResultsD1.txt")

    print(f"All done with the assignment")

    '''
    var = 10

    proc = subprocess.Popen(
        ['/bin/bash'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

    proc.stdin.write("source /opt/intel/oneapi/setvars.sh >> temp.txt")
    proc.stdin.flush()

    proc.stdin.write("echo 'yes i am working' >> temp.txt")
    proc.stdin.flush()
    '''

