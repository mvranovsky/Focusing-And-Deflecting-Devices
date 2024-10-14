#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import os
import re
import yaml
import sys
import subprocess
import math
import glob
import time
import random
from statistics import mean


# # Parallel focusing 
# This is python code written in jupyter notebook which implements different methods for point-point to parallel-parallel focusing. It uses software ASTRA, a program to simulate beam dynamics in accelerators. Programs in this notebook run in the same directory as are the ASTRA executables, mainly generator and Astra. 
# 
# The initial information are regarding the input file to Astra and information about reference particles. I used 5 different reference particles to determine the focusing properties of a setup- the first particle with 0 offset and 0 angle, moving along the z axis. This particle should not move in the transverse direction. Next 2 particles would be with initial offsets in the x and y directions respectively, but because this is point-point focusing, I am not using these. Last 2 have angles in the x and y direction respectively.
# 
# The magnets that are used are permanent quadrupole magnets with set gradients, lengths and bore diameters. These parameters can be changed, but for now they are set to values of 3 quadrupole magnets in LLR laboratory. The variables which will be changing are distances between them and the initial momentum. D1 is the distance from the source to the 1. quadrupole magnet. Realistically, D1 is only up to fringe fields which are magnetic fields outside the magnet's bores (reach 3*bore size in ASTRA). This option can be changed using TopHatShapedQuads() function. D2 and D3 are distances between first 2 and last 2 magnets in sequence. Last variable that can be changed is the initial longitudinal momentum of particles.
# 
# For running beam simulations, one can define it's initial parameters like spread of transverse momenta, spread of longitudinal energy, spread of offsets in the x and y directions as well as in the longitudinal direction. Also number of initial particles, space charge, secondary particle emission or other parameters can be changed in file parallelBeam.in.
# 

# In[2]:


fileName = "parallelBeam"
longitudalEnergy = "5.0E+8" #eV

#offsets and angles for reference particles
xoffset = "2.0E-4" #m
yoffset =  "2.0E-4" #m
xmom = "1.0E+6" #eV
ymom = "1.0E+6" #eV


#parameters of magnets 
#all methods are equivalent in integrated gradient along z


#input parameters of the beam
nParticles = "500"
sig_z=0.1    #mm
sig_Ekin=3E+3     #keV
sig_x=2.0E-3    #mm
sig_y=2.0E-3    #mm
#sig_px =3E+6    #eV
#sig_py =3E+6    #eV

sig_xAngle = 1  #mrad
sig_yAngle = 1  #mrad









def func(D, D1,D4, mom):

    dataCurrent = runRef(D1, D[0], D[1], D4, mom, False)
    if dataCurrent == 1:
        return 1E+9
    sumX = xLineFocusing(dataCurrent)
    #print(D, sumX)

    return sumX


# In[37]:


D3vals = []
funcVals = []
funcValsY = []


# In[38]:


def func3(D,D1, D2, D4, momZ):
    
    dataCurrent = runRef(D1,D2, D[0], D4, momZ, False)
    if dataCurrent == 1:
        return 1E+9
    sumX = xLineFocusing(dataCurrent)
    #print(D[0], sumX)

    D3vals.append(D[0]*1e+2)
    funcVals.append(sumX)
    
    return sumX


# In[219]:


Dmin = [0.0, 0.0]
Dmax = [1., 1.]
bounds = [(low, high) for low, high in zip(Dmin, Dmax)]


with open("../../MAXIMA/analyticalResults.txt", "r") as file:
    input = file.readlines()


setups = []
for line in input:
    line = line.replace("\n","")
    line = line.split(" ")  
    num = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])]
    setups.append(num)

proc = subprocess.Popen(
    ['/bin/bash'], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True
)

#proc.stdin.write("source /opt/intel/oneapi/setvars.sh\n")
proc.stdin.flush()


results = [[],[]]
for setup in setups:
    time1 = time.time()
    #res = sc.optimize.minimize(func, (0.1, 0.1),method="COBYLA", bounds=bounds,tol=1e-4, args=(setup[0],setup[3], setup[4] ))
    print(runRef(*setup, False))
    time2 = time.time()
    plotRefXY3(setup[0], *res.x, setup[3], setup[4], f"solution: {res.x}")
    print(f"Timing: {time2 - time1}")
    break
    
    
    '''
    results[0] = list(D3vals)
    results[1] = list(funcVals)
    D3vals.clear()
    funcVals.clear()

    plt.scatter(results[0], results[1], label=methods[0] + " tol=1e-4")
    plt.xlabel("D3 [cm]")
    plt.ylabel("f(D3) [mrad^2]")
    plt.xlim(0,15)
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    plt.scatter(results[0], results[1], label=methods[0] + " tol=1e-4")
    plt.scatter(results[2], results[3], label=methods[1] + " tol=1e-4")
    plt.scatter(results[4], results[5], label=methods[2] + " tol=1e-4")
    plt.xlabel("D3 [cm]")
    plt.ylabel("f(D3) [mrad^2]")
    plt.xlim(7.25,7.4)
    plt.ylim(0,0.00001)
    '''



'''
time1 = time.time()
for i in range(100):
    D3 = (7.25 +i*0.15/100)/100
    resX= func3(setups[0][0], setups[0][1],D3,setups[0][3],setups[0][4])
    print(D3*100, resX)
    D3vals.append(D3*100)
    funcVals.append(resX)

time2 = time.time()
print(time2-time1)

plt.scatter(D3vals, funcVals, label="x angle")
plt.xlabel("D3 [cm]")
plt.ylabel("f(D3) [mrad^2]")
plt.xlim(7.25,7.4)
plt.ylim(0,0.0002)

plt.legend()
plt.show()


plt.plot(D3vals, funcVals, label="x angle")
plt.xlabel("D3 [cm]")
plt.ylabel("f(D3) [mrad^2]")
plt.xlim(7.25,7.4)
plt.ylim(0,0.0001)

plt.legend()
plt.show()
'''

proc.stdin.close()
#proc.wait()  # This waits for the shell process to terminate


# In[70]:


plt.scatter(D3vals, funcVals, label="tol=1e-4")
plt.xlabel("D3 [cm]")
plt.ylabel("f(D3) [mrad^2]")
plt.xlim(7.25,7.4)
plt.ylim(0,0.0002)

plt.legend()
plt.show()


plt.plot(D3vals, funcVals, label="tol=1e-4")
plt.xlabel("D3 [cm]")
plt.ylabel("f(D3) [mrad^2]")
plt.xlim(7.25,7.4)
plt.ylim(0,0.0002)

plt.legend()
plt.show()


# ## ComparisonAnaNum()
# For this comparison, the fringe fields stay off. The first run is with analytical solution, the second run is with found numerical solution. The 2 results can be compared in a table below. The differences in D2,D3 are in Delta D2 and Delta D3. Parameters of runs are also there.

# In[39]:


def someFunc():
    '''
    args = sys.argv
    args.pop(0)
    if len(args) != 1:
        print(f"more than 1 argument")
    tolerance = float(args[0])
    '''
    topHatShapedQuads(0)
    #boundaries for D2, D3    
    Dmin = [0.0, 0.0]
    Dmax = [0.4, 0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    tolerance=1e-2
    D1 = [0.1]
    #for i in range(1,21):
    #    D1.append(i/100)
    
    D2P1, D2P2, D2NM1, D2NM2, D2C1,D2C2 = [], [], [],[], [], []
    D3P1, D3P2, D3NM1, D3NM2, D3C1,D3C2 = [], [], [],[], [], []
    funkValP1, funkValP2, funkValNM1, funkValNM2, funkValC1, funkValC2 = [], [], [],[] ,[],[]
    timeP1, timeP2, timeNM1, timeNM2, timeC1, timeC2 = [], [], [], [], [], []
    
    for d1 in D1:    
        time1 = time.time()
        res = sc.optimize.minimize(func, (0.1, 0.1),method="Powell", bounds=bounds,tol=tolerance, args=( d1, 1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        print(f"Number of iterations: {res.niter}")
        #D2P1.append(res.x[0])
        #D3P1.append(res.x[1])
        #funkValP1.append(parallelFocusing(runRef(d1, *res.x, 4.5E+8, False)))
        #timeP1.append(time2 -time1)
        
        time1 = time.time()
        bounds2 = [(res.x[0]-0.05, res.x[0]+0.05),(res.x[1]-0.05, res.x[1]+0.05)]
        res = sc.optimize.minimize(func, (random.uniform(res.x[0]-0.05, res.x[0]+0.05), random.uniform(res.x[1]-0.05, res.x[1]+0.05)),method="Powell", bounds=bounds2,tol=tolerance, args=( d1,1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        #D2P2.append(res.x[0])
        #D3P2.append(res.x[1])
        #funkValP2.append(angleCalculation(runRef(d1, *res.x, 4.5E+8, False)))
        #timeP2.append(time2 -time1)
        #------------------------------------------------------------------
        time1 = time.time()
        res = sc.optimize.minimize(func, (0.1, 0.1),method="Nelder-Mead", bounds=bounds,tol=tolerance, args=( d1,1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        #D2NM1.append(res.x[0])
        #D3NM1.append(res.x[1])
        #funkValNM1.append(angleCalculation(runRef(d1, *res.x, 4.5E+8, False)))
        #timeNM1.append(time2 -time1)
        
        time1 = time.time()
        bounds2 = [(res.x[0]-0.05, res.x[0]+0.05),(res.x[1]-0.05, res.x[1]+0.05)]
        res = sc.optimize.minimize(func, (random.uniform(res.x[0]-0.05, res.x[0]+0.05), random.uniform(res.x[1]-0.05, res.x[1]+0.05)),method="Nelder-Mead", bounds=bounds2,tol=tolerance, args=( d1,1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        #D2NM2.append(res.x[0])
        #D3NM2.append(res.x[1])
        #funkValNM2.append(angleCalculation(runRef(d1, *res.x, 4.5E+8, False)))
        #timeNM2.append(time2 -time1)
        #------------------------------------------------------------------
        
        time1 = time.time()
        res = sc.optimize.minimize(func, (0.1, 0.1),method="COBYLA", bounds=bounds,tol=tolerance, args=( d1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        #D2C1.append(res.x[0])
        #D3C1.append(res.x[1])
        #funkValC1.append(angleCalculation(runRef(d1, *res.x, 4.5E+8, False)))
        #timeC1.append(time2 -time1)
        
        time1 = time.time()
        bounds2 = [(res.x[0]-0.05, res.x[0]+0.05),(res.x[1]-0.05, res.x[1]+0.05)]
        res = sc.optimize.minimize(func, (random.uniform(res.x[0]-0.05, res.x[0]+0.05), random.uniform(res.x[1]-0.05, res.x[1]+0.05)),method="COBYLA", bounds=bounds2,tol=tolerance, args=( d1,1, 4.5E+8 ))
        time2 = time.time()
        plotRefXY(d1, *res.x, 1, 4.5E+8)
        print(f"Time to find solution: {time2-time1} s")
        #D2C2.append(res.x[0])
        #D3C2.append(res.x[1])
        #funkValC2.append(angleCalculation(runRef(d1, *res.x, 4.5E+8, False)))
        #timeC2.append(time2 -time1)
    
    results = [D1, 
               D2P1,D3P1, funkValP1, timeP1, 
               D2P2, D3P2, funkValP2, timeP2, 
               D2NM1, D3NM1, funkValNM1, timeNM1, 
               D2NM2, D3NM2, funkValNM2, timeNM2, 
               D2C1, D3C1, funkValC1, timeC1, 
               D2C2, D3C2, funkValC2, timeC2]
    df = pd.DataFrame(results)
    df.to_csv(f"table{tolerance}.csv",index=False)


# In[101]:


someFunc()


# In[145]:


def runPlots(file):

    df = pd.read_csv(file)
    results = df.values.tolist()
    plt.plot(results[0][:], results[3][:], label='Powell', color='blue')
    plt.plot(results[0][:], results[7][:], label='Powell w. constraints', color='red')
    plt.plot(results[0][:], results[11][:], label='Nelder-Mead', color='green')
    plt.plot(results[0][:], results[15][:], label='Nelder-Mead w. constraints', color='yellow')
    plt.plot(results[0][:], results[19][:], label='COBYLA', color='purple')
    plt.plot(results[0][:], results[23][:], label='COBYLA w. constraints', color='pink')

    plt.xlabel('D1 [mm]')
    plt.ylabel('f(D1) [mrad]')
    #plt.xlim(0.085, 0.1)
    #plt.ylim(0., 0.0)
    plt.yscale('log')
    plt.title(f"found minima for tolerance {file.replace('tables/table','').replace('.csv','')}")
    plt.legend()
    plt.show()

    

    plt.plot(results[0][:], results[4][:], label='Powell', color='blue')
    plt.plot(results[0][:], results[8][:], label='Powell w. constraints', color='red')
    plt.plot(results[0][:], results[12][:], label='Nelder-Mead', color='green')
    plt.plot(results[0][:], results[16][:], label='Nelder-Mead w. constraints', color='yellow')
    plt.plot(results[0][:], results[20][:], label='COBYLA', color='purple')
    plt.plot(results[0][:], results[24][:], label='COBYLA w. constraints', color='pink')

    plt.xlabel('D1 [mm]')
    plt.ylabel('time [s]')
    #plt.xlim(0.085, 0.1)
    #plt.ylim(0., 0.0)
    #plt.yscale('log')
    plt.title(f"found minima for tolerance {file.replace('tables/table','').replace('.csv','')}")
    plt.legend()
    plt.show()

    tables = os.listdir("tables/")
    tables = sorted(tables)
    data = {"func values [mrad]": ['Powell', "Powell w. constraints", "Nelder-Mead", "Nelder-Mead w. constraints", "COBYLA", "COBYLA w. constraints" ]}
    df = pd.DataFrame(data)
    data = {"time [s]": ['Powell', "Powell w. constraints", "Nelder-Mead", "Nelder-Mead w. constraints", "COBYLA", "COBYLA w. constraints" ]}    
    df2 = pd.DataFrame(data)
    for table in tables:
        d = pd.read_csv("tables/" + table)
        results = d.values.tolist()            
        
        df['tol = ' + table.replace("table","").replace(".csv","")] = [ mean(results[3][:]), mean(results[7][:]), mean(results[11][:]), mean(results[15][:]), mean(results[19][:]), mean(results[23][:]) ]
        df2['tol = ' + table.replace("table","").replace(".csv","")] = [ mean(results[4][:]), mean(results[8][:]), mean(results[12][:]), mean(results[16][:]), mean(results[20][:]), mean(results[24][:]) ]
        
    return df, df2


# In[148]:


df, df2 = runPlots("tables/table0.001.csv")
#function values
df


# In[147]:


#timing
df2


# In[38]:


def study(inputFile):

    update()

    #boundaries for D2, D3    
    Dmin = [0.0,0.0]
    Dmax = [0.4,0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    
    '''
    results = D1, D2, D3, momZ,f_MAXIMA, f_topHat_found, f_fringeFields, f_fringeFields_found, f_fieldProfile, f_fieldProfile_found
    '''
    with open(inputFile, "r") as file:
        setups = file.readlines()

    
    setup = []
    for line in setups:
        line = line.replace("\n","")
        line = line.split(" ")  
        setup.append(line)
    #setup = [[0.1, 0.056362, 0.101014 ,4.5E+8]]

    results = []
    #sigInAccept = ""
    #sigOutAccept = ""

    for line in setup:
        current_res = []
        current_res.extend(line)
        topHatShapedQuads(0)
        current_res.append(angleCalculation(runRef(float(line[0]), float(line[1]), float(line[2]), float(line[3]), False)))
        plotRefXY1(float(line[0]), float(line[1]), float(line[2]), float(line[3]), f"MAXIMA solution for top hat field, setup: {line}, f={current_res[-1]} mrad")
        
        res = sc.optimize.minimize(func, (0.15, 0.15),method="Nelder-Mead", bounds=bounds,tol=1e-3, args=(float(line[0]), float(line[3])))
        current_res.append(angleCalculation(runRef(float(line[0]),*res.x , float(line[3]), False)))
        plotRefXY1(float(line[0]), *res.x, float(line[3]), f"Numerical solution for top hat field, setup: {res.x} with f={current_res[-1]} mrad")

        topHatShapedQuads(1)
        current_res.append(angleCalculation(runRef(float(line[0]), float(line[1]), float(line[2]), float(line[3]), False)))
        plotRefXY1(float(line[0]), float(line[1]), float(line[2]), float(line[3]), f"MAXIMA solution Astra fringe fields, setup: {line}, f={current_res[-1]} mrad")
        
        res = sc.optimize.minimize(func, (0.15, 0.15),method="Nelder-Mead", bounds=bounds,tol=1e-3, args=(float(line[0]), float(line[3] )))
        current_res.append(angleCalculation(runRef(float(line[0]),*res.x , float(line[3]), False)))
        plotRefXY1(float(line[0]), *res.x, float(line[3]), f"Numerical solution for Astra fringe fields, setup: {res.x} with f={current_res[-1]} mrad")

        topHatShapedQuads(2)
        current_res.append(angleCalculation(runRef(float(line[0]), float(line[1]), float(line[2]), float(line[3]), False)))
        plotRefXY1(float(line[0]), float(line[1]), float(line[2]), float(line[3]), f"MAXIMA solution field profile, setup: {line}, f={current_res[-1]} mrad")
        
        res = sc.optimize.minimize(func, (0.15, 0.15),method="Nelder-Mead", bounds=bounds,tol=1e-3, args=(float(line[0]), float(line[3] )))
        current_res.append(angleCalculation(runRef(float(line[0]),*res.x , float(line[3]), False)))
        plotRefXY1(float(line[0]), *res.x, float(line[3]), f"Numerical solution for field profile, setup: {res.x} with f={current_res[-1]} mrad")

        #current_res.extend(checkAngleAcceptance()[0:2])
        results.append(current_res)


    df = pd.DataFrame(results)
    df.to_csv("table.csv",index=False)

    #with open("results.txt" , "w") as file:
    #    file.write(sigInAccept)
    #with open("errors.txt" , "w") as file:
    #   file.write(sigOutAccept)
    

    return df


# In[44]:


'''
args = sys.argv
args.pop(0)
if len(args) != 1:
    print(f"more than 1 argument")
file = args[0]
'''
#file = "../../MAXIMA/analyticalResultsP.txt"
#study(file)



# In[40]:


def comparisonAnaNum(setupFileName, minimalFunVal):

    update()
    #boundaries for D2, D3    
    Dmin = [0.0,0.0]
    Dmax = [0.3,0.3]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    
    with open("../../MAXIMA/" + setupFileName, "r") as file:
        stringdata = file.readlines()
        
    analyticData = []
    for line in stringdata:
        line = line.replace("\n","")
        line = line.split(" ")  
        analyticData.append(line)

    '''
    resultsTable = {
        "" : ["D1 [m]", "D2 [m]", "D3 [m]", "Pz [eV]", 
              "F_ana [mrad]", "F_num [mrad]", 
              "Delta D2 [mm]" , "Delta D3 [mm]",
              "setup length [m]", 
              "Angle accept. x [mrad]", "Angle accept. y [mrad]",
              "beam size x [mm]", "beam size y [mm]","beam ratio [-]",
              "active particles for sig_xyAngle = 1 mrad [%],
              "number of func. evaluations [-]",
              "x angle [mrad], y angle [mrad]",
             ]
    }
    '''
    finalTable = []
    
    resultSetups = ""
    errorSetups = ""
    initialAng = []
    
    for row in analyticData:
        #first run the analytical solution and show plots
        topHatShapedQuads(True)
        changeMom(sig_xAngle, sig_yAngle, float(row[3]), -1, -1)
        
        sumAna = angleCalculation(runRef(float(row[0]),float(row[1]), float(row[2]), float(row[3]), False))
        plotRefXY1(float(row[0]),float(row[1]), float(row[2]), float(row[3]), f"Analytic results, '{row}', top hat fields")
        
        res = sc.optimize.minimize(func, (0.15, 0.15),method="Powell", bounds=bounds,tol=1e-6, args=(float(row[0]), float(row[3])))
        #res = sc.optimize.minimize(func, (0.15, 0.15), method="Powell", bounds=bounds, options={'ftol': 1e-8}, args=(float(row[0]), float(row[3])))
        #res = sc.optimize.minimize(func, (0.15, 0.15), method="Powell", bounds=bounds, options={'xtol': 1e-8}, args=(float(row[0]), float(row[3])))

        if not res.success or res.fun > minimalFunVal:
            print(f"Could not find numerical solution for d1 = {row[0]} and pz = {row[3]}. Skipping.")
            errorSetups += row[0] + " " + row[1] + " " + row[2] + " " + row[3] + "\n"
            continue
        
        sumNum = angleCalculation(runRef(float(row[0]),*res.x , float(row[3]), False))
        plotRefXY1(float(row[0]),*res.x, float(row[3]), f"Numerical results, ['{row[0]}', '{res.x[0]}', '{res.x[1]}', '{row[3]}'], top hat fields")
        resultSetups += row[0] + " " + str(res.x[0]) + " " + str(res.x[1]) + " " + row[3] + "\n"
        row.append(sumAna)
        row.append(sumNum)
        
        DeltaD2 = math.fabs(float(row[1]) - res.x[0])*1000 
        DeltaD3 = math.fabs(float(row[2]) - res.x[1])*1000
        row.append( DeltaD2 )
        row.append( DeltaD3 )
        
        row.append( setupSize(float(row[0]), *res.x ))
        angleAcceptance = checkAngleAcceptance(float(row[0]), *res.x, float(row[3]))
        row.append(angleAcceptance[0])
        row.append(angleAcceptance[1])
        row.append(angleAcceptance[2])
        row.append(angleAcceptance[3])
        row.append(angleAcceptance[2]/angleAcceptance[3])
        row.append(angleAcceptance[4])
        row.append(res.nfev)
        finalTable.append(row)
        

    print(resultSetups)
    print(errorSetups)
    if resultSetups != "":
        with open("results.txt", "w") as file:
            file.write(resultSetups)
    if errorSetups != "":
        with open("errors.txt","w") as file:
            file.write(errorSetups)

    
    return finalTable


# In[41]:


def plotResultsAcceptance(file, file2):

    df = pd.read_csv(file)
    list = df.values.tolist()

    data = []
    for i in range(160):
        data.append(list[i*18])

    
    for i in range(10):
        D1 = [value[0] for value in data[i*16:(i+1)*16]]        
        Pz = [value[3]*1e-6 for value in data[i*16:(i+1)*16]]
        acceptX = [value[4] for value in data[i*16:(i+1)*16]]
        acceptY = [value[5] for value in data[i*16:(i+1)*16]]

        plt.scatter(Pz, acceptX, color='blue', label='acceptance X')
        plt.scatter(Pz, acceptY, color='red', label='acceptance Y')
        plt.legend()
        plt.xlabel("Pz [MeV]")
        plt.ylabel("acceptance [mrad]")
        plt.title(f"Plot of acceptances w.r.t. momentum for D1 = {i +1} cm")
        plt.show()

    
    df = pd.read_csv(file2)
    list = df.values.tolist()

    data = []
    for i in range(160):
        data.append(list[i*18])

    for i in range(16):
        D1 = [value[0]*1e+2 for value in data[i*10:(i+1)*10]]
        Pz = [value[3]*1e-6 for value in data[i*10:(i+1)*10]]
        acceptX = [value[4] for value in data[i*10:(i+1)*10]]
        acceptY = [value[5] for value in data[i*10:(i+1)*10]]

        plt.scatter(D1, acceptX, color='blue', label='acceptance X')
        plt.scatter(D1, acceptY, color='red', label='acceptance Y')
        plt.legend()
        plt.xlabel("D1 [cm]")
        plt.ylabel("acceptance [mrad]")
        plt.title(f"Plot of acceptances w.r.t. D1 for Pz = {250 + i*50} MeV")
        plt.show()

    



# In[42]:


def plotResultsSigmaSpread(file):


    df = pd.read_csv(file)
    data = df.values.tolist()

    #results = D1, D2, D3, momZ, xAccept, yAccept,xAng_sig, yAng_sig, beam size X, beam size Y, percentagePassed,


    for i in range(1):
        #D1 = [value[0] for value in data[i*18:(i+1)*18]]        
        #Pz = [value[3]*1e-6 for value in data[i*18:(i+1)*18]]
        #acceptance
        angleSig = [value[6] for value in data]
        beamX = [value[8] for value in data]
        beamY = [value[9] for value in data]
        perPassed = [value[10] for value in data]

        plt.scatter(angleSig, perPassed, color='blue', label='percentage of passed particles')
        #plt.legend()
        plt.xlabel("initial sigma spread [mrad]")
        plt.ylabel("particles passed [%]")
        plt.title(f"Plot of particles passed w.r.t. initial sigma angle spread for setup with acceptance {data[0][4]}, {data[0][5]}")
        plt.show()  
        
        plt.scatter(angleSig, beamX, color='blue', label='beam size X')
        plt.scatter(angleSig, beamY, color='red', label='beam size Y')
        plt.legend()
        plt.xlabel("initial sigma spread [mrad]")
        plt.ylabel("beam size [mm]")
        plt.title(f"Plot of beam sizes w.r.t. to initial angle spread  for setup with acceptance {data[0][4]}, {data[0][5]}")
        plt.show()
        


        


# In[43]:


def plotResultsD1(data):

    for i in range(10): 
        D1 = [row[0]*100 for row in data[i*16: i*16 + 15]]
    
        Pz = [value[3]*1e-6 for value in data[i*16:i*16 + 15]]
        deltaD2 = [value[6] for value in data[i*16:i*16 + 15]]
        deltaD3 = [value[7] for value in data[i*16:i*16 + 15]]
        funkAna = [value[4] for value in data[i*16: i*16 + 15]]
        funkNum = [value[5] for value in data[i*16: i*16 + 15]]
        setupLength = [value[8] for value in data[i*16:i*16 + 15]]
        xAcceptance = [value[9] for value in data[i*16:i*16 + 15]]
        yAcceptance = [value[11] for value in data[i*16:i*16 + 15]]

        #deltas w.r.t. D1, Pz
        plt.scatter(Pz, deltaD2, color='blue', label='Delta D2')
        plt.scatter(Pz, deltaD3, color='red', label='Delta D3')
        plt.legend()
        plt.xlabel("Pz [MeV]")
        plt.ylabel("delta [mm]")
        plt.title(f"Plot of differences between MAXIMA results and ASTRA minimization for D1 = {1*i +1} cm")
        plt.show()
    
        plt.scatter(Pz,funkAna , color='blue', label='MAXIMA solution')
        plt.scatter(Pz, funkNum, color='red', label='ASTRA minimization')
        plt.legend()
        plt.xlabel("Pz [MeV]")
        plt.ylabel("f(D2,D3) [mrad]")
        plt.title(f"Plot of function results between MAXIMA and ASTRA minimization for D1 = {1*i +1} cm")
        plt.show()

        
        #setup length w.r.t. D1, Pz
        plt.scatter(Pz, setupLength, color='blue', label='length of setup [m]')
        plt.legend()
        plt.xlabel("Pz [MeV]")
        plt.ylabel("length [m]")
        plt.title(f"Length of setup with ASTRA minimized solution for D1 = {i + 1} cm")
        plt.show()
        
    
        plt.scatter(Pz, xAcceptance, color='blue', label='Acceptance x ')
        plt.scatter(Pz, yAcceptance, color='red', label='Acceptance y')
        plt.legend()
        plt.xlabel("Pz [MeV]")
        plt.ylabel("initial angle [mrad]")
        plt.title(f"Plot of acceptance with ASTRA minimized solution for D1 = {i + 1} cm")
        plt.show()
    



    return


# In[44]:


def plotResultsPz(data):

    for i in range(16): 
        D1 = [row[0]*100 for row in data[i*10: i*10 + 9]]

        Pz = [value[3]*1e-6 for value in data[i*10: i*10 + 9]]
        deltaD2 = [value[6] for value in data[i*10: i*10 + 9]]
        deltaD3 = [value[7] for value in data[i*10: i*10 + 9]]
        funkAna = [value[4] for value in data[i*10: i*10 + 9]]
        funkNum = [value[5] for value in data[i*10: i*10 + 9]]
        setupLength = [value[8] for value in data[i*10: i*10 + 9]]
        xAcceptance = [value[9] for value in data[i*10: i*10 + 9]]
        yAcceptance = [value[11] for value in data[i*10: i*10 + 9]]
        '''
        #deltas w.r.t. D1, Pz
        plt.scatter(D1, deltaD2, color='blue', label='Delta D2')
        plt.scatter(D1, deltaD3, color='red', label='Delta D3')
        plt.legend()
        plt.xlabel("D1 [cm]")
        plt.ylabel("delta [mm]")
        plt.title(f"Plot of differences between MAXIMA results and ASTRA minimization for Pz = {250 + i*50} MeV")
        plt.show()
    
        plt.scatter(D1,funkAna , color='blue', label='MAXIMA solution')
        plt.scatter(D1, funkNum, color='red', label='ASTRA minimization')
        plt.legend()
        plt.xlabel("D1 [cm]")
        plt.ylabel("f(D2,D3) [mrad]")
        plt.title(f"Plot of function results between MAXIMA and ASTRA minimization for Pz = {250 + i*50} MeV")
        plt.show()

        
        #setup length w.r.t. D1, Pz
        plt.scatter(D1, setupLength, color='blue', label='length of setup [m]')
        plt.legend()
        plt.xlabel("D1 [cm]")
        plt.ylabel("length [m]")
        plt.title(f"Length of setup with ASTRA minimized solution for Pz = {250 + i*50} MeV")
        plt.show()
        '''
    
        plt.scatter(D1, xAcceptance, color='blue', label='Acceptance x ')
        plt.scatter(D1, yAcceptance, color='red', label='Acceptance y')
        plt.legend()
        plt.xlabel("D1 [cm]")
        plt.ylabel("initial angle [mrad]")
        #plt.yscale('log')
        plt.title(f"Plot of acceptance with ASTRA minimized solution for Pz = {250 + 50*i} MeV")
        plt.show()
    



    return


# In[77]:


'''
args = sys.argv
args.pop(0)
if len(args) != 1:
    print(f"more than 1 argument")

dataFileName = str(args[0])
matrix = comparisonAnaNum(dataFileName, 1)
df = pd.DataFrame(matrix)
df.to_csv('table.csv', index=False)


matrix = comparisonAnaNum("analyticalResultsP_1.txt", 100)
df = pd.DataFrame(matrix)
df.to_csv('table.csv', index=False)
matrix = [[1,2,3,4],[5,6,7,8]]
df = pd.DataFrame(matrix)
df.to_csv('table.csv', index=False)
df = pd.read_csv('../runParallel/Run2/tablePz.csv')
#df = pd.read_csv('../runParallel/Run2/tableD1.csv')
list_of_lists = df.values.tolist()

plotResultsPz(list_of_lists)
'''


# In[78]:


def getResults(setupFileName):
    #this a function which compares analytical solution with numerical solution and with regards to fringe fields

    update()

    #boundaries for D2, D3    
    Dmin = [0.0,0.0]
    Dmax = [0.4,0.4]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    
    D1 = []
    Pz = []
    for i in range(1,4):
        D1.append(0.1*i)
        Pz.append(2E+8 + i*2E+7)

    
    results = ""
    resultsTable = {
        "" : ["D1 [m]", "D2 [m]", "D3 [m]", "Pz [eV]", "F_num [keV]","D [m]"]
    }

    df = pd.DataFrame(resultsTable)
    i = 1

    for d1 in D1:
        print(f"Running {d1}")
        for pz in Pz:
            print(f"Running {pz}")            
            topHatShapedQuads(True)
            res = sc.optimize.minimize(func, (0.15, 0.15),method="Powell", bounds=bounds,tol=1e-8, args=(d1, pz))
            sum = angleCalculation(runRef(d1,*res.x , pz, False))
            results += str(d1) + " " + str(res.x[0]) + " " + str(res.x[1]) + " " + str(pz) + "\n"
            fill4DGraph(d1, res.x[0], res.x[1], pz, sum)
            df['setup ' + str(i)] = [d1, res.x[0], res.x[1], pz, sum, d1+res.x[0] +res.x[1] ]
        
        i += 1
        
    with open(setupFileName,"w") as file:
        file.write(results)

    
    return df
    


# In[79]:


#df = getResults("results.txt")
#df.to_csv('resFigs/table.csv', index=False)


# ## Functions to study sensitivity
# The following functions are implemented with a goal to study how sensitive or stable a solution is when some parameters or variables are being alternated. runAna() studies variability in D1, D2, D3, Pz and initial Px, Py. The input of the function is a solution- a functioning setup. For each variable function prints a graph with logarithmic x axis representing change in the variable, the logarithmic y axis returns relative change in the function (angleCalculation() ). 
# Below that is another function which studies the initial x and y offset. 

# In[80]:


#analytic: 0.10 0.1767908617405159 0.1859304244423013 700000000
def runAna(D1,D2, D3, momZ, switch):  
    topHatShapedQuads(True)
    update()
    input = [D1, D2, D3, momZ]
    #--------------------------------------------------------------------------------------------------------------------------
    print(f"Varying D1 in range from 1 cm to 10 mikrometers")
    difs = [0.01, 0.005,0.003, 0.002, 0.001, 0.0005,0.0003, 0.0002, 0.0001,0.00005 ,0.00003, 0.00002, 0.00001]
    graphDataMX = []
    graphDataMY = []
    graphDataPX = []
    graphDataPY = []
    if switch == 1:
        for dif in difs:
            dataControl = runRef(D1,D2,D3,momZ, False)
            dataTestP = runRef(D1+dif,D2,D3, momZ, False)
            dataTestM = runRef(D1-dif,D2, D3, momZ, False)
    
            relChangeM = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestM))*100/angleCalculation(dataControl)
            graphDataMX.append(dif)
            graphDataMY.append(relChangeM)
            relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
            graphDataPX.append(dif)
            graphDataPY.append(relChangeP)        
    
        plt.scatter(graphDataPX, graphDataPY, color='blue', label='D1 + change')
        plt.scatter(graphDataMX, graphDataMY, color='red', label='D1 - change')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
        plt.xlabel('change [m]')
        plt.ylabel('rel. change in angle sum [%]')
        plt.title(f"varying D1 with input '{input}'")
        plt.legend()
        plt.show()

    elif switch == 2:
        #--------------------------------------------------------------------------------------------------------------------------
        print(f"Varying D2 in range from 1 cm to 10 mikrometers")
        
        for dif in difs:
            dataControl = runRef(D1,D2,D3,momZ, False)
            dataTestP = runRef(D1,D2+dif,D3, momZ, False)
            dataTestM = runRef(D1,D2-dif, D3, momZ, False)
    
            relChangeM = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestM))*100/angleCalculation(dataControl)
            graphDataMX.append(dif)
            graphDataMY.append(relChangeM)
            relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
            graphDataPX.append(dif)
            graphDataPY.append(relChangeP)        
    
        plt.scatter(graphDataPX, graphDataPY, color='blue', label='D2 + change')
        plt.scatter(graphDataMX, graphDataMY, color='red', label='D2 - change')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
        plt.xlabel('change [m]')
        plt.ylabel('rel. change in angle sum [%]')
        plt.title(f"varying D2 with input '{input}'")
        plt.legend()
        plt.show()


    elif switch == 3:
        #--------------------------------------------------------------------------------------------------------------------------
        print(f"Varying D3 in range from 1 cm to 10 mikrometers")
        
        for dif in difs:
            dataControl = runRef(D1,D2,D3,momZ, False)
            dataTestP = runRef(D1,D2,D3+dif, momZ, False)
            dataTestM = runRef(D1,D2, D3-dif, momZ, False)
    
            relChangeM = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestM))*100/angleCalculation(dataControl)
            graphDataMX.append(dif)
            graphDataMY.append(relChangeM)
            relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
            graphDataPX.append(dif)
            graphDataPY.append(relChangeP)        
    
        plt.scatter(graphDataPX, graphDataPY, color='blue', label='D3 + change')
        plt.scatter(graphDataMX, graphDataMY, color='red', label='D3 - change')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
        plt.xlabel('change [m]')
        plt.ylabel('rel. change in angle sum [%]')
        plt.title(f"varying D3 with input '{input}'")
        plt.legend()
        plt.show()

    elif switch == 4:
        #--------------------------------------------------------------------------------------------------------------------------
        print(f"Varying Pz in range from 50 MeV to 1 eV")
        difMoms = [5e+7, 3e+7 , 1e+7, 5e+6, 3e+6, 1e+6, 5e+5, 3e+5, 1e+5, 5e+4, 3e+4, 1e+4, 5e+3, 3e+3, 1e+3, 5e+2 , 3e+2, 1e+2, 50, 30 ,10, 5, 3, 1] #from 50 MeV to 1 keV
        
        for dif in difMoms:
            dataControl = runRef(D1,D2,D3,momZ, False)
            dataTestP = runRef(D1,D2,D3, momZ+dif, False)
            dataTestM = runRef(D1,D2, D3, momZ-dif, False)
    
            relChangeM = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestM))*100/angleCalculation(dataControl)
            graphDataMX.append(dif)
            graphDataMY.append(relChangeM)
            relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
            graphDataPX.append(dif)
            graphDataPY.append(relChangeP)        
    
        plt.scatter(graphDataPX, graphDataPY, color='blue', label='Pz + change')
        plt.scatter(graphDataMX, graphDataMY, color='red', label='Pz - change')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
        plt.xlabel('change [eV]')
        plt.ylabel('rel. change in angle sum [%]')
        plt.title(f"varying Pz with input '{input}'")
        plt.legend()
        plt.show()

    elif switch == 5:
        #--------------------------------------------------------------------------------------------------------------------------
        print(f"Varying initial angle in x and y direction in range from 10 mrad to 0.01 mrad")
        difAngle = [1e-1, 5e-2, 3e-2,1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5] 
        
        for dif in difAngle:
            print(f"Running '{dif*1000}' mrad")
            changeMom(xmom,ymom,momZ,-1, -1)
            dataControl = runRef(D1,D2,D3,momZ, False)
            changeMom(dif*momZ, dif*momZ, momZ, -1, -1)
            dataTestP = runRef(D1,D2,D3, momZ, False)
    
            if dataControl == 1:
                print("Something is wrong with control data")
                return
            
            if dataTestP == 1:
                relChangeP = 1000
            else:
                relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
                plotRefXY1(D1,D2,D3,momZ,f"dif '{dif*1000}' mrad")
    
            
            graphDataPX.append(dif)
            graphDataPY.append(relChangeP)        
    
        plt.scatter(graphDataPX, graphDataPY, color='blue', label='Vary Px')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
        plt.xlabel('change [rad]')
        plt.ylabel('rel. change in angle sum [%]')
        plt.title(f"varying Px and Py with input '{input}'")
        plt.legend()
        plt.show()

    return


# In[81]:


def runAnaOffset(D1,D2, D3, momZ):
    #this function varies initial offset 
    
    topHatShapedQuads(True)
    update()
    
    input = [D1, D2, D3, momZ]

    difs = [0.005,0.003, 0.001, 0.0005,0.0003, 0.0001,0.00005 ,0.00003, 0.00001, 5.0E-6, 3.0E-6, 1.0E-6,5.0E-7, 3.0E-7, 1.0E-7, 5.0E-8, 3.0E-8, 1.0E-8,5.0E-9, 3.0E-9, 1.0E-9 ]
    #difs = [5.0E-7, 3.0E-7, 1.0E-7, 1E-8, 1E-9, 1E-10 ]

    graphDataMX = []
    graphDataMY = []
    graphDataPX = []
    graphDataPY = []
    
    #--------------------------------------------------------------------------------------------------------------------------
    print(f"Varying initial offset in x and y direction in range from 10 mrad to 0.01 mrad")
    
    for dif in difs:
        print(f"Running '{dif*1000}' mm")
        changeMom(1E+4,1E+4,momZ,0, 0)
        dataControl = runRef(D1,D2,D3,momZ, False)
        changeMom(xmom,ymom, momZ, dif, dif)
        dataTestP = runRef(D1,D2,D3, momZ, False)

        if dataControl == 1:
            print("Something is wrong with control data")
            return
        
        if dataTestP == 1:
            relChangeP = 1000
            relChangeM = 1000
        else:
            relChangeP = math.fabs(angleCalculation(dataControl)-angleCalculation(dataTestP))*100/angleCalculation(dataControl)
            relChangeM = math.fabs(angleCalculation(dataControl)-angleCalculationInverse(dataTestP))*100/angleCalculation(dataControl)
            plotRefXY2(D1,D2,D3,momZ,f"offset '{dif*1000}' mm")

        
        graphDataPX.append(dif*1e+3)
        graphDataPY.append(relChangeP)        
        graphDataMX.append(dif*1e+3)
        graphDataMY.append(relChangeM)   
    
    plt.scatter(graphDataPX, graphDataPY, color='blue', label='x offset in px != 0...')
    plt.scatter(graphDataMX, graphDataMY, color='red', label='y offset in px != 0...')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.yscale('log')
    plt.xlabel('change [mm]')
    plt.ylabel('rel. change in angle sum [%]')
    plt.title(f"varying Px and Py with input '{input}'")
    plt.legend()
    plt.show()



# In[82]:


#study of sensitivity w.r.t. varying to D1, D2, D3
#runAna(0.10, 0.176790, 0.185930, 7e+8)
#runAnaOffset(0.10, 0.176790, 0.185930, 7e+8)


# In[85]:


def findInfo(D1,D2,D3,D4, momZ):

    currentData = runRef(D1,D2,D3,D4,momZ,True)

    result = []
    result.append( setupSize(D1,D2,D3,D4) )
    accept = checkAngleAcceptance(D1,D2,D3,D4, momZ)
    if accept == 1:
        print(f"Cannot check angle acceptance for {D1, D2, D3, D4,momZ}.")
        return 1
    result.append(accept[0])
    result.append(accept[1])

    positions = changePositions(D1,D2,D3,D4)

    data = []
    data.append(list(currentData[1][0]))
    data.append(list(currentData[2][0]))
    closest = 0.1
    bestLine = []
    #look for closest in the x direction
    for j in range(len(currentData[1])):
        dist = math.fabs(currentData[1][j][0] - positions[3])
        if dist < closest:
            bestLine = list(currentData[1][j])
            closest = float(dist)
    data.append(list(bestLine))

    
    #look for closest in the y direction
    closest= 0.1
    for j in range(len(currentData[2])):
        dist = math.fabs(currentData[2][j][0] - positions[3])
        if dist < closest:
            bestLine = list(currentData[2][j])
            closest = float(dist)
    data.append(list(bestLine))

    
    #data: 0=z, 1=t, 2=Pz [MeV],5-x [mm], 6=y [mm], 7=px [eV], 8=py [eV] 

    #x_2/x'_1
    num = data[2][5]/ (data[0][7]*1e-3/data[0][2])
    result.append(num)
    #y_2/y'_1
    num = data[3][6]/ (data[1][8]*1e-3/data[1][2])
    result.append(num)

    #x_2/x_1
    num = data[2][5]/data[0][5]
    result.append(num)
    num = data[3][6]/data[1][6]
    result.append(num)

    #x'_2/x'_1
    num =  (data[2][7]*1e-3/data[2][2])/ (data[0][7]*1e-3/data[0][2])
    result.append(num)
    num =  (data[3][8]*1e-3/data[3][2])/ (data[1][8]*1e-3/data[1][2])
    result.append(num)

    return result
    


# In[81]:


def function(D, D1,D4, momZ, switch):

    #print(f"input D2, D3 = {D}")
    
    dataCurrent = runRef(D1, D[0], D[1], D4, momZ, False)
    if dataCurrent == 1:
        return 1E+9
    
    if switch == "parallel":    
        sum = parallelFocusing(dataCurrent)
    elif switch == "point":
        sum = pointFocusing(dataCurrent)
    elif switch == "lineX":
        sum = xLineFocusing(dataCurrent)
    elif switch == "lineY":
        sum = yLineFocusing(dataCurrent)

    #print(D, sum)
        
    return sum


# In[96]:


def runAstra(inputFile):
    #complex function to run analysis of a setup
    
    Dmin = [0.0,0.0]
    Dmax = [1.0,1.0]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]
    method = "Powell"
    tolerance = 1e-6
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]
    
    with open(inputFile, "r") as file:
        input = file.readlines()


    setups = []
    for i,line in enumerate(input):
        line = line.replace("\n","")
        line = line.split(" ")  
        num = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        setups.append(num)

    results = [[], [], []]
    #print(setups)
    for i in range(3): 
        topHatShapedQuads(i)
        for setup in setups:
            #parallel
            time1 = time.time()
            res1 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0],setup[3],  setup[4], "parallel") )
            time2 = time.time()
            if not res1.success:
                print(f"Did not obtain result for parallel focusing. The result turned out {res1.x}")
                continue 
            plotRefXY2(setup[0], *res1.x, setup[3],setup[4], f"Found solution for {fields[i]}, parallel focusing,\nPz= {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s:\nD2 = {math.ceil(res1.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res1.x[1]*1e+5)/1000} cm", f"parallel_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, 0 ,res1.x[0]*1e+2, res1.x[1]*1e+2] + findInfo(setup[0], *res1.x, setup[3], setup[4]))
            
            #point
            time1 = time.time()
            res2 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "point"))
            time2 = time.time()
            if not res2.success:
                print(f"Did not obtain result for point focusing. The result turned out {res2.x}")
                continue
            plotRefXY3(setup[0], *res2.x, setup[3],setup[4], f"Found solution for {fields[i]}, point-point focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s: D2 = {math.ceil(res2.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res2.x[1]*1e+5)/1000} cm", f"pointpoint_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res2.x[0]*1e+2, res2.x[1]*1e+2] + findInfo(setup[0], *res2.x, setup[3], setup[4]))
            
            #x line
            time1 = time.time()
            res3 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "lineX"))
            time2 = time.time()
            if not res3.success:
                print(f"Did not obtain result for line X focusing. The result turned out {res3.x}")
                continue
            plotRefXY3(setup[0], *res3.x, setup[3],setup[4], f"Found solution for {fields[i]}, line X focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s:\nD2 = {math.ceil(res3.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res3.x[1]*1e+5)/1000} cm", f"lineX_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res3.x[0]*1e+2, res3.x[1]*1e+2] + findInfo(setup[0], *res3.x, setup[3], setup[4]))

            #y line
            time1 = time.time()
            res4 = sc.optimize.minimize(function, (0.1,0.1), method=method, tol=tolerance, bounds=bounds,args=(setup[0], setup[3], setup[4], "lineY"))
            time2 = time.time()
            if not res4.success:
                print(f"Did not obtain result for line Y focusing. The result turned out {res4.x}")
                continue
            plotRefXY3(setup[0], *res4.x, setup[3],setup[4], f"Found solution for {fields[i]}, line Y focusing,\nPz = {setup[4]*1e-6} MeV, time= {math.ceil(10*(time2-time1))/10} s: D2 = {math.ceil(res4.x[0]*1e+5)/1000} cm, D3 = {math.ceil(res4.x[1]*1e+5)/1000} cm", f"lineY_Pz={setup[4]*1e-6}")
            results[i].append([setup[4]*1e-6, setup[0]*1e+2, setup[3]*1e+2 ,res4.x[0]*1e+2, res4.x[1]*1e+2] + findInfo(setup[0], *res4.x, setup[3], setup[4]))
            print(f"Timing with fields {fields[i]}: the entire process for setup: {setup} took {time2 - time1} s")
        
    df = pd.DataFrame(results[0])
    df.to_csv("topHatFields0.csv", index=False)
    
    df = pd.DataFrame(results[1])
    df.to_csv("AstraFringeFields0.csv", index=False)

    df = pd.DataFrame(results[2])
    df.to_csv("fieldProfiles0.csv", index=False)
    
    


# In[99]:


proc = subprocess.Popen(
    ['/bin/bash'], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True
)

proc.stdin.write("source /opt/intel/oneapi/setvars.sh\n")
proc.stdin.flush()

runAstra("../../MAXIMA/inputForTable.txt")
#print(findInfo(0.1,0.2,0.3,0.4, 5E+8))

proc.stdin.write("exit\n")
proc.stdin.flush()

#function([0.2,0.3], 0.1, 2E+8, "parallel")
#plotRefXY(0.04, 0.00716, 0.000302, None, 200000000.0)


# In[238]:


proc = subprocess.Popen(
    ['/bin/bash'], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True
)

proc.stdin.write("source /opt/intel/oneapi/setvars.sh\n")
proc.stdin.flush()

runAstra("../../MAXIMA/inputForTable.txt")

proc.stdin.write("exit\n")
proc.stdin.flush()

#function([0.2,0.3], 0.1, 2E+8, "parallel")
#plotRefXY(0.04, 0.00716, 0.000302, None, 200000000.0)


# In[60]:


df = pd.read_csv("topHatFields.csv")
df


# In[53]:


df = pd.read_csv("AstraFringeFields.csv")
df


# In[54]:


df = pd.read_csv("fieldProfiles.csv")
df


# # Beam analytics
# Here are functions that do not run only on 3 reference particles, but run the whole beam. The beam has it's energy/momentum spread whether it is in the magnitude of longitudinal momentum or in transverse direction.

# In[83]:


def updateBeam(x_off, sig_x, sig_px, y_off, sig_y, sig_py, sig_z, sig_pz , pz):
    #sig_xyz in mm, sig_pxy in eV, sig_pz in keV
    #if any value is equal to -1, it skips

    if x_off != -1:
        changeInputData("x_off", x_off)
    if sig_x != -1:
        changeInputData("sig_x", sig_x)
    if sig_px != -1:
        changeInputData("sig_px", sig_px)

    if y_off != -1:
        changeInputData("y_off", y_off)
    if sig_y != -1:
        changeInputData("sig_y", sig_y)
    if sig_py != -1:
        changeInputData("sig_py", sig_py)

    if sig_z != -1:
        changeInputData("sig_z", sig_z)
    if sig_pz != -1:
        changeInputData("sig_Ekin", sig_pz)
    if pz != -1:
        changeInputData("Ref_Ekin", pz*1E-6) #convert from eV to MeV
        
    subprocess.run("./generator " + fileName + " > output.txt", shell=True,check=True,executable='/bin/bash' )

    return    


# In[84]:


def runBeam(D1,D2,D3, momZ, px_sig, py_sig, moreData):


    changePositions(D1,D2,D3)

    #here can modify spreads in x, y, z directions
    updateBeam(-1, -1, px_sig, -1, -1, py_sig, -1, -1 ,momZ )
    
    
    #if moreData, then provide tracking for each of the reference particles and return it for plotting
    outputMoreData = []
    changeInputData("Distribution", fileName + ".ini" )
    
    res = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt", shell=True,check=True,executable='/bin/bash' )
    '''
    if res.returncode != 0:
        res = subprocess.run("./Astra " + fileName + " > output.txt", shell=True,check=True,executable='/bin/bash' )
    if res.returncode != 0:
        print(f"Astra returned with an error")
        return 1
    '''
    
    
    dataX = loadDataRef("Xemit")
    dataY = loadDataRef("Yemit")
    
    if moreData:
        return [dataX, dataY]
    else:
        return dataX[-1], dataY[-1]
        


# In[85]:


def divergence(dataX, dataY):

    data = loadDataRef(setupLengthStr)

    p = 0
    for line in data:
        p += (line[3]/line[5])**2 + (line[4]/line[5])**2
        
    
    return math.sqrt(p)         


# In[86]:


def funcBeam(D,D1, mom, sig_px, sig_py):

    data = runBeam( D1, D[0], D[1], mom, sig_px, sig_py, False)
    divSum = divergence(*data)

    return divSum


# In[87]:


def Beam():
# function which each setup runs only once and looks at the outcome of all 
    update()

    #boundaries for D2, D3    
    Dmin = [0.0,0.0]
    Dmax = [0.3,0.3]
    
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    Pz = []
    D1 = []
    for i in range(1,10):
        Pz.append(1.5E+8 + 5E+7*i)
        D1.append(0.01*i)
    
    results = ""
    resultsTable = {
        "" : ["D1 [m]", "D2 [m]", "D3 [m]", "Pz [eV]", "F_num [mrad]", "xAngle_sig [mrad]", "yAngle_sig [mrad]", "active_particles [%]"]
    }

    df = pd.DataFrame(resultsTable)
    i = 1
    for pz in Pz:
        for d1 in D1:
            topHatShapedQuads(True)
            res = sc.optimize.minimize(funcBeam, (0.15, 0.15),method="Powell", bounds=bounds,tol=1e-8, args=(d1, pz, sig_px, sig_py))
            if not res.success:
                results += str(d1) + " " + str(0) + " " + str(0) + " " + str(pz) + "\n"
                continue
            
            sumNum = divergence(*runBeam(d1, *res.x , pz, sig_px, sig_py,False ))
            plotBeam(d1,res.x[0], res.x[1], pz, sig_px, sig_py, f"Numerical solution, [{d1}, {res.x}, {pz}], top hat fields")
            
            results += str(d1) + " " + str(res.x[0]) + " " + str(res.x[1]) + " "  + str(pz) + "\n"
            row = [d1, *res.x, pz, sumNum, sig_px, sig_py, activeParticles()]
            
            df['setup ' + str(i)] = row
            
    i += 1


    with open("resFigs/results.txt","w") as file:
        file.write(results)
    
        
    
    return df
    


# In[88]:


#df = Beam()
#df.to_csv('resFigs/table.csv', index=False)


# In[89]:


def plotBeam(D1, D2, D3, momZ, px_sig, py_sig, title):
    #this function plots px over py of all particles in a beam

    dataX, dataY = runBeam(D1, D2, D3, momZ,px_sig, py_sig, True)

    x = []
    x_avr = []
    xz = []
    y = []
    y_avr = []
    yz = []
    for line in dataX:
        xz.append(line[0])
        x.append(line[3])
        x_avr.append(line[2])
        
    for line in dataY:
        yz.append(line[0])
        y.append(line[3])   
        y_avr.append(line[2])

    plt.plot(yz, y, label="y rms", color='red')
    #plt.plot(yz, y_avr,label='y avr [mm]', color='yellow')
    plt.plot(xz, x, label="x rms", color='blue')
    #plt.plot(xz, x_avr, label='x avr [mm]', color='green')
    
    plt.xlabel('z [m]')
    plt.ylabel('offset rms [mm]')
    plt.legend()
    plt.title(title)

    plt.show()
    
    return   


# In[90]:


def comparisonAnaBeam(setupFilePath):
    #this function takes in solutions from analytical and numerical calculations 
    
    update()
    #boundaries for D2, D3    

    
    
    with open(setupFilePath + "/resultsAna.txt", "r") as file:
        stringdata = file.readlines()
    
    analyticData = []
    for line in stringdata:
        line = line.replace("\n","")
        line = line.split(" ")  
        analyticData.append(line)

    with open(setupFilePath + "/resultsBeam.txt","r") as file:
        stringdata = file.readlines()

    beamData = []
    for line in stringdata:
        line = line.replace("\n","")
        line = line.split(" ")  
        beamData.append(line)        
    
    if len(beamData) != len(analyticData):
        print(f"Length of beamData({len(beamData)}) is not equal to analyticData({len(analyticData)})")
        return

    
    DeltaD2 = []
    DeltaD3 = []
    D1 = []
    resultsTable = {
        "" : ["D1 [m]", "D2 [m]", "D3 [m]", "Pz [eV]", "F_ana [mrad]", "F_num [mrad]", "Delta D2 [mm]" , "Delta D3 [mm]", "xAngle_sig [mrad]", "yAngle_sig [mrad]", "active_particles [%]"]
    }

    df = pd.DataFrame(resultsTable)

    for i in range(1,9):
        if i == 0:
            continue
        #first run the analytical solution and show how it looks in Astra
        topHatShapedQuads(True)
        sumAna = divergence(*runBeam(float(analyticData[i][0]),float(analyticData[i][1]), float(analyticData[i][2]), float(analyticData[i][3]), sig_px, sig_py, False))
        plotBeam(float(analyticData[i][0]),float(analyticData[i][1]), float(analyticData[i][2]), float(analyticData[i][3]), sig_px, sig_py, f"Analytic results, {analyticData[i]} top hat fields")

        #res = sc.optimize.minimize(funcBeam, (0.15, 0.15),method="Powell", bounds=bounds,tol=1e-5, args=(float(row[0]), float(row[3]), sig_px, sig_py))
        #res = sc.optimize.minimize(funcBeam, (0.15, 0.15), method="Powell", bounds=bounds, options={'ftol': 1e-6}, args=(float(row[0]), float(row[3]), sig_px, sig_py))

        sumNum = divergence(*runBeam(float(beamData[i][0]),math.ceil(float(beamData[i][1])*1E+5)*1E-5,math.ceil(float(beamData[i][2])*1E+5)*1E-5, float(beamData[i][3]), sig_px, sig_py, False))
        plotBeam(float(beamData[i][0]),math.ceil(float(beamData[i][1])*1E+5)*1E-5,math.ceil(float(beamData[i][2])*1E+5)*1E-5, float(beamData[i][3]), sig_px, sig_py, f"Numerical results, {beamData[i]} top hat fields")
        
        analyticData[i].append(sumAna)
        analyticData[i].append(sumNum)
        analyticData[i].append(math.fabs(float(analyticData[i][1]) - float(beamData[i][1]) )*1000 )
        analyticData[i].append(math.fabs(float(analyticData[i][2]) - float(beamData[i][2]))*1000 )
        analyticData[i].append(sig_px/float(beamData[i][3]))
        analyticData[i].append(sig_py/float(beamData[i][3]))
        analyticData[i].append(activeParticles())

        D1.append(analyticData[i][0]*100)
        DeltaD2.append(math.fabs(float(analyticData[i][1]) - float(beamData[i][1]))*1000 )
        DeltaD3.append(math.fabs(float(analyticData[i][2]) - float(beamData[i][2]))*1000 )
        
        df['setup ' + str(i)] = analyticData[i]
        

    plt.scatter(D1, DeltaD2, label='Delta D2', color='blue')
    plt.scatter(D1, DeltaD3, label='Delta D3', color='red')

    plt.xlabel('D1 [cm]')
    plt.ylabel('delta [mm]')
    plt.yscale('log')
    plt.title(f"Plot of differences between Astra and Maxima")
    plt.legend()
    plt.show()        
    
    return df

