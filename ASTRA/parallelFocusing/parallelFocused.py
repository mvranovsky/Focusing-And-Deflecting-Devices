#!/usr/bin/env python
# coding: utf-8

# In[288]:


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


# # Parallel focusing 
# This is python code written in jupyter notebook which implements different methods for point-point to parallel-parallel focusing. It uses software ASTRA, a program to simulate beam dynamics in accelerators. Programs in this notebook run in the same directory as are the ASTRA executables, mainly generator and Astra. 
# 
# The initial information are regarding the input file to Astra and information about reference particles. I used 5 different reference particles to determine the focusing properties of a setup- the first particle with 0 offset and 0 angle, moving along the z axis. This particle should not move in the transverse direction. Next 2 particles would be with initial offsets in the x and y directions respectively, but because this is point-point focusing, I am not using these. Last 2 have angles in the x and y direction respectively.
# 
# The magnets that are used are permanent quadrupole magnets with set gradients, lengths and bore diameters. These parameters can be changed, but for now they are set to values of 3 quadrupole magnets in LLR laboratory. The variables which will be changing are distances between them and the initial momentum. D1 is the distance from the source to the 1. quadrupole magnet. Realistically, D1 is only up to fringe fields which are magnetic fields outside the magnet's bores (reach 3*bore size in ASTRA). This option can be changed using TopHatShapedQuads() function. D2 and D3 are distances between first 2 and last 2 magnets in sequence. Last variable that can be changed is the initial longitudinal momentum of particles.
# 
# For running beam simulations, one can define it's initial parameters like spread of transverse momenta, spread of longitudinal energy, spread of offsets in the x and y directions as well as in the longitudinal direction. Also number of initial particles, space charge, secondary particle emission or other parameters can be changed in file parallelBeam.in.
# 

# In[289]:


fileName = "parallelBeam"
fillNumber = "001"
setupLength = 4 #m
setupLengthStr = "0400"
longitudalEnergy = "5.0E+8" #eV

#offsets and angles for reference particles
xoffset = "2.0E-4" #m
yoffset =  "2.0E-4" #m
xmom = "1.0E+6" #eV
ymom = "1.0E+6" #eV


#parameters of magnets 
lengthQ1 = 0.036  #m
lengthQ2 = 0.12  #m
lengthQ3 = 0.1  #m

#diameters of bores
boreQ1 = 0.007
boreQ2 = 0.018
boreQ3 = 0.030
topHatField = True

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


# In[290]:


dataD1 = []
dataD2 = []
dataD3 = []
dataMom = []
dataSum = []


# ## Function to change input settings
# Function changeInputData() is a function created to change input variables for ASTRA. The first argument is the name of the parameter that needs to be changed, the second is the value. After that is topHatShapedQuads() which changes settings between ideal or realistic quadrupoles. To change the momentum in the z direction, one can use changeMomZ(). Files test0.ini all the way to test4.ini are input data for 5 different runs. File test0.ini has 0. reference particle with 0 offset and 0 initial angle, 1 and 2 have offsets in the x and y directions respectively and the last 2 have predefined angles. All of them should be parallel in the end. Lastly, function update() should be run to double check that all parameters are set to the right values.

# In[291]:


def changeInputData(tag, newVar):
#universal function which changes input parameter in input file to Astra
    try:
        # Open the file for reading
        with open(fileName + ".in", 'r') as file:
            lines = file.readlines()

        # Prepare the replacement string
        replacement = " " + tag + "=" + str(newVar) + "\n"

        # Iterate over the lines to find and replace the target line
        for i, line in enumerate(lines):
            if tag in line:
                lines[i] = replacement
                break  # Assuming there's only one occurrence to replace

        # Write the modified lines back to the file
        with open(fileName + ".in", 'w') as file:
            file.writelines(lines)

        #print(f"Replaced '{tag}' with '{replacement.strip()}'.")

    except FileNotFoundError:
        print("The file " + fileName + ".ini was not found.")
    except Exception as e:
        print(f"An error occurred when trying to change '{tag}' to variable '{newVar}': {e}")


    
    return


# In[292]:


def topHatShapedQuads(ideal):
    #switcher between idealised quadrupoles without fringe fields or not

    global topHatField 
    topHatField = ideal

    if ideal:
        changeInputData("Q_bore(1)", "1E-9")
        changeInputData("Q_bore(2)", "1E-9")
        changeInputData("Q_bore(3)", "1E-9")
    else: 
        changeInputData("Q_bore(1)", str(boreQ1))
        changeInputData("Q_bore(2)", str(boreQ2))
        changeInputData("Q_bore(3)", str(boreQ3))

    
    return       


# In[293]:


def changeMom(xAngle, yAngle, pz, xoff, yoff): 
    nameOfFiles = ["test0.ini", "test1.ini", "test2.ini", "test3.ini", "test4.ini"]
    try:
        testData = ""

        #change longitudinal momentum for files test0.ini through test4.ini and test.ini
        for name in nameOfFiles:
            with open(name, "r") as file:
                line = file.readlines()[0].split()

            #offset update
            if name == "test3.ini" and yoff != -1:
                line[1] = str(yoff) 
            if name == "test4.ini" and xoff != -1:
                line[0] = str(xoff)


            
            #momentum update
            if name == "test1.ini" and xAngle != -1:
                line[3] = str(xAngle*pz*1e-3)
            if name == "test2.ini" and yAngle != -1:
                line[4] = str(yAngle*pz*1e-3)

            
            line[5] = str(pz)

            inputData = ""
            for num in line:
                inputData += num + " "
            testData += inputData + "\n"
            with open(name, "w") as file:
                file.write(inputData)

    
        with open("test.ini","w") as file:
            file.write(testData)
    
        #generate new data with new momentum
        changeInputData("Ref_Ekin", str(pz))
        
        #uncomment once beam is being used!!!
        #subprocess.run("./generator " + fileName + " > output.txt" , shell=True,check=True,executable='/bin/bash' )
        #print(f"Successfully changed momentum to files and ran a generation of particles saved to '{fileName}'.")

    except FileNotFoundError:
        print("One of the files when changing initial offsets and momenta was not found.")
    except Exception as e:
        print(f"An error occurred when trying to change longitudinal momentum: {e}")

    return
    


# In[294]:


def changePositions(D1,D2,D3):

    Q1pos = D1 + lengthQ1/2
    Q2pos = D1 + lengthQ1 + D2 + lengthQ2/2
    Q3pos = D1 + lengthQ1 + D2 + lengthQ2 + D3 + lengthQ3/2
    
    changeInputData("Q_pos(1)",str(Q1pos) )
    changeInputData("Q_pos(2)",str(Q2pos) )
    changeInputData("Q_pos(3)",str(Q3pos) )

    #changing the positions of apertures
    ap1 = str(Q1pos - lengthQ1/2) + " " + str(boreQ1*1E+3/2) + "\n" + str(Q1pos + lengthQ1/2) + " " + str(boreQ1*1E+3/2)
    with open("aperture1.dat", "w") as file:
        file.write(ap1)
    ap2 = str(Q2pos - lengthQ2/2) + " " + str(boreQ2*1E+3/2) + "\n" + str(Q2pos + lengthQ2/2) + " " + str(boreQ2*1E+3/2)
    with open("aperture2.dat", "w") as file:
        file.write(ap2)
    ap3 = str(Q3pos - lengthQ3/2) + " " + str(boreQ3*1E+3/2) + "\n" + str(Q3pos + lengthQ3/2) + " " + str(boreQ3*1E+3/2)
    with open("aperture3.dat", "w") as file:
        file.write(ap3)
    return


# In[295]:


def update():

    changeInputData("ZSTOP", str(setupLength) )

    #change properties of magnets
    changeInputData("Q_length(1)", str(lengthQ1))
    changeInputData("Q_length(2)", str(lengthQ2))
    changeInputData("Q_length(3)", str(lengthQ3))
    
    #change parameters of the beam
    changeInputData("IPart", str(nParticles))
    '''
    changeInputData("sig_z",str(sig_z))
    changeInputData("sig_Ekin", str(sig_Ekin))
    changeInputData("sig_x", str(sig_x))
    changeInputData("sig_y", str(sig_y))
    changeInputData("sig_px", str(sig_px))
    changeInputData("sig_py", str(sig_py))
    '''
    dataD1.clear()
    dataD2.clear()
    dataD3.clear()
    dataSum.clear()
    
    print(f"Succesfully updated all variables.")

    return


# ## Functions that make life easier
# Here below are some one-liners or almost one-liners that run several times and return some specific values used in bigger algorithms implemented for example in refParticles().

# In[296]:


def setupSize(D1, D2, D3):

    size = D1 + lengthQ1 + D2 + lengthQ2 + D3 + lengthQ3 
    return size


# In[297]:


def activeParticles():
    #function which returns the number of particles that stayed in the beam w.r.t. original number of particles in the beam
    data = loadDataRef(setupLengthStr)

    lost = 0
    for line in data:
        if line[2] == setupLength:
            continue
        zpos = math.fabs(line[2])
        if zpos > 0.05: #if particle more than 5 cm from the reference particle, it is considered lost
            lost += 1

    return (str(len(data) - lost)+"/" + str(len(data)))


# In[298]:


def isRef0Straight(px, py):
    #function which checks if 0. ref particle did not move
    if px == 0 and py == 0:
        return True
    else:
        return False


# In[299]:


def differLine(line):
    #splits a line and converts string to float
    lineSplitted = line.split()
    
    return [float(num) for num in lineSplitted]    


# In[300]:


def loadDataRef(arg):
#open and load data about reference particle
#z [m], t [ns], pz [MeV/c], dE/dz [MeV/c], Larmor Angle [rad], x off [mm], y off [mm], px [eV/c], py [eV/c]
    data = []
    #assuming setup length
    with open(fileName + "." + arg + "." + fillNumber,"r") as file:
        for line in file:
            newLine = differLine(line)
            data.append(newLine)

    return data
    


# In[301]:


def fill4DGraph(D1, D2,D3,mom,sum):
    #function that fills info about each run, these data can be later used for some plots
    dataD1.append(D1)
    dataD2.append(D2)
    dataD3.append(D3)
    dataMom.append(mom)
    dataSum.append(sum)
    
    return


# In[302]:


def angleCalculation(data):
    #calculate sum of transverse momenta in final state in mrad
    
    sum =  math.sqrt((data[1][3]/data[1][5])**2 + (data[2][4]/data[2][5])**2 )*1e+3
    return sum


# In[303]:


def angleCalculationX(data):
    # return angle in x direction in mrad
    sum =  data[1][3]*1e+3/data[1][5]
    return sum


# In[304]:


def angleCalculationY(data):
    #return angle in y direction in mrad 
    sum = data[2][4]*1e+3/data[2][5]
    
    return sum


# In[305]:


def checkAngleAcceptance(D1,D2,D3, momZ):
    #goes from the larger values, if a reference particle gets lost, it's last z position is not setup length
    
    #print(f"Varying initial angle in x and y direction in range from 10 mrad to 0.01 mrad")
    difAngle = [1e-1, 5e-2, 4e-2, 3e-2,2.5e-2, 2e-2,1.5e-2, 1e-2,9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3,3e-3,2e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5] 

    resultAng = []
    for dif in difAngle:
        #print(f"Running '{dif*1000}' mrad")
    
        changeMom(dif, -1, momZ, -1, -1)
        changeInputData("Distribution", "test1.ini" )
        result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )

        data = loadDataRef("ref")[-1]
        distFromEnd = math.fabs( data[0] - setupLength )
        if distFromEnd > 1e-3:
            continue
        else:
            resultAng.append(dif)
            resultAng.append(data[5])
            break

    
    for dif in difAngle:
        #print(f"Running '{dif*1000}' mrad")
    
        changeMom(-1, dif, momZ, -1, -1)
        changeInputData("Distribution", "test2.ini" )
        result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )
    
        data = loadDataRef("ref")[-1]
        distFromEnd = math.fabs( data[0] - setupLength )
        if distFromEnd > 1e-3:
            continue
        else:
            resultAng.append(dif)
            resultAng.append(data[6])
            break

    changeMom(sig_xAngle, sig_yAngle, momZ, -1, -1)
    return resultAng


# # Function RunRef()
# Function runRef() is the function that does most of the work. The arguments are the specific D1, D2, D3 and longitudinal momentum that is of interest. It is created for 3 reference particles: 0 angle, x angle, y angle. It changes the variables in the input file for Astra, runs the program for each reference particle separately, loads the output of the program. If argument moreData is set to True, it returns the entire trajectories of the particles, if it is false only information at the end of setup. 

# In[306]:


'''
def runRef(D1, D2, D3,momZ, moreData):
    #this function runs Astra with 5 different reference particles for specific D1,D2,D3

    changePositions(D1,D2,D3)
    changeMom(-1, -1, momZ, -1, -1)


    #if moreData, then provide tracking for each of the reference particles and return it for plotting
    if moreData:
        inputDataName = ["test0.ini", "test1.ini", "test2.ini", "test3.ini", "test4.ini"]
        outputMoreData = []
        for i in range(len(inputDataName)):
            changeInputData("Distribution", inputDataName[i] )
            result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )

            if result.returncode != 0:
                print(f"Astra returned an error '{subprocess.CalledProcessError.stderr}'. ")
                return 1
            
            currentData = loadDataRef("ref")
            
            #condition for 0. ref particle-> it cannot move
            if i == 0 and not isRef0Straight(currentData[-1][7], currentData[-1][8]):
                print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
                return 1
                
            #condition to check if the particle came all the way to the end
            distFromEnd = math.fabs(currentData[-1][0] - setupLength)
            if distFromEnd > 0.1:
                print(f"Reference particle '{i}' did not get to the end in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'. ")
                return 1
            
            outputMoreData.append(currentData)

        return outputMoreData
        
    else:
        #inputDataName = ["test0.ini", "test3.ini", "test4.ini", "test1.ini", "test2.ini"]
        #outputData = []
        changeInputData("Distribution", "test.ini" )
        result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )

        if result.returncode != 0:
            print(f"Astra returned an error '{subprocess.CalledProcessError.stderr}'. ")
            return 1

        currentData = loadDataRef(setupLengthStr)
     
        #condition for 0. ref particle-> it cannot move
        if not isRef0Straight(currentData[0][3], currentData[0][4]):
            print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
            return 1
        
        #condition to check if the particle came all the way to the end
        for line in currentData:
            distFromEnd = math.fabs(line[2])
            if distFromEnd > 0.1 and distFromEnd != 4:
                print(f"One of particles did not reach the end in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'. ")
                return 1

        return currentData
'''


# In[307]:


def runRef(D1, D2, D3,momZ, moreData):
    #this function runs Astra with 5 different reference particles for specific D1,D2,D3

    changePositions(D1,D2,D3)
    changeMom(-1, -1, momZ, -1, -1)


    #if moreData, then provide tracking for each of the reference particles and return it for plotting
    if moreData:
        inputDataName = ["test0.ini", "test1.ini", "test2.ini", "test3.ini", "test4.ini"]
        outputMoreData = []
        for i in range(len(inputDataName)):
            changeInputData("Distribution", inputDataName[i] )
            result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )

            if result.returncode != 0:
                print(f"Astra returned an error '{subprocess.CalledProcessError.stderr}'. ")
                return 1
            
            currentData = loadDataRef("ref")
            
            #condition for 0. ref particle-> it cannot move
            if i == 0 and not isRef0Straight(currentData[-1][7], currentData[-1][8]):
                print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
                return 1
                
            #condition to check if the particle came all the way to the end
            distFromEnd = math.fabs(currentData[-1][0] - setupLength)
            if distFromEnd > 0.1:
                print(f"Reference particle '{i}' did not get to the end in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'. ")
                return 1
            
            outputMoreData.append(currentData)

        return outputMoreData
        
    else:
        inputDataName = ["test0.ini", "test1.ini", "test2.ini"]
        outputMoreData = []
        for i in range(len(inputDataName)):
            changeInputData("Distribution", inputDataName[i] )
            result = subprocess.run("source /opt/intel/oneapi/setvars.sh > out.txt && ./Astra " + fileName + " > output.txt",capture_output=True, text=True, shell=True,check=True,executable='/bin/bash' )

            if result.returncode != 0:
                print(f"Astra returned an error '{subprocess.CalledProcessError.stderr}'. ")
                return 1
            
            currentData = loadDataRef("ref")
            
            #condition for 0. ref particle-> it cannot move
            if i == 0 and not isRef0Straight(currentData[-1][7], currentData[-1][8]):
                print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
                return 1
                
            #condition to check if the particle came all the way to the end
            distFromEnd = math.fabs(currentData[-1][0] - setupLength)
            if distFromEnd > 0.1:
                print(f"Reference particle {i} did not get to the end in setup with D1 = {D1}, D2 = {D2} and D3 = {D3}. ")
                return 1
            outputMoreData.append( [currentData[-1][5]*1e-3, currentData[-1][6]*1e-3, currentData[-1][0], currentData[-1][7], currentData[-1][8], currentData[-1][2]*1e+6] )

        
        return outputMoreData


# ## Plotting functions
# Several functions to plot output from reference particles.

# In[308]:


def separateDataXYZ(data):

    z0 = []
    x0 = []
    y0 = []
    for element in data:
        z0.append(element[0])
        x0.append(element[5])
        y0.append(element[6])
        
    XYZ = []
    XYZ.append(x0)
    XYZ.append(y0)
    XYZ.append(z0)
    
    return XYZ


# In[309]:


def plotRefXY(D1, D2, D3, mom):

    #print(f"Running best setup again to get full data.")
    dataBest = runRef(D1, D2, D3, mom, True)

    data0 = separateDataXYZ(dataBest[0])
    #data1 = separateDataXYZ(dataBest[1])
    #data2 = separateDataXYZ(dataBest[2])
    data3 = separateDataXYZ(dataBest[1])
    data4 = separateDataXYZ(dataBest[2])


    plt.plot(data0[2], data0[0], label='0 offset, 0 angle', color='blue')
    #plt.plot(data1[2], data1[0], label='x offset, 0 angle', color='green')
    #plt.plot(data2[2], data2[0], label='y offset, 0 angle', color='red')
    plt.plot(data3[2], data3[0], label='0 offset, x angle', color='yellow')
    plt.plot(data4[2], data4[0], label='0 offset, y angle', color='purple')

    plt.legend()

    plt.xlabel("z [m]")
    plt.ylabel("x_offset [mm]")
    plt.title(f"x offset along z('{D1}', '{D2}', '{D3}')")

    plt.show()


    
    plt.plot(data0[2], data0[1], label='0 offset, 0 angle', color='blue')
    #plt.plot(data1[2], data1[1], label='x offset, 0 angle', color='green')
    #plt.plot(data2[2], data2[1], label='y offset, 0 angle', color='red')
    plt.plot(data3[2], data3[1], label='0 offset, x angle', color='yellow')
    plt.plot(data4[2], data4[1], label='0 offset, y angle', color='purple')

    plt.legend()

    plt.xlabel("z [m]")
    plt.ylabel("y_offset [mm]")
    plt.title(f"y offset along z('{D1}', '{D2}', '{D3}')")

    plt.show()
    
    return


# In[310]:


def plotRefXY1(D1, D2, D3, mom,title):

    #print(f"Running best setup again to get full data.")
    dataBest = runRef(D1, D2, D3, mom, True)

    data0 = separateDataXYZ(dataBest[0])
    data3 = separateDataXYZ(dataBest[1])
    data4 = separateDataXYZ(dataBest[2])


    plt.plot(data0[2], data0[0], label='0 offset, initial 0 angle', color='blue')
    plt.plot(data3[2], data3[0], label='x offset, initial x angle', color='red')
    plt.plot(data3[2], data3[1], label='y offset, initial x angle', color='yellow')
    plt.plot(data4[2], data4[0], label='x offset, initial y angle', color='green')
    plt.plot(data4[2], data4[1], label='y offset, initial y angle', color='purple')

    plt.legend()

    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")
    plt.title(title)

    plt.show()
    
    return


# In[311]:


def plotRefXY2(D1, D2, D3, mom,title):

    #print(f"Running best setup again to get full data.")
    dataBest = runRef(D1, D2, D3, mom, True)

    data0 = separateDataXYZ(dataBest[0])
    data3 = separateDataXYZ(dataBest[1])
    data4 = separateDataXYZ(dataBest[2])
    data1 = separateDataXYZ(dataBest[3])
    data2 = separateDataXYZ(dataBest[4])

    plt.plot(data0[2], data0[0], label='x offset: 0 offset, initial 0 angle', color='blue')
    plt.plot(data3[2], data3[0], label='x offset: initial x angle, x offset', color='red')
    plt.plot(data3[2], data3[1], label='y offset: initial x angle, x offset', color='yellow')
    plt.plot(data4[2], data4[0], label='x offset: initial y angle, y offset', color='green')
    plt.plot(data4[2], data4[1], label='y offset: initial y angle, y offset', color='purple')

    plt.legend()

    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")
    plt.title(title)

    plt.show()

    plt.plot(data0[2], data0[1], label='y offset: initial 0 offset ,0 angle', color='blue')
    plt.plot(data1[2], data1[0], label='x offset: initial x angle, y offset', color='red')
    plt.plot(data1[2], data1[1], label='y offset: initial x angle, y offset', color='yellow')
    plt.plot(data2[2], data2[0], label='x offset: initial y angle, x offset', color='green')
    plt.plot(data2[2], data2[1], label='y offset: initial y angle, x offset', color='purple')

    plt.legend()

    plt.xlabel("z [m]")
    plt.ylabel("offset [mm]")
    plt.title(title + "inverse offsets")

    plt.show()
    return


# In[312]:


def plotSumData(XorY):

    dataSum = []
    if XorY == "X":
        dataSum = dataSumX
    else:
        dataSum = dataSumY
    
    
    plt.plot(dataSum, dataD1, label="D1", color='red')
    plt.plot(dataSum, dataD2, label="D2", color='blue')
    plt.plot(dataSum, dataD3, label="D3", color='green')

    plt.legend()
    plt.xlabel(" [m]")
    plt.ylabel("sum [mrad]")
    plt.title(f"Inverted function of D1, D2, D3 depending on dataSum'{XorY}' ")
    plt.show()

    return
    


# This is the code where one can run the equidistant intervals algorithm, I put it into a function so it does not do anything when I am running the entire notebook. One sets manually the ranges for each parameter/variable and the number of intervals. If the lower and upper limits are equal, then the variable is constant. Remember, the number of iterations is nInt to the number of non-constant variables times 3.

# In[313]:


def func(D, D1, mom):
    
    dataCurrent = runRef(D1, D[0], D[1], mom, False)
    sum = angleCalculation(dataCurrent)
    return sum


# In[314]:


def func3(D, mom):
    
    dataCurrent = runRef(D[0], D[1], D[2], mom, False)
    sumX = angleCalculation(dataCurrent)

    return sumX


# In[315]:


def ResultsTable(results, methodNames):

    d1 = []
    d2 = []
    funkMin = []
    nEval = []
    message = []
    success = []

    for res in results:
        d1.append(res.x[0])
        d2.append(res.x[1])
        funkMin.append(res.fun)
        nEval.append(res.nfev)
        message.append(res.message)
        success.append(res.success)

    resultTable = {
        "method name:" : methodNames,
        "D1" : d1,
        "D2" : d2,
        "Minimum of function" : funkMin,
        "Number of evaluations" : nEval,
        "Message" : message,
        "success" : success,   
    }

    df = pd.DataFrame(resultTable)
    

    return df


# ## ComparisonAnaNum()
# For this comparison, the fringe fields stay off. The first run is with analytical solution, the second run is with found numerical solution. The 2 results can be compared in a table below. The differences in D2,D3 are in Delta D2 and Delta D3. Parameters of runs are also there.

# In[316]:


def comparisonAnaNum(setupFileName):

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

    
    resultsTable = {
        "" : ["D1 [m]", "D2 [m]", "D3 [m]", "Pz [eV]", 
              "F_ana [keV**2]", "F_num [keV**2]", 
              "Delta D2 [mm]" , "Delta D3 [mm]",
              "setup length [m]", 
              "Angle accept. x [mrad]", "Angle accept. y [mrad]",
              "beam size x [mm]", "beam size y [mm]"            
             ]
    }

    df = pd.DataFrame(resultsTable)
    i = 1
    for row in analyticData:
        #first run the analytical solution and show plots
        print(f"Running {row}")
        topHatShapedQuads(True)
        changeMom(sig_xAngle, sig_yAngle, float(row[3]), -1, -1)
        
        sumAna = angleCalculation(runRef(float(row[0]),float(row[1]), float(row[2]), float(row[3]), False))
        plotRefXY1(float(row[0]),float(row[1]), float(row[2]), float(row[3]), f"Analytic results, '{row}', top hat fields")
        
        res = sc.optimize.minimize(func, (0.15, 0.15),method="Powell", bounds=bounds,tol=1e-8, args=(float(row[0]), float(row[3])))
        sumNum = angleCalculation(runRef(float(row[0]),*res.x , float(row[3]), False))
        plotRefXY1(float(row[0]),*res.x, float(row[3]), f"Numerical results, ['{row[0]}', '{res.x[0]}', '{res.x[1]}', '{row[3]}'], top hat fields")
        
        row.append(sumAna)
        row.append(sumNum)
        
        DeltaD2 = math.fabs(float(row[1]) - res.x[0])*1000 
        DeltaD3 = math.fabs(float(row[2]) - res.x[1])*1000
        row.append( DeltaD2 )
        row.append( DeltaD3 )
        
        row.append( setupSize(float(row[0]), *res.x ))
        angleAcceptance = checkAngleAcceptance(float(row[0]), *res.x, float(row[3]))
        row.append(angleAcceptance[0])
        row.append(angleAcceptance[2])
        row.append(angleAcceptance[1])
        row.append(angleAcceptance[3])
        
        df['setup ' + str(i)] = row
        
        i += 1
        
    
    return df


# In[ ]:


df = comparisonAnaNum("analyticalResultsP.txt")


# In[274]:


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
    


# In[275]:


#df = getResults("results.txt")
#df.to_csv('resFigs/table.csv', index=False)


# ## Functions to study sensitivity
# The following functions are implemented with a goal to study how sensitive or stable a solution is when some parameters or variables are being alternated. runAna() studies variability in D1, D2, D3, Pz and initial Px, Py. The input of the function is a solution- a functioning setup. For each variable function prints a graph with logarithmic x axis representing change in the variable, the logarithmic y axis returns relative change in the function (angleCalculation() ). 
# Below that is another function which studies the initial x and y offset. 

# In[276]:


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


# In[277]:


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



# In[278]:


#study of sensitivity w.r.t. varying to D1, D2, D3
#runAna(0.10, 0.176790, 0.185930, 7e+8)
#runAnaOffset(0.10, 0.176790, 0.185930, 7e+8)


# # Beam analytics
# Here are functions that do not run only on 3 reference particles, but run the whole beam. The beam has it's energy/momentum spread whether it is in the magnitude of longitudinal momentum or in transverse direction.

# In[279]:


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


# In[280]:


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
        


# In[281]:


def divergence(dataX, dataY):

    data = loadDataRef(setupLengthStr)

    p = 0
    for line in data:
        p += (line[3]/line[5])**2 + (line[4]/line[5])**2
        
    
    return math.sqrt(p)         


# In[282]:


def funcBeam(D,D1, mom, sig_px, sig_py):

    data = runBeam( D1, D[0], D[1], mom, sig_px, sig_py, False)
    divSum = divergence(*data)

    return divSum


# In[283]:


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
    


# In[284]:


#df = Beam()
#df.to_csv('resFigs/table.csv', index=False)


# In[285]:


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


# In[286]:


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


# In[287]:


#df = comparisonAnaBeam("results/0-1")
#df
#data = runBeam(0.1,0.17679, 0.18593, 7E+8, False)
#print(divergence(data))


