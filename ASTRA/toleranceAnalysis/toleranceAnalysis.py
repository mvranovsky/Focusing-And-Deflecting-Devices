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
import sys
import ROOT
import array
import random


# Define all variables with explicit types
# Setup info


d1 = array.array('f', [0.0]) 
d2 = array.array('f', [0.0])
d3 = array.array('f', [0.0])
d4 = array.array('f', [0.0])
L = array.array('f', [0.0])
pz = array.array('f', [0.0])
accX = array.array('f', [0.0])
accY = array.array('f', [0.0])

# Initial beam info
sigX = array.array('f', [0.0])
sigPx = array.array('f', [0.0])
iniAvrX = array.array('f', [0.0])
iniRMSX = array.array('f', [0.0])
iniRMSXPrime = array.array('f', [0.0])
iniEmitNormX = array.array('f', [0.0])
sigY = array.array('f', [0.0])
sigPy = array.array('f', [0.0])
iniAvrY = array.array('f', [0.0])
iniRMSY = array.array('f', [0.0])
iniRMSYPrime = array.array('f', [0.0])
iniEmitNormY = array.array('f', [0.0])

# Output beam info
outAvrX = array.array('f', [0.0])
outRMSX = array.array('f', [0.0])
outRMSXPrime = array.array('f', [0.0])
outEmitNormX = array.array('f', [0.0])
outAvrY = array.array('f', [0.0])
outRMSY = array.array('f', [0.0])
outRMSYPrime = array.array('f', [0.0])
outEmitNormY = array.array('f', [0.0])

# Quadrupole wobbles info
offsetInMicrons = array.array('f', [0.0])

# Quadrupole 1
Q1MCXWobbles = array.array('i', [0])
Q1MCXAmp1 = array.array('f', [0.0])
Q1MCXFreq1 = array.array('f', [0.0])
Q1MCXIniPhase1 = array.array('f', [0.0])
Q1MCXAmp2 = array.array('f', [0.0])
Q1MCXFreq2 = array.array('f', [0.0])
Q1MCXIniPhase2 = array.array('f', [0.0])

Q1MCYWobbles = array.array('i', [0])
Q1MCYAmp1 = array.array('f', [0.0])
Q1MCYFreq1 = array.array('f', [0.0])
Q1MCYIniPhase1 = array.array('f', [0.0])
Q1MCYAmp2 = array.array('f', [0.0])
Q1MCYFreq2 = array.array('f', [0.0])
Q1MCYIniPhase2 = array.array('f', [0.0])

Q1SkewAngleWobbles = array.array('i', [0])
Q1SkewAngleAmp1 = array.array('f', [0.0])
Q1SkewAngleFreq1 = array.array('f', [0.0])
Q1SkewAngleIniPhase1 = array.array('f', [0.0])
Q1SkewAngleAmp2 = array.array('f', [0.0])
Q1SkewAngleFreq2 = array.array('f', [0.0])
Q1SkewAngleIniPhase2 = array.array('f', [0.0])

Q1GradWobbles = array.array('i', [0])
Q1GradAmp = array.array('f', [0.0])
Q1GradFreq = array.array('f', [0.0])
Q1GradIniPhase = array.array('f', [0.0])

# offsets that go in Astra input file
Q1offsetX = array.array('f',[0.0])
Q1offsetY = array.array('f',[0.0])
Q1rotX = array.array('f',[0.0])
Q1rotY = array.array('f',[0.0])
Q1rotZ = array.array('f',[0.0])


# Quadrupole 2
Q2MCXWobbles = array.array('i', [0])
Q2MCXAmp1 = array.array('f', [0.0])
Q2MCXFreq1 = array.array('f', [0.0])
Q2MCXIniPhase1 = array.array('f', [0.0])
Q2MCXAmp2 = array.array('f', [0.0])
Q2MCXFreq2 = array.array('f', [0.0])
Q2MCXIniPhase2 = array.array('f', [0.0])

Q2MCYWobbles = array.array('i', [0])
Q2MCYAmp1 = array.array('f', [0.0])
Q2MCYFreq1 = array.array('f', [0.0])
Q2MCYIniPhase1 = array.array('f', [0.0])
Q2MCYAmp2 = array.array('f', [0.0])
Q2MCYFreq2 = array.array('f', [0.0])
Q2MCYIniPhase2 = array.array('f', [0.0])

Q2SkewAngleWobbles = array.array('i', [0])
Q2SkewAngleAmp1 = array.array('f', [0.0])
Q2SkewAngleFreq1 = array.array('f', [0.0])
Q2SkewAngleIniPhase1 = array.array('f', [0.0])
Q2SkewAngleAmp2 = array.array('f', [0.0])
Q2SkewAngleFreq2 = array.array('f', [0.0])
Q2SkewAngleIniPhase2 = array.array('f', [0.0])

Q2GradWobbles = array.array('i', [0])
Q2GradAmp = array.array('f', [0.0])
Q2GradFreq = array.array('f', [0.0])
Q2GradIniPhase = array.array('f', [0.0])

# offsets that go in Astra input file
Q2offsetX = array.array('f',[0.0])
Q2offsetY = array.array('f',[0.0])
Q2rotX = array.array('f',[0.0])
Q2rotY = array.array('f',[0.0])
Q2rotZ = array.array('f',[0.0])


# Quadrupole 3
Q3MCXWobbles = array.array('i', [0])
Q3MCXAmp1 = array.array('f', [0.0])
Q3MCXFreq1 = array.array('f', [0.0])
Q3MCXIniPhase1 = array.array('f', [0.0])
Q3MCXAmp2 = array.array('f', [0.0])
Q3MCXFreq2 = array.array('f', [0.0])
Q3MCXIniPhase2 = array.array('f', [0.0])

Q3MCYWobbles = array.array('i', [0])
Q3MCYAmp1 = array.array('f', [0.0])
Q3MCYFreq1 = array.array('f', [0.0])
Q3MCYIniPhase1 = array.array('f', [0.0])
Q3MCYAmp2 = array.array('f', [0.0])
Q3MCYFreq2 = array.array('f', [0.0])
Q3MCYIniPhase2 = array.array('f', [0.0])

Q3SkewAngleWobbles = array.array('i', [0])
Q3SkewAngleAmp1 = array.array('f', [0.0])
Q3SkewAngleFreq1 = array.array('f', [0.0])
Q3SkewAngleIniPhase1 = array.array('f', [0.0])
Q3SkewAngleAmp2 = array.array('f', [0.0])
Q3SkewAngleFreq2 = array.array('f', [0.0])
Q3SkewAngleIniPhase2 = array.array('f', [0.0])

Q3GradWobbles = array.array('i', [0])
Q3GradAmp = array.array('f', [0.0])
Q3GradFreq = array.array('f', [0.0])
Q3GradIniPhase = array.array('f', [0.0])


# offsets that go in Astra input file
Q3offsetX = array.array('f',[0.0])
Q3offsetY = array.array('f',[0.0])
Q3rotX = array.array('f',[0.0])
Q3rotY = array.array('f',[0.0])
Q3rotZ = array.array('f',[0.0])


output_file = ROOT.TFile("output.root", "RECREATE")
tree = ROOT.TTree("outputTree", "output TTree")

# setup info
tree.Branch("D1", d1, "D1/F")  #m
tree.Branch("D2", d2, "D2/F")  #m
tree.Branch("D3", d3, "D3/F")  #m
tree.Branch("D4", d4, "D4/F")  #m
tree.Branch("setupLength", L, "L/F")  #m
tree.Branch("Pz", pz, "Pz/F")  #MeV
tree.Branch("acceptanceX",accX , "acceptanceXInMrad/F")  #mrad
tree.Branch("acceptanceY",accY , "acceptanceYInMrad/F")  #mrad

# initial beam info
tree.Branch("sig_x", sigX, "sigX/F")   #mu m
tree.Branch("sig_px", sigPx, "sigPx/F")  #eV
tree.Branch("iniAvrX", iniAvrX, "iniAvrX/F")  #mm
tree.Branch("iniRMSX", iniRMSX, "iniRMSX/F")  #mm
tree.Branch("iniRMSXPrime", iniRMSXPrime, "iniRMSXPrime/F")  #mrad
tree.Branch("iniEmitNormX", iniEmitNormX, "iniEmitNormX/F")  #pi mm mrad
tree.Branch("sig_y", sigY, "sigY/F")
tree.Branch("sig_py", sigPy, "sigPy/F")
tree.Branch("iniAvrY", iniAvrY, "iniAvrY/F")
tree.Branch("iniRMSY", iniRMSY, "iniRMSY/F")
tree.Branch("iniRMSYPrime", iniRMSYPrime, "iniRMSYPrime/F")
tree.Branch("iniEmitNormY", iniEmitNormY, "iniEmitNormY/F")


# output beam info
tree.Branch("outAvrX", outAvrX, "outAvrX/F")  
tree.Branch("outRMSX", outRMSX, "outRMSX/F")
tree.Branch("outRMSXPrime", outRMSXPrime, "outRMSXPrime/F")
tree.Branch("outEmitNormX", outEmitNormX, "outEmitNormX/F")
tree.Branch("outAvrY", outAvrY, "outAvrY/F")
tree.Branch("outRMSY", outRMSY, "outRMSY/F")
tree.Branch("outRMSYPrime", outRMSYPrime, "outRMSYPrime/F")
tree.Branch("outEmitNormY", outEmitNormY, "outEmitNormY/F")


# quadrupole wobbles info
tree.Branch("offsetInMicrons", offsetInMicrons, "offsetInMicrons/F")  #mu m

# quadrupole 1
tree.Branch("Q1MCXWobbles", Q1MCXWobbles, "Q1MCXWobbles/I")  #0 == False, 1 == True
tree.Branch("Q1MCXAmp1", Q1MCXAmp1, "Q1MCXAmp1/F")           #mu m
tree.Branch("Q1MCXFreq1",Q1MCXFreq1 , "Q1MCXFreq1/F")        #1/m
tree.Branch("Q1MCXIniPhase1",Q1MCXIniPhase1 , "Q1MCXIniPhase1/F")  #rad
tree.Branch("Q1MCXAmp2", Q1MCXAmp2, "Q1MCXAmp2/F")
tree.Branch("Q1MCXFreq2",Q1MCXFreq2 , "Q1MCXFreq2/F")
tree.Branch("Q1MCXIniPhase2",Q1MCXIniPhase2 , "Q1MCXIniPhase2/F")

tree.Branch("Q1MCYWobbles", Q1MCYWobbles, "Q1MCYWobbles/I")
tree.Branch("Q1MCYAmp1", Q1MCYAmp1, "Q1MCYAmp1/F")
tree.Branch("Q1MCYFreq1",Q1MCYFreq1 , "Q1MCYFreq1/F")
tree.Branch("Q1MCYIniPhase1",Q1MCYIniPhase1 , "Q1MCYIniPhase1/F")
tree.Branch("Q1MCYAmp2", Q1MCYAmp2, "Q1MCYAmp2/F")
tree.Branch("Q1MCYFreq2",Q1MCYFreq2 , "Q1MCYFreq2/F")
tree.Branch("Q1MCYIniPhase2",Q1MCYIniPhase2 , "Q1MCYIniPhase2/F")

tree.Branch("Q1SkewAngleWobbles", Q1SkewAngleWobbles, "Q1SkewAngleWobbles/I")
tree.Branch("Q1SkewAngleAmp1", Q1SkewAngleAmp1, "Q1SkewAngleAmp1/F")
tree.Branch("Q1SkewAngleFreq1", Q1SkewAngleFreq1, "Q1SkewAngleFreq1/F")
tree.Branch("Q1SkewAngleIniPhase1",Q1MCXIniPhase1 , "Q1MCXIniPhase1/F")
tree.Branch("Q1SkewAngleAmp2", Q1SkewAngleAmp1, "Q1SkewAngleAmp2/F")
tree.Branch("Q1SkewAngleFreq2", Q1SkewAngleFreq1, "Q1SkewAngleFreq2/F")
tree.Branch("Q1SkewAngleIniPhase2",Q1MCXIniPhase1 , "Q1MCXIniPhase2/F")

tree.Branch("Q1GradWobbles", Q1GradWobbles, "Q1GradWobbles/I")
tree.Branch("Q1GradAmp",Q1GradAmp , "Q1GradAmp/F")
tree.Branch("Q1GradFreq", Q1GradFreq, "Q1GradFreq/F")
tree.Branch("Q1GradIniPhase", Q1GradIniPhase, "Q1GradIniPhase/F")


tree.Branch("Q1offsetX", Q1offsetX, "Q1offsetX/F")
tree.Branch("Q1offsetY", Q1offsetY, "Q1offsetY/F")
tree.Branch("Q1rotX", Q1rotX, "Q1rotX/F")
tree.Branch("Q1rotY", Q1rotY, "Q1rotY/F")
tree.Branch("Q1rotZ", Q1rotZ, "Q1rotZ/F")


# quadrupole 2
tree.Branch("Q2MCXWobbles", Q2MCXWobbles, "Q2MCXWobbles/I")
tree.Branch("Q2MCXAmp1", Q2MCXAmp1, "Q2MCXAmp1/F")
tree.Branch("Q2MCXFreQ2",Q2MCXFreq1 , "Q2MCXFreq1/F")
tree.Branch("Q2MCXIniPhase1",Q2MCXIniPhase1 , "Q2MCXIniPhase1/F")
tree.Branch("Q2MCXAmp2", Q2MCXAmp2, "Q2MCXAmp2/F")
tree.Branch("Q2MCXFreq2",Q2MCXFreq2 , "Q2MCXFreq2/F")
tree.Branch("Q2MCXIniPhase2",Q2MCXIniPhase2 , "Q2MCXIniPhase2/F")

tree.Branch("Q2MCYWobbles", Q2MCYWobbles, "Q2MCYWobbles/I")
tree.Branch("Q2MCYAmp1", Q2MCYAmp1, "Q2MCYAmp1/F")
tree.Branch("Q2MCYFreq1",Q2MCYFreq1 , "Q2MCYFreq1/F")
tree.Branch("Q2MCYIniPhase1",Q2MCYIniPhase1 , "Q2MCYIniPhase1/F")
tree.Branch("Q2MCYAmp2", Q2MCYAmp2, "Q2MCYAmp2/F")
tree.Branch("Q2MCYFreq2",Q2MCYFreq2 , "Q2MCYFreq2/F")
tree.Branch("Q2MCYIniPhase2",Q2MCYIniPhase2 , "Q2MCYIniPhase2/F")

tree.Branch("Q2SkewAngleWobbles", Q2SkewAngleWobbles, "Q2SkewAngleWobbles/I")
tree.Branch("Q2SkewAngleAmp1", Q2SkewAngleAmp1, "Q2SkewAngleAmp1/F")
tree.Branch("Q2SkewAngleFreq1", Q2SkewAngleFreq1, "Q2SkewAngleFreq1/F")
tree.Branch("Q2SkewAngleIniPhase1",Q2MCXIniPhase1 , "Q2MCXIniPhase1/F")
tree.Branch("Q2SkewAngleAmp2", Q2SkewAngleAmp1, "Q2SkewAngleAmp2/F")
tree.Branch("Q2SkewAngleFreq2", Q2SkewAngleFreq1, "Q2SkewAngleFreq2/F")
tree.Branch("Q2SkewAngleIniPhase2",Q2MCXIniPhase1 , "Q2MCXIniPhase2/F")

tree.Branch("Q2GradWobbles", Q2GradWobbles, "Q2GradWobbles/I")
tree.Branch("Q2GradAmp",Q2GradAmp , "Q2GradAmp/F")
tree.Branch("Q2GradFreq", Q2GradFreq, "Q2GradFreq/F")
tree.Branch("Q2GradIniPhase", Q2GradIniPhase, "Q2GradIniPhase/F")

tree.Branch("Q2offsetX", Q2offsetX, "Q2offsetX/F")
tree.Branch("Q2offsetY", Q2offsetY, "Q2offsetY/F")
tree.Branch("Q2rotX", Q2rotX, "Q2rotX/F")
tree.Branch("Q2rotY", Q2rotY, "Q2rotY/F")
tree.Branch("Q2rotZ", Q2rotZ, "Q2rotZ/F")


# quadrupole 3
tree.Branch("Q3MCXWobbles", Q3MCXWobbles, "Q3MCXWobbles/I")
tree.Branch("Q3MCXAmp1", Q3MCXAmp1, "Q3MCXAmp1/F")
tree.Branch("Q3MCXFreQ3",Q3MCXFreq1 , "Q3MCXFreq1/F")
tree.Branch("Q3MCXIniPhase1",Q3MCXIniPhase1 , "Q3MCXIniPhase1/F")
tree.Branch("Q3MCXAmp2", Q3MCXAmp2, "Q3MCXAmp2/F")
tree.Branch("Q3MCXFreq2",Q3MCXFreq2 , "Q3MCXFreq2/F")
tree.Branch("Q3MCXIniPhase2",Q3MCXIniPhase2 , "Q3MCXIniPhase2/F")

tree.Branch("Q3MCYWobbles", Q3MCYWobbles, "Q3MCYWobbles/I")
tree.Branch("Q3MCYAmp1", Q3MCYAmp1, "Q3MCYAmp1/F")
tree.Branch("Q3MCYFreq1",Q3MCYFreq1 , "Q3MCYFreq1/F")
tree.Branch("Q3MCYIniPhase1",Q3MCYIniPhase1 , "Q3MCYIniPhase1/F")
tree.Branch("Q3MCYAmp2", Q3MCYAmp2, "Q3MCYAmp2/F")
tree.Branch("Q3MCYFreq2",Q3MCYFreq2 , "Q3MCYFreq2/F")
tree.Branch("Q3MCYIniPhase2",Q3MCYIniPhase2 , "Q3MCYIniPhase2/F")

tree.Branch("Q3SkewAngleWobbles", Q3SkewAngleWobbles, "Q3SkewAngleWobbles/I")
tree.Branch("Q3SkewAngleAmp1", Q3SkewAngleAmp1, "Q3SkewAngleAmp1/F")
tree.Branch("Q3SkewAngleFreq1", Q3SkewAngleFreq1, "Q3SkewAngleFreq1/F")
tree.Branch("Q3SkewAngleIniPhase1",Q3MCXIniPhase1 , "Q3MCXIniPhase1/F")
tree.Branch("Q3SkewAngleAmp2", Q3SkewAngleAmp1, "Q3SkewAngleAmp2/F")
tree.Branch("Q3SkewAngleFreq2", Q3SkewAngleFreq1, "Q3SkewAngleFreq2/F")
tree.Branch("Q3SkewAngleIniPhase2",Q3MCXIniPhase1 , "Q3MCXIniPhase2/F")

tree.Branch("Q3GradWobbles", Q3GradWobbles, "Q3GradWobbles/I")
tree.Branch("Q3GradAmp",Q3GradAmp , "Q3GradAmp/F")
tree.Branch("Q3GradFreq", Q3GradFreq, "Q3GradFreq/F")
tree.Branch("Q3GradIniPhase", Q3GradIniPhase, "Q3GradIniPhase/F")

tree.Branch("Q3offsetX", Q3offsetX, "Q3offsetX/F")
tree.Branch("Q3offsetY", Q3offsetY, "Q3offsetY/F")
tree.Branch("Q3rotX", Q3rotX, "Q3rotX/F")
tree.Branch("Q3rotY", Q3rotY, "Q3rotY/F")
tree.Branch("Q3rotZ", Q3rotZ, "Q3rotZ/F")


def func(D,D1, Pz, focusing):

    data = []
    Sum = 1E+9
    try:
        data = astra.runRef(D1, *D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return Sum
    else:
        if focusing == "parallel":
            Sum = astra.parallelFocusing(data)
        else:
            Sum = astra.pointFocusing(data)
        print(D, Sum)
        return Sum



def tripletFocusing(Pz, D1 = None , focusing = "point", limitValue = 0.0000001, FFFactor = 1 ):
    method = "Powell"
    tolerance = 1e-6

    
    Dmin = [FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
    Dmax = [0.6, 0.6]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    res = sc.optimize.minimize(func, (0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(D1,Pz,focusing) )

    funcVal = func(res.x,D1, Pz, focusing=focusing)

    print("result: ",D1, res.x)

    if funcVal > limitValue:
        return 1

    #beamRatio = math.ceil(astra.beamRatio(*result, None, astra.setupLength, Pz)*100)/100
    acc = [math.floor(num*10)/10 for num in astra.checkAngleAcceptance(D1, *res.x, None, astra.setupLength, Pz)]

    return [D1, *[math.ceil(num*1000000)/10000 for num in res.x],None, astra.setupLength,Pz, funcVal,*acc]



def generate_1D(emittance, num_particles=1000):
    """
    Generate a beam with a given emittance.
    """
    # Generate random points in normalized phase space (x, px)
    theta = np.random.uniform(0, 2 * np.pi, num_particles)  # Random angles
    r = np.sqrt(np.random.normal(0, 1, num_particles))  # Random radii (normalized)
    
    # Scale points to satisfy the emittance ellipse
    x = r * np.sqrt(emittance) * np.cos(theta)
    px = r * np.sqrt(emittance) * np.sin(theta)
    
    return x, px

def generate_beam(emittance, num_particles=1000):
    # emittance in pi mu m mrad
    x, px = generate_1D(emittance, num_particles)
    y, py = generate_1D(emittance, num_particles)


    x = x/1000000
    y = y/1000000

    px = px*Pz/1000
    py = py*Pz/1000

    #here in future, add z, Pz spread

    out = f"0 0 0 0 0 {Pz} 0 -1E-4   1   5\n"

    for i in range(num_particles):
        out += f"{x[i]} {y[i]} {0} {px[i]} {py[i]} {0} {0} -1E-4   1   5\n"

    print(out)

    with open(iniFile, "w") as file:
        file.write(out)

    return x, xPrime, y, yPrime

def plot_beams(x,xPrime,y,yPrime):


    plt.figure(figsize=(10, 8))
    plt.scatter(x, xPrime, s=1, label=f'Emittance: x')
    plt.scatter(y, yPrime, s=1, label=f'Emittance: y')
    plt.xlabel('pos [mu m]')
    plt.ylabel('angle [mrad]')
    plt.title('Phase Space Distribution of Beams')
    plt.legend()
    plt.grid(True)
    plt.show()

def generateFreq(length):
    #generate a random number between 1 and 10 and return a frequency, which will keep the integrated gradient the same
    numbers = list(range(1, 6))  # [1, 2, 3, ..., 10]
    weights = [10,8, 5, 2,1]  # [10, 9, 8, ..., 1]
    
    return 2*random.choices(numbers, weights=weights, k=1)[0]*np.pi/length


def generateNewOffsetsWithConstantIntGrad(offset, switch, wobbles):


    global offsetInMicrons
    global Q1MCXWobbles,Q1MCYWobbles, Q1GradWobbles, Q1SkewAngleWobbles
    global Q2MCXWobbles,Q2MCYWobbles, Q2GradWobbles, Q2SkewAngleWobbles
    global Q3MCXWobbles,Q3MCYWobbles, Q3GradWobbles, Q3SkewAngleWobbles

    global Q1MCXAmp1, Q1MCXFreq1, Q1MCXIniPhase1, Q1MCXAmp2, Q1MCXFreq2, Q1MCXIniPhase2
    global Q1MCYAmp1, Q1MCYFreq1, Q1MCYIniPhase1, Q1MCYAmp2, Q1MCYFreq2, Q1MCYIniPhase2
    global Q1GradAmp, Q1GradFreq, Q1GradIniPhase
    global Q1SkewAngleAmp1 ,Q1SkewAngleFreq1, Q1SkewAngleIniPhase1, Q1SkewAngleAmp2 ,Q1SkewAngleFreq2, Q1SkewAngleIniPhase2 

    global Q2MCXAmp1, Q2MCXFreq1, Q2MCXIniPhase1, Q2MCXAmp2, Q2MCXFreq2, Q2MCXIniPhase2
    global Q2MCYAmp1, Q2MCYFreq1, Q2MCYIniPhase1, Q2MCYAmp2, Q2MCYFreq2, Q2MCYIniPhase2
    global Q2GradAmp, Q2GradFreq, Q2GradIniPhase
    global Q2SkewAngleAmp1 ,Q2SkewAngleFreq1, Q2SkewAngleIniPhase1, Q2SkewAngleAmp2 ,Q2SkewAngleFreq2, Q2SkewAngleIniPhase2 

    global Q3MCXAmp1, Q3MCXFreq1, Q3MCXIniPhase1, Q3MCXAmp2, Q3MCXFreq2, Q3MCXIniPhase2
    global Q3MCYAmp1, Q3MCYFreq1, Q3MCYIniPhase1, Q3MCYAmp2, Q3MCYFreq2, Q3MCYIniPhase2
    global Q3GradAmp, Q3GradFreq, Q3GradIniPhase
    global Q3SkewAngleAmp1 ,Q3SkewAngleFreq1, Q3SkewAngleIniPhase1, Q3SkewAngleAmp2 ,Q3SkewAngleFreq2, Q3SkewAngleIniPhase2 

    Qlength = [36000, 120000, 100000]
    Qgrad = [222,94,57]

    gen.MCXAmp1 = np.random.uniform(0, offset)
    #gen.MCXFreq1 = generateFreq(Qlength[switch]/1000000)
    gen.MCXFreq1 = np.random.uniform(1,50)
    gen.MCXIniPhase1 = np.random.uniform(0,2*np.pi)

    gen.MCXAmp2 = 0
    gen.MCXFreq2 = 0
    gen.MCXIniPhase2 = 0

    gen.MCYAmp1 = np.random.uniform(0, offset)
    gen.MCXFreq1 = np.random.uniform(1,50)
    #gen.MCYFreq1 = generateFreq(Qlength[switch]/1000000)
    gen.MCYIniPhase1 = np.random.uniform(0,2*np.pi)


    gen.MCYAmp2 = 0
    gen.MCYFreq2 = 0
    gen.MCYIniPhase2 = 0

    
    offsetInMicrons[0] = offset*1000000

    Q1MCXWobbles[0],Q1MCYWobbles[0], Q1GradWobbles[0], Q1SkewAngleWobbles[0] = wobbles
    Q2MCXWobbles[0],Q2MCYWobbles[0], Q2GradWobbles[0], Q2SkewAngleWobbles[0] = wobbles
    Q3MCXWobbles[0],Q3MCYWobbles[0], Q3GradWobbles[0], Q3SkewAngleWobbles[0] = wobbles


    if switch == 0:
        gen.generateFieldMap(0.036, 0.777, grad1=222, grad2=222, fileOutputName='quad1', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q1MCXWobbles[0], magCentreYWobbles = Q1MCYWobbles[0], skewAngleWobbles = Q1SkewAngleWobbles[0], gradWobbles = Q1GradWobbles[0])
    
        # save parameters
        Q1MCXAmp1[0], Q1MCXFreq1[0], Q1MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q1MCXAmp2[0], Q1MCXFreq2[0], Q1MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q1MCYAmp1[0], Q1MCYFreq1[0], Q1MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q1MCYAmp2[0], Q1MCYFreq2[0], Q1MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q1GradAmp[0], Q1GradFreq[0], Q1GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q1SkewAngleAmp1[0] ,Q1SkewAngleFreq1[0], Q1SkewAngleIniPhase1[0] = 1000000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q1SkewAngleAmp2[0] ,Q1SkewAngleFreq2[0], Q1SkewAngleIniPhase2[0] = 1000000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)

    elif switch == 1:
        gen.generateFieldMap(0.12, 0.846, grad1=-94, grad2=-94, fileOutputName='quad2', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q2MCXWobbles[0], magCentreYWobbles = Q2MCYWobbles[0], skewAngleWobbles = Q2SkewAngleWobbles[0], gradWobbles = Q2GradWobbles[0])
    
        # save parameters
        Q2MCXAmp1[0], Q2MCXFreq1[0], Q2MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q2MCXAmp2[0], Q2MCXFreq2[0], Q2MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q2MCYAmp1[0], Q2MCYFreq1[0], Q2MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q2MCYAmp2[0], Q2MCYFreq2[0], Q2MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q2GradAmp[0], Q2GradFreq[0], Q2GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q2SkewAngleAmp1[0] ,Q2SkewAngleFreq1[0], Q2SkewAngleIniPhase1[0] = 1000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q2SkewAngleAmp2[0] ,Q2SkewAngleFreq2[0], Q2SkewAngleIniPhase2[0] = 1000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)

    elif switch == 2:
        gen.generateFieldMap(0.1, 0.855, grad1=57, grad2=57, fileOutputName='quad3', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q3MCXWobbles[0], magCentreYWobbles = Q3MCYWobbles[0], skewAngleWobbles = Q3SkewAngleWobbles[0], gradWobbles = Q3GradWobbles[0])
    
        # save parameters
        Q3MCXAmp1[0], Q3MCXFreq1[0], Q3MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q3MCXAmp2[0], Q3MCXFreq2[0], Q3MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q3MCYAmp1[0], Q3MCYFreq1[0], Q3MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q3MCYAmp2[0], Q3MCYFreq2[0], Q3MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q3GradAmp[0], Q3GradFreq[0], Q3GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q3SkewAngleAmp1[0] ,Q3SkewAngleFreq1[0], Q3SkewAngleIniPhase1[0] = 1000000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q3SkewAngleAmp2[0] ,Q3SkewAngleFreq2[0], Q3SkewAngleIniPhase2[0] = 1000000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)



def generateNewOffsets(offset, switch, wobbles):


    global offsetInMicrons
    global Q1MCXWobbles,Q1MCYWobbles, Q1GradWobbles, Q1SkewAngleWobbles
    global Q2MCXWobbles,Q2MCYWobbles, Q2GradWobbles, Q2SkewAngleWobbles
    global Q3MCXWobbles,Q3MCYWobbles, Q3GradWobbles, Q3SkewAngleWobbles

    global Q1MCXAmp1, Q1MCXFreq1, Q1MCXIniPhase1, Q1MCXAmp2, Q1MCXFreq2, Q1MCXIniPhase2
    global Q1MCYAmp1, Q1MCYFreq1, Q1MCYIniPhase1, Q1MCYAmp2, Q1MCYFreq2, Q1MCYIniPhase2
    global Q1GradAmp, Q1GradFreq, Q1GradIniPhase
    global Q1SkewAngleAmp1 ,Q1SkewAngleFreq1, Q1SkewAngleIniPhase1, Q1SkewAngleAmp2 ,Q1SkewAngleFreq2, Q1SkewAngleIniPhase2 

    global Q2MCXAmp1, Q2MCXFreq1, Q2MCXIniPhase1, Q2MCXAmp2, Q2MCXFreq2, Q2MCXIniPhase2
    global Q2MCYAmp1, Q2MCYFreq1, Q2MCYIniPhase1, Q2MCYAmp2, Q2MCYFreq2, Q2MCYIniPhase2
    global Q2GradAmp, Q2GradFreq, Q2GradIniPhase
    global Q2SkewAngleAmp1 ,Q2SkewAngleFreq1, Q2SkewAngleIniPhase1, Q2SkewAngleAmp2 ,Q2SkewAngleFreq2, Q2SkewAngleIniPhase2 

    global Q3MCXAmp1, Q3MCXFreq1, Q3MCXIniPhase1, Q3MCXAmp2, Q3MCXFreq2, Q3MCXIniPhase2
    global Q3MCYAmp1, Q3MCYFreq1, Q3MCYIniPhase1, Q3MCYAmp2, Q3MCYFreq2, Q3MCYIniPhase2
    global Q3GradAmp, Q3GradFreq, Q3GradIniPhase
    global Q3SkewAngleAmp1 ,Q3SkewAngleFreq1, Q3SkewAngleIniPhase1, Q3SkewAngleAmp2 ,Q3SkewAngleFreq2, Q3SkewAngleIniPhase2 

    Qlength = [36000, 120000, 100000]
    Qgrad = [222,94,57]

    gen.MCXAmp1 = np.random.uniform(0, offset)
    gen.MCXFreq1 = np.random.uniform(1, 100)
    gen.MCXIniPhase1 = np.random.uniform(0,2*np.pi)

    gen.MCXAmp2 = np.random.uniform(0, offset)
    gen.MCXFreq2 = np.random.uniform(1, 100)
    gen.MCXIniPhase2 = np.random.uniform(0,2*np.pi)

    gen.MCYAmp1 = np.random.uniform(0, offset)
    gen.MCYFreq1 = np.random.uniform(1, 100)
    gen.MCYIniPhase1 = np.random.uniform(0,2*np.pi)


    gen.MCYAmp2 = np.random.uniform(0, offset)
    gen.MCYFreq2 = np.random.uniform(1, 100)
    gen.MCYIniPhase2 = np.random.uniform(0,2*np.pi)


    # parameters for wobbles of gradient function
    gen.gradAmp = np.random.uniform(0, offset*10*Qgrad[switch])
    gen.gradFreq = np.random.uniform(1,100)
    gen.gradIniPhase = np.random.uniform(0,2*np.pi)


    gen.skewAmp1 = np.random.uniform(0, 2*offset*1000000/(Qlength[switch]) )
    gen.skewFreq1 = np.random.uniform(1, 50)
    gen.skewIniPhase1 = np.random.uniform(0,2*np.pi)

    gen.skewAmp2 = np.random.uniform(0, 2*offset*1000000/(Qlength[switch]) )
    gen.skewFreq2 = np.random.uniform(1, 50)
    gen.skewIniPhase2 = np.random.uniform(0,2*np.pi)
    
    offsetInMicrons[0] = offset*1000000

    Q1MCXWobbles[0],Q1MCYWobbles[0], Q1GradWobbles[0], Q1SkewAngleWobbles[0] = wobbles
    Q2MCXWobbles[0],Q2MCYWobbles[0], Q2GradWobbles[0], Q2SkewAngleWobbles[0] = wobbles
    Q3MCXWobbles[0],Q3MCYWobbles[0], Q3GradWobbles[0], Q3SkewAngleWobbles[0] = wobbles


    if switch == 0:
        gen.generateFieldMap(0.036, 0.777, grad1=222, grad2=222, fileOutputName='quad1', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q1MCXWobbles[0], magCentreYWobbles = Q1MCYWobbles[0], skewAngleWobbles = Q1SkewAngleWobbles[0], gradWobbles = Q1GradWobbles[0])
    
        # save parameters
        Q1MCXAmp1[0], Q1MCXFreq1[0], Q1MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q1MCXAmp2[0], Q1MCXFreq2[0], Q1MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q1MCYAmp1[0], Q1MCYFreq1[0], Q1MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q1MCYAmp2[0], Q1MCYFreq2[0], Q1MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q1GradAmp[0], Q1GradFreq[0], Q1GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q1SkewAngleAmp1[0] ,Q1SkewAngleFreq1[0], Q1SkewAngleIniPhase1[0] = 1000000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q1SkewAngleAmp2[0] ,Q1SkewAngleFreq2[0], Q1SkewAngleIniPhase2[0] = 1000000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)

    elif switch == 1:
        gen.generateFieldMap(0.12, 0.846, grad1=-94, grad2=-94, fileOutputName='quad2', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q2MCXWobbles[0], magCentreYWobbles = Q2MCYWobbles[0], skewAngleWobbles = Q2SkewAngleWobbles[0], gradWobbles = Q2GradWobbles[0])
    
        # save parameters
        Q2MCXAmp1[0], Q2MCXFreq1[0], Q2MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q2MCXAmp2[0], Q2MCXFreq2[0], Q2MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q2MCYAmp1[0], Q2MCYFreq1[0], Q2MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q2MCYAmp2[0], Q2MCYFreq2[0], Q2MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q2GradAmp[0], Q2GradFreq[0], Q2GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q2SkewAngleAmp1[0] ,Q2SkewAngleFreq1[0], Q2SkewAngleIniPhase1[0] = 1000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q2SkewAngleAmp2[0] ,Q2SkewAngleFreq2[0], Q2SkewAngleIniPhase2[0] = 1000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)

    elif switch == 2:
        gen.generateFieldMap(0.1, 0.855, grad1=57, grad2=57, fileOutputName='quad3', nFMPoints = 20, showPlot = False, magCentreXWobbles = Q3MCXWobbles[0], magCentreYWobbles = Q3MCYWobbles[0], skewAngleWobbles = Q3SkewAngleWobbles[0], gradWobbles = Q3GradWobbles[0])
    
        # save parameters
        Q3MCXAmp1[0], Q3MCXFreq1[0], Q3MCXIniPhase1[0] = 1000000*float(gen.MCXAmp1), float(gen.MCXFreq1), float(gen.MCXIniPhase1)
        Q3MCXAmp2[0], Q3MCXFreq2[0], Q3MCXIniPhase2[0] = 1000000*float(gen.MCXAmp2), float(gen.MCXFreq2), float(gen.MCXIniPhase2)
        Q3MCYAmp1[0], Q3MCYFreq1[0], Q3MCYIniPhase1[0] = 1000000*float(gen.MCYAmp1), float(gen.MCYFreq1), float(gen.MCYIniPhase1)
        Q3MCYAmp2[0], Q3MCYFreq2[0], Q3MCYIniPhase2[0] = 1000000*float(gen.MCYAmp2), float(gen.MCYFreq2), float(gen.MCYIniPhase2)
        Q3GradAmp[0], Q3GradFreq[0], Q3GradIniPhase[0] = 1000000*float(gen.gradAmp), float(gen.gradFreq), float(gen.gradIniPhase)
        Q3SkewAngleAmp1[0] ,Q3SkewAngleFreq1[0], Q3SkewAngleIniPhase1[0] = 1000000*float(gen.skewAmp1), float(gen.skewFreq1), float(gen.skewIniPhase1)
        Q3SkewAngleAmp2[0] ,Q3SkewAngleFreq2[0], Q3SkewAngleIniPhase2[0] = 1000000*float(gen.skewAmp2), float(gen.skewFreq2), float(gen.skewIniPhase2)


def fillIniBeamInfo(beamX, beamY):

    global sigX
    global sigPx
    global iniAvrX
    global iniRMSX
    global iniRMSXPrime
    global iniEmitNormX

    global sigY
    global sigPy
    global iniAvrY
    global iniRMSY
    global iniRMSYPrime
    global iniEmitNormY


    # Initial beam info
    sigX[0], sigPx[0],iniAvrX[0],iniRMSX[0], iniRMSXPrime[0],iniEmitNormX[0] = beamX
    sigY[0] ,sigPy[0],iniAvrY[0],iniRMSY[0], iniRMSYPrime[0],iniEmitNormY[0] = beamY


def fillOutBeamInfo(beamX, beamY):

    global outAvrX
    global outRMSX
    global outRMSXPrime
    global outEmitNormX

    global outAvrY
    global outRMSY
    global outRMSYPrime
    global outEmitNormY

    # Output beam info
    outAvrX[0],outRMSX[0], outRMSXPrime[0],outEmitNormX[0] = beamX
    outAvrY[0],outRMSY[0] ,outRMSYPrime[0],outEmitNormY[0] = beamY

def fillSetupInfo(D1,D2,D3,D4,l,PZ, acc):

    global d1, d2, d3, d4, L, pz, accX,accY
    d1[0] = D1
    d2[0] = D2
    d3[0] = D3
    d4[0] = D4
    L[0] = l
    pz[0] = PZ/1000000
    accX[0], accY[0] = acc

def generateAstraOffsets(offset):
    # argument offset expected in m

    Qlength = [0.036, 0.12, 0.1]

    global Q1offsetX,Q1offsetY,Q1rotX,Q1rotY, Q1rotZ
    global Q2offsetX,Q2offsetY,Q2rotX,Q2rotY, Q2rotZ
    global Q3offsetX,Q3offsetY,Q3rotX,Q3rotY, Q3rotZ

    Q1offsetX = np.random.uniform(-offset, offset)
    Q1offsetY = np.random.uniform(-offset, offset)
    Q1rotX = np.random.uniform(-offset/Qlength[0], offset/Qlength[0])
    Q1rotY = np.random.uniform(-offset/Qlength[0], offset/Qlength[0])
    Q1rotZ = np.random.uniform(-offset/Qlength[0], offset/Qlength[0])
    setFile.changeInputData("C_xoff(1)", Q1offsetX)
    setFile.changeInputData("C_yoff(1)", Q1offsetY)
    setFile.changeInputData("C_xrot(1)", Q1rotY)
    setFile.changeInputData("C_yrot(1)", Q1rotX)    
    setFile.changeInputData("C_zrot(1)", Q1rotZ)

    Q2offsetX = np.random.uniform(-offset, offset)
    Q2offsetY = np.random.uniform(-offset, offset)
    Q2rotX = np.random.uniform(-offset/Qlength[1], offset/Qlength[1])
    Q2rotY = np.random.uniform(-offset/Qlength[1], offset/Qlength[1])
    Q2rotZ = np.random.uniform(-offset/Qlength[1], offset/Qlength[1])
    setFile.changeInputData("C_xoff(2)", Q2offsetX)
    setFile.changeInputData("C_yoff(2)", Q2offsetY)
    setFile.changeInputData("C_xrot(2)", Q2rotY)
    setFile.changeInputData("C_yrot(2)", Q2rotX)    
    setFile.changeInputData("C_zrot(2)", Q2rotZ)

    Q3offsetX = np.random.uniform(-offset, offset)
    Q3offsetY = np.random.uniform(-offset, offset)
    Q3rotX = np.random.uniform(-offset/Qlength[2], offset/Qlength[2])
    Q3rotY = np.random.uniform(-offset/Qlength[2], offset/Qlength[2])
    Q3rotZ = np.random.uniform(-offset/Qlength[2], offset/Qlength[2])
    setFile.changeInputData("C_xoff(3)", Q3offsetX)
    setFile.changeInputData("C_yoff(3)", Q3offsetY)
    setFile.changeInputData("C_xrot(3)", Q3rotY)
    setFile.changeInputData("C_yrot(3)", Q3rotX)    
    setFile.changeInputData("C_zrot(3)", Q3rotZ)


def tolerance(Pz, D1, wobbles, offsetAll, runAstraOffsets = False):
    
    # set length at which the focusing will be done
    astra.setupLength = 2.0  #m
    astra.quadType(3)

    setFile.changeInputData("ZSTOP", str(astra.setupLength))

    setFile.changeInputData('File_Efield(1)', "'cavity/3Dquad1'")
    setFile.changeInputData('File_Efield(2)', "'cavity/3Dquad2'")
    setFile.changeInputData('File_Efield(3)', "'cavity/3Dquad3'")

    setFile.changeInputData('File_Aperture(1)', "'aperture/quad1.dat'")
    setFile.changeInputData('File_Aperture(2)', "'aperture/quad2.dat'")
    setFile.changeInputData('File_Aperture(3)', "'aperture/quad3.dat'")

    generateAstraOffsets(0)

    raySolution = [0.15, 0.10423853, 0.28500707]

    acc = astra.checkAngleAcceptance(*raySolution, None, astra.setupLength, Pz)

    # find solution with rays
    #raySolution = tripletFocusing(Pz, D1 = D1)
    if raySolution == 1:
        return 1

    D2 = raySolution[1]
    D3 = raySolution[2]

    # Generate and plot beams
    #plot_beams( generate_beam(1) )

    sigx = 1
    sigy = 1
    sigpx = 1*Pz/1000
    sigpy = 1*Pz/1000
    setFile.changeInputData("sig_x", sigx/1000)
    setFile.changeInputData("sig_y", sigy/1000)
    setFile.changeInputData("sig_px", sigpx)
    setFile.changeInputData("sig_py", sigpy)

    #run the perfect beam
    astra.setupLength = 2.2
    setFile.changeInputData("ZSTOP", str(astra.setupLength))
    setFile.changeInputData("FNAME", iniFile )
    setFile.changeInputData("Distribution", iniFile)
    setFile.changeInputData("RUN", "1")

    astra.runGenerator()

    #offsetAll = [1, 5, 10, 50, 100, 500] #microns
    #offsetAll = [100] #microns

    for offset in offsetAll:
        for i in range(1000):

            print(f"Offset {offset}, Run {i+1}")
            # generate new random offsets, wobbles and field maps of quads
            if i == 0:
                generateNewOffsetsWithConstantIntGrad(0, 0, [0,0,0,0])
                generateNewOffsetsWithConstantIntGrad(0, 1, [0,0,0,0])
                generateNewOffsetsWithConstantIntGrad(0, 2, [0,0,0,0])
            else:
                generateNewOffsetsWithConstantIntGrad(offset/1000000, 0, wobbles)
                generateNewOffsetsWithConstantIntGrad(offset/1000000, 1, wobbles)
                generateNewOffsetsWithConstantIntGrad(offset/1000000, 2, wobbles)
                if runAstraOffsets:
                    generateAstraOffsets(offset/1000000)


            # fill info about setup
            fillSetupInfo(D1,D2,D3,2-D1-D2-D3 - 0.036 - 0.22,2, Pz, acc)

            # generate new beam
            astra.runGenerator()

            # run astra 
            astra.runAstra()

            # plot initial distribution
            '''
            iniBeam = astra.loadData("0000")
            xIni = [math.ceil(line[0]*1000000000)/1000 for line in iniBeam]
            yIni = [math.ceil(line[1]*1000000000)/1000 for line in iniBeam]
            xPIni = [math.ceil(line[3]/Pz*1000000)/1000 for line in iniBeam]
            yPIni = [math.ceil(line[4]/Pz*1000000)/1000 for line in iniBeam]
            plot_beams(xIni,xPIni, yIni,yPIni)
            '''

            # load initial distribution
            iniBeamX = astra.getClosest( astra.loadData("Xemit"), 0 )
            iniBeamY = astra.getClosest( astra.loadData("Yemit"), 0 )

            # save initial distribution to tree
            fillIniBeamInfo([sigx,sigpx, *iniBeamX[2:6] ], [sigy, sigpy, *iniBeamY[2:6]])

            # plot distribution on screen
            '''
            outBeam = astra.loadData("0200")
            xOut = [line[0]*1000000 for line in outBeam]
            yOut = [line[1]*1000000 for line in outBeam]
            xPOut = [line[3]/Pz*1000 for line in outBeam]
            yPOut = [line[4]/Pz*1000 for line in outBeam]
            plot_beams(xOut,xPOut, yOut,yPOut)
            '''
            # load and save final position info
            outBeamX = astra.getClosest( astra.loadData("Xemit"), 2 )
            outBeamY = astra.getClosest( astra.loadData("Yemit"), 2 )
            fillOutBeamInfo(outBeamX[2:6] ,outBeamY[2:6])


            tree.Fill()



if __name__ == "__main__":

    args = sys.argv
    args.pop(0)
    if len(args) != 1:
        print(f"more than 1 argument")
    inputFile = args[0]

    iniFile = "tolerance.ini"

    setFile = SettingsFile("toleranceAnalysis")
    astra = Astra(setFile)
    gen = Generator("generatorFile")

    wobbles = [1,1,0,0] #magnetic centre x, magnetic centre y, gradient, skew angle

    D1 = 0.15
    Pz = 600

    lines = []
    result = []
    with open(inputFile, "r") as file:
        lines = file.readlines()
        for line in lines:
            offs = float(line.split(" ")[0])

            tolerance(Pz*1000000, D1, wobbles, [offs], runAstraOffsets = True)


    tree.Write()
    output_file.Close()

    print(f"All runs finished, output saved to output.root.")
