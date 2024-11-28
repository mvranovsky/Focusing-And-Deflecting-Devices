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


def plotAll3QuadTypes():
    try:
        astra.quadType(0)
        data = astra.runRef(*result, None, astra.setupLength,Pz, True)

        astra.quadType(1)
        data1 = astra.runRef(*result, None, astra.setupLength,Pz, True)

        astra.quadType(2)
        data2 = astra.runRef(*result, None, astra.setupLength,Pz, True)


        plt.plot([row[0] for row in data[1]], [row[5] for row in data[1]], color="blue", label='top hat fields, x offset ')
        plt.plot([row[0] for row in data[2]], [row[6] for row in data[2]], color="green", label='top hat fields, y offset ')

        plt.plot([row[0] for row in data1[1]], [row[5] for row in data1[1]], color="red", label='astra fringe fields, x offset')
        plt.plot([row[0] for row in data1[2]], [row[6] for row in data1[2]], color="yellow", label='astra fringe fields, y offset')

        plt.plot([row[0] for row in data2[1]], [row[5] for row in data2[1]], color="pink", label='field profiles, x offset')
        plt.plot([row[0] for row in data2[2]], [row[6] for row in data2[2]], color="magenta", label='field profiles, y offset')

        plt.plot([0, astra.setupLength], [0,0], color='black', label='beamline')
        plt.xlabel("z [m]")
        plt.ylabel("x [mm]")
        plt.legend()
        plt.savefig("quadStudy/trajectoriesForQuadTypes.png", format='png', dpi=300)
        plt.show()
        #plt.close()
    except Exception as e:
        print(f"Problem when plotting all 3 quad types: {e}")


def func(D, Pz):
    data = []
    try:
        data = astra.runRef(*D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return 1E+9
    else:
        Sum = astra.pointFocusing(data)
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
        Sum = astra.pointFocusing(data)
        print(D, Sum)
        return Sum


def tripletFocusing(Pz, D1 = None , defaultFieldType=1, limitValue = 0.0001, FFFactor = 1 ):
    method = "COBYLA"
    method = "Powell"
    tolerance = 1e-10
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    astra.quadType(defaultFieldType)
    astra.setupLength = 2.0  #m

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
        #beamRatio = astra.beamRatio(*result, None, astra.setupLength, Pz)
        #acc = astra.checkAngleAcceptance(*result, None, astra.setupLength, Pz)
        #astra.quadType(defaultFieldType)
        funcVal = astra.pointFocusing( astra.runRef(*result, None, astra.setupLength,Pz, False ) )
        #astra.plotRefXY(*result, None, astra.setupLength, Pz)

        return [*[math.ceil(num*10000000)/10000000 for num in result],None, astra.setupLength,Pz, funcVal]
    except:
        raise ValueError("Did not find solution.")


def checkMomentum():

    data = astra.loadData("0000")

    Pz = data[0][5]

    for line in data[1:-1]:
        num = math.sqrt( line[3]**2 + line[4]**2 + (Pz + line[5])**2 )/1000000
        print(num)


def study(solution, fields , key , Vals, description):


    # here define the ranges and number of bins
    histRange = [-0.2,0.2, -1, 1]  # mm
    nBins = 51
    nParticles = 10000

    hist = []

    acc = astra.checkAngleAcceptance(*solution[:6])

    setFile.changeInputData("Distribution", astra.fileName + ".ini")
    setFile.changeInputData("sig_px", str(5500000))
    setFile.changeInputData("sig_py", str(5500000))    
    setFile.changeInputData("RUN", 1)
    setFile.changeInputData("IPart", str(nParticles))
    astra.quadType(fields)


    histPx = []
    histPy = []
    histX = []
    histY = []

    for i in range(len(Vals)):
        histPx.append( ROOT.TH1D("histPx" + str(i), "Initial distribution of p_{x} of particles in a beam; p_{x} [MeV/c];counts",40, -math.ceil( acc[0]*solution[5]/1000000000 ) , math.ceil( acc[0]*solution[5]/1000000000 ) ) )
        histPy.append( ROOT.TH1D("histPy" + str(i), "Initial distribution of p_{y} of particles in a beam; p_{y} [MeV/c];counts", 40, -math.ceil( acc[0]*solution[5]/1000000000 ) , math.ceil( acc[0]*solution[5]/1000000000 ) ) )
        histX.append( ROOT.TH1D("histX" + str(i) ,description[0] + " = " + Vals[i] + " " + description[1] + "; x [mm]; counts", nBins, histRange[0], histRange[1]) )
        histY.append( ROOT.TH1D("histY" + str(i) ,description[0] + " = " + Vals[i] + " " + description[1] + "; y [mm]; counts", nBins, histRange[2], histRange[3]) )

        originalVal = setFile.readOption(key) 
        print(f"Now running {key} with value {float(originalVal) + float(Vals[i])}.")
        setFile.changeInputData( key, str(float(originalVal) + float(Vals[i]) ) )


        if key == 'sig_x':
            generator.generateSource(nParticles, Pz = solution[5],distPx='U', distPy='U', sig_Px = 5500000, sig_Py=5500000, distX='U', distY='U', sig_X=float(Vals[i])/1000)
        elif key == 'sig_y':
            generator.generateSource(nParticles, Pz = solution[5],distPx='U', distPy='U', sig_Px = 5500000, sig_Py=5500000, distX='U', distY='U', sig_Y=float(Vals[i])/1000)
        elif key == 'x_off':
            generator.generatePointSource(nParticles, Pz = solution[5], sig_Px = 5500000, sig_Py=5500000, distPx='U', distPy='U', xOffset=float(Vals[i])/1000)
        elif key == 'y_off':
            generator.generatePointSource(nParticles, Pz = solution[5], sig_Px = 5500000, sig_Py=5500000, distPx='U', distPy='U', yOffset=float(Vals[i])/1000)
        else:
            astra.runGenerator()




        res = astra.runAstra()
        print(res.stdout)
        print(res.stderr)
        if res.stderr != '':
            raise ValueError(f"Astra returned with error: {res.stderr}")

        data = astra.loadData("0200")
        dataBegin = astra.loadData("0000")

        if len(data) != len(dataBegin):
            raise ValueError("Somethings wrong, input data length not equal to output.")

        Pz = data[0][5]
        for j, row in enumerate(data):

            if row[9] == 5:
                numX = 1e+3*(row[0] + row[3]*row[2]/(Pz + row[5] ) )
                numY = 1e+3*(row[1] + row[4]*row[2]/(Pz + row[5] ) )
                #print(numX, numY)

                histX[i].Fill( float( numX ) )
                histY[i].Fill( float( numY ) )
            
                histPx[i].Fill(float(dataBegin[j][3]*1e-6))
                histPy[i].Fill(float(dataBegin[j][4]*1e-6))

        setFile.changeInputData(key, originalVal)


    file = ROOT.TFile("file.root", "recreate")
    file.cd()
    for i in range( len(histX) ):
        histX[i].Write()
        histY[i].Write()
        histPx[i].Write()
        histPy[i].Write()
    file.Close()

def changeInput(filename, tag, newVal):

    line = []
    with open(filename, "r") as file:
        line = file.readlines()[0].split()

    if tag == "x":
        line[0] = str(newVal)
    elif tag == "y":
        line[1] = str(newVal)
    elif tag == "z":
        line[2] = str(newVal)
    elif tag == "px":
        line[3] = str(newVal)
    elif tag == "py":
        line[4] = str(newVal)
    elif tag == "pz":
        line[5] = str(newVal)

    out = ''
    for num in line:
        out += str(num) + " "

    out = out[:-1] + "\n"

    with open(filename, "w") as file:
        file.write(out)

def linear_function(x, slope):
    return slope * x

def linear_function_with_C(x,slope, C):
    return slope * x + C

def quadratic_function(x, pa2, pa1, pa0):
    return x*x*pa2 + pa1*x1 + pa0

def fit(xVals, x1, y1, x2, y2, key):

    x = np.array(xVals)
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)

    slope = []
    if  "Q_grad" in key:
        params, covariance = sc.optimize.curve_fit(linear_function_with_C, x, x1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function_with_C, x, y1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function_with_C, x, x2)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function_with_C, x, y2)
        slope.append(float(params[0]))
    elif "sig_Ekin" in key:
        params, covariance = sc.optimize.curve_fit(quadratic_function, x, x1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(quadratic_function, x, y1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(quadratic_function, x, x2)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(quadratic_function, x, y2)    
        slope.append(float(params[0]))
    else:
        params, covariance = sc.optimize.curve_fit(linear_function, x, x1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function, x, y1)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function, x, x2)
        slope.append(float(params[0]))

        params, covariance = sc.optimize.curve_fit(linear_function, x, y2)
        slope.append(float(params[0]))

    return slope

def quantify(solution, fields , key , Vals, description, outputDir):

    #su v podstate 3 rozne moznosti ako budem kvantifikovat jednotlive vlastnosti:
    #bud vzdy prebehnem jeden ray s nejakym offsetom a budem sa pozerat ako sa final position bude menit
    #emittance graf zavislosti eps_2 na eps_1 -> problem je, ze tam menim pociatocnu emittance a nevidim tam 
    
    Q1pos = float(setFile.readOption("Q_pos(1)"))
    Q2pos = float(setFile.readOption("Q_pos(2)"))
    Q3pos = float(setFile.readOption("Q_pos(3)"))


    #acc = astra.checkAngleAcceptance(*solution[:6])
    astra.changeMom(solution[5], xAngle=5, yAngle=5, xoff=0.1, yoff = 0.1)
    astra.changePositions(*solution[:5])

    astra.quadType(fields)
    xVals, xAngle_X, xAngle_Y,yAngle_X, yAngle_Y  = [], [],[], [],[]
    xOffset_X, xOffset_Y, yOffset_X, yOffset_Y, noOA_X, noOA_Y = [],[],[],[],[],[]
    xDescript, yDescript = '',''
    rang = [-5,5]

    for i in range(-100, 100):
        if i % 10 != 0:
            continue

        if key == "x_off":
            changeInput("test1.ini","x", i/100000)
            changeInput("test2.ini","x", i/100000)
            xVals.append(i/100)
            xDescript = 'initial offset x [mm]'
            yDescript = 'final offset x [mm]'
        elif key == "y_off":
            changeInput("test1.ini","y", i/100000)
            changeInput("test2.ini","y", i/100000)
            xVals.append(i/100)
            xDescript = 'initial offset y [mu m]'
            yDescript = 'final offset y [mm]'
        elif key == "sig_Ekin":
            changeInput("test1.ini","pz",solution[5] + i*500000)
            changeInput("test2.ini","pz", solution[5] + i*500000)
            xVals.append( (solution[5] + i*500000)/1000000 )
            xDescript = 'Pz [MeV]'
            yDescript = 'final offset x [mm]'
        elif "off(" in key:
            setFile.changeInputData(key, i/100000)
            xVals.append(i/100)
            xDescript = 'quadOffset [mm]'
            yDescript = 'final offset [mm]'
        elif "rot(" in key:
            setFile.changeInputData(key, i/5000)
            xVals.append(i/5)
            xDescript = 'quadRotation [mrad]'
            yDescript = 'final offset [mm]'
        elif "Q_pos(1)" in key:
            setFile.changeInputData(key, str(Q1pos + i/100000))
            xVals.append(i/100)
            xDescript = 'quad1 offset_Z [mm]'
            yDescript = 'final offset [mm]'
        elif "Q_pos(2)" in key:
            setFile.changeInputData(key, str(Q2pos + i/100000))
            xVals.append(i*100)
            xDescript = 'quad2 offset_Z [mm]'
            yDescript = 'final offset [mm]'
        elif "Q_pos(3)" in key:
            setFile.changeInputData(key, str(Q3pos + i/100000))
            xVals.append(i*100)
            xDescript = 'quad3 offset_Z [mm]'
            yDescript = 'final offset [mm]'
        elif "Q_grad(1)" in key:
            setFile.changeInputData(key, str(222 + i/5))
            xVals.append(222 + i/5)
            xDescript = 'quad1 gradient [T/m]'
            yDescript = 'final offset [mm]'
        elif "Q_grad(2)" in key:
            setFile.changeInputData(key, str(-94 + i/5))
            xVals.append(-94 + i/5)
            xDescript = 'quad1 gradient [T/m]'
            yDescript = 'final offset [mm]'
        elif "Q_grad(3)" in key:
            setFile.changeInputData(key, str(57 + i/5))
            xVals.append(57 + i/5)
            xDescript = 'quad1 gradient [T/m]'
            yDescript = 'final offset [mm]'
            

        setFile.changeInputData("Distribution", "test1.ini")
        setFile.changeInputData("RUN", 1)

        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout):
            cleanUp(key, solution[5], Q1pos, Q2pos, Q3pos)
            raise ValueError(f"Astra did not run properly in runRef() with moreData=False.")

        bestLine = astra.getClosest( astra.loadData("ref", 1) )
        if bestLine == 1:
            cleanUp(key, solution[5], Q1pos, Q2pos, Q3pos)
            raise ValueError("Could not get close to the end screen in runRef() method, check it.")

        xAngle_X.append( bestLine[5] )   #in mm
        xAngle_Y.append( bestLine[6] )

        setFile.changeInputData("Distribution", "test2.ini")
        setFile.changeInputData("RUN", 2)

        res = astra.runAstra()

        if not (res.stderr == '' or 'Goodbye' in res.stdout):
            cleanUp(key, solution[5], Q1pos, Q2pos, Q3pos)
            raise ValueError(f"Astra did not run properly in runRef() with moreData=False.")

        bestLine = astra.getClosest( astra.loadData("ref", 2) )
        if bestLine == 1:
            cleanUp(key, solution[5], Q1pos, Q2pos, Q3pos)
            raise ValueError("Could not get close to the end screen in runRef() method, check it.")

        yAngle_X.append( bestLine[5] )
        yAngle_Y.append( bestLine[6] )


    fitRes = fit(xVals, xAngle_X, xAngle_Y, yAngle_X, yAngle_Y, key)
    print(fitRes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # Plot in each subplot
    axes[0].scatter(xVals, xAngle_X, color='blue', label='x offset' )
    axes[0].scatter(xVals, xAngle_Y, color='red', label='y offset' )
    axes[0].plot(xVals, linear_function(fitRes[0],np.array(xVals)), color='green',label = f"fit: y = {fitRes[0]}x" )
    axes[0].plot(xVals, linear_function(fitRes[1],np.array(xVals)), color='purple',label = f"fit: y = {fitRes[1]}x" )
    axes[0].set_xlabel(xDescript)
    axes[0].set_ylabel(yDescript)
    axes[0].set_ylim(*rang)
    axes[0].set_title("ray with initial x angle")
    axes[0].legend(loc="upper right")


    axes[1].scatter(xVals, yAngle_X, color='blue', label='x offset' )
    axes[1].scatter(xVals, yAngle_Y, color='red', label='y offset' )
    axes[1].plot(xVals, linear_function(fitRes[2],np.array(xVals)), color='green',label = f"fit: y = {fitRes[2]}x" )
    axes[1].plot(xVals, linear_function(fitRes[3],np.array(xVals)), color='purple',label = f"fit: y = {fitRes[3]}x" )
    axes[1].set_xlabel(xDescript)
    axes[1].set_ylabel(yDescript)
    axes[1].set_ylim(*rang)
    axes[1].set_title("ray with initial y angle")
    axes[1].legend(loc="upper right")
    plt.tight_layout()


    plt.savefig(outputDir + "/" + variable + f"D1:{solution[0]*100}cm_Pz{solution[5]/1000000}MeV" + ".png", format="png", dpi=300)

    
    plt.close()

    cleanUp(key, solution[5], Q1pos, Q2pos, Q3pos)

    return fitRes


def cleanUp(key, Pz,Q1pos, Q2pos,Q3pos):

    #after running, set everything back to default values
    changeInput("test1.ini","x", 0)
    changeInput("test1.ini","y", 0)
    changeInput("test1.ini","z", 0)

    changeInput("test1.ini","px", Pz/1000)
    changeInput("test1.ini","py", 0)
    changeInput("test1.ini","pz", Pz)

    changeInput("test2.ini","x", 0)
    changeInput("test2.ini","y", 0)
    changeInput("test2.ini","z", 0)

    changeInput("test2.ini","px", 0)
    changeInput("test2.ini","py",Pz/1000)
    changeInput("test2.ini","pz", Pz)

    if "off(" in key:
        setFile.changeInputData(key, 0)
    elif "rot(" in key:
        setFile.changeInputData(key, 0)
    elif "Q_pos(1)" in key:
        setFile.changeInputData(key, str(Q1pos) )
    elif "Q_pos(2)" in key:
        setFile.changeInputData(key, str(Q2pos) )
    elif "Q_pos(3)" in key:
        setFile.changeInputData(key, str(Q3pos) )
    elif "Q_grad(1)" in key:
        setFile.changeInputData(key, str(222))
    elif "Q_grad(2)" in key:
        setFile.changeInputData(key, str(-94))
    elif "Q_grad(3)" in key:
        setFile.changeInputData(key, str(57))




if __name__ == "__main__":

    setFile = SettingsFile("parallelBeam")
    astra = Astra(setFile)
    generator = Generator('parallelBeam')
    outputFileDir = 'quadStudy'

    # default value 
    setFile.changeInputData("H_max", "0.001")
    

    Pzs = [3E+8, 3.5E+8,4E+8,4.5E+8, 5E+8,5.5E+8, 6E+8,6.5E+8, 7E+8,7.5E+8, 8E+8,8.5E+8, 9E+8, 9.5E+8, 1E+9]
    #Pzs = [5E+8]

    args = sys.argv
    args.pop(0)
    if len(args) != 1:
        print(f"more than 1 argument")
    inputFile = args[0]

    stri = ''
    with open(inputFile, "r") as file:
        stri = file.readlines()

    D1s = [float(d1) for d1 in stri ]



    astra.quadType(1)

    lines = []
    with open("study2.txt", "r") as file:
        lines = file.readlines()
    
    out = []

    for d1 in D1s:
        for Pz in Pzs:
            try:
                sol = tripletFocusing(Pz, D1=d1)
            except Exception:
                print("error when looking for sol")
                continue

            if sol[6] > 0.0001:
                print("smal func val")
                continue

            for line in lines:
                if line[0] == "#":
                    continue
                line = line.replace("\n", "").split(" ")
                if len(line) < 8:
                    print("not good length of line")
                    continue
                variable = line[0]
                key = line[1]
                rangeVal = [j for j in line[2:6] ]
                descript = [line[6], line[7]]


                print(f"Now running analysis of {variable} with solution found for Pz = {Pz/1000000} MeV and D1 = {d1*100} cm")


                try:
                    dat = quantify(sol, 1 , key, rangeVal, descript, f"systematicAnalysis")
                except Exception as e:
                    print(f"An error occured when trying to quantify: {e}")
                    continue

                dat = dat.flatten() if hasattr(dat, 'flatten') else dat

                current = [math.ceil(num*100000)/1000 for num in sol[:3]] + [None, sol[4]] + [math.floor(sol[5]/1000000), sol[6]] + [variable, key] + [math.ceil(num*1000)/1000 for num in dat]
            
                out.append( list(current) )
                break
            break
        break


    df = pd.DataFrame(out)
    print(out)
    custom_header = ["D1 [cm]", "D2 [cm]", "D3 [cm]", "D4 [cm]", "setup length [m]", "Pz [MeV/c]", "f(x',y') [mrad^2]", "studied variable", "Astra key", "xAngle xOffset slope[mm]", "xAngle yOffset slope [mm]", "yAngle xOffset slope [mm]", "yAngle yOffset slope [mm]"]
    df.to_csv(f"outputSysStudy.csv", header=custom_header, index=False)



    '''
    solution1 = [[0.1, 0.07479541, 0.21488641, None, 2.0, 500000000.0, 1.0430630338900002e-05, 18.96047, 23.98575, 13.71674], 
                [0.1, 0.07541914, 0.21774701, None, 2.0, 500000000.0, 3.0991848625e-28, 18.84115, 23.97918, -6.9294], 
                [0.1, 0.07654301, 0.22160818, None, 2.0, 500000000.0, 2.520778753000001e-27, 18.66717, 24.25503, -2.89903]
    ]


    solution2 = [[0.2, 0.04974947, 0.18774188, None, 2.0, 500000000.0, 2.907600658e-05, 16.05652, 13.75787, -5.10633], 
                [0.2, 0.04992214, 0.1927, None, 2.0, 500000000.0, 1.205243179076e-26, 16.05947, 13.75462, -12.99598], 
                [0.2, 0.05066138, 0.19653194, None, 2.0, 500000000.0, 7.706379658e-27, 16.08974, 13.86523, -0.13584]

    ]

    solution3 = [[0.3, 0.03715967, 0.15925456, None, 2.0, 500000000.0, 1.0105527860000002e-05, 11.20682, 9.63127, -1.68065],
                [0.3, 0.03784807, 0.16359225, None, 2.0, 500000000.0, 1.1436397997e-28, 11.21436, 9.62756, 0.73781],
                [0.3, 0.03813884, 0.16765291, None, 2.0, 500000000.0, 8.7935112421e-28, 11.22875, 9.69154, 3.22959]]
    '''


