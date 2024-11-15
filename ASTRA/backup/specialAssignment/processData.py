#!/usr/bin/python3


from AstraWrapper.SettingsFile import SettingsFile
from AstraWrapper.Astra import Astra
import scipy as sc
import subprocess
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import ROOT 
import os
#-----------------------------------------------------------------------------------------------
def mergeResults():

    path = "/home/michal/Desktop/RPIT/ASTRA/runParallel/Run11/"
    data_frames = []

    dirs = os.listdir(path)

    for d in dirs:
        d = os.path.join(path,d)

        csv_files = [file for file in os.listdir( d ) if file.endswith('.csv')]
        print(csv_files)

        # Loop through and read each CSV file
        for file in csv_files:
            file_path = os.path.join(d, file)
            df = pd.read_csv(file_path)
            data_frames.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(data_frames, ignore_index=True)

    # Save merged DataFrame to a new CSV file (optional)
    merged_df.to_csv('merged_output.csv', index=False)

#-----------------------------------------------------------------------------------------------
def createTTree(file_path, tree_name, output_file):
    """
    Reads a CSV file and stores its contents into a ROOT TTree, then saves the TTree to a ROOT file.
    
    Parameters:
        file_path (str): Path to the .csv file.
        tree_name (str): Name of the TTree.
        output_file (str): Name of the output ROOT file.
    """
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Create a ROOT file and a TTree
    root_file = ROOT.TFile(output_file, "RECREATE")
    tree = ROOT.TTree(tree_name, "Data from CSV")

    # Dictionary to store branches, automatically named after DataFrame columns
    branches = {}
    
    # Create a branch for each column in the DataFrame
    for col in df.columns:
        # Set up branch with correct type for each column
        if df[col].dtype == 'int64':
            branches[col] = np.zeros(1, dtype=np.int32)  # Convert int64 to int32 for ROOT compatibility
            tree.Branch(col, branches[col], f"{col}/I")
        elif df[col].dtype == 'float64':
            branches[col] = np.zeros(1, dtype=np.float32)  # Convert float64 to float32 for ROOT compatibility
            tree.Branch(col, branches[col], f"{col}/F")
        else:
            raise TypeError(f"Unsupported data type for column '{col}'")

    # Loop over each row in the DataFrame and fill the tree
    for index, row in df.iterrows():
        for col in df.columns:
            branches[col][0] = row[col]
        tree.Fill()

    # Write and close the ROOT file
    root_file.Write()
    root_file.Close()
    print(f"TTree '{tree_name}' has been saved to '{output_file}'")
#-----------------------------------------------------------------------------------------------
def func(D,D1, Pz):

    data = []
    try:
        data = astra.runRef(D1, *D, None, astra.setupLength,Pz, False)
    except Exception as e:
        print(f"exception: {e}")
        return 1E+9
    else:
        Sum = astra.parallelFocusing(data)
        return Sum
#-----------------------------------------------------------------------------------------------

def tripletFocusing(D1,Pz, FFFactor = 1 , limitValue = 0.0001):
    method = "Powell"
    tolerance = 1e-6
    fields = ["top hat fields", "Astra fringe fields", "field profiles"]

    astra.quadType(1)
    astra.setupLength = 1.2  #m


    Dmin = [FFFactor*(astra.bores[0] + astra.bores[1]) , FFFactor*(astra.bores[1] + astra.bores[2]) ]
    Dmax = [0.6, 0.6]
    bounds = [(low, high) for low, high in zip(Dmin, Dmax)]

    res = sc.optimize.minimize(func, (0.1,0.1), method=method, tol=tolerance, bounds=bounds, args=(D1,Pz) )

    funcVal = func(res.x,D1, Pz)

    dataX = astra.loadData('ref', 1)
    dataY = astra.loadData('ref', 2)
    #print(dataX, dataY)

    if funcVal > limitValue:
        raise ValueError(f"Something went wrong, this time when running the minimization procedure, the solution was not smaller than {limitValue}.")


    return dataX, dataY
#-----------------------------------------------------------------------------------------------

def fixData():

    df = pd.read_csv("merged_output.csv")

    df.iloc[:, :3] = df.iloc[:, :3] / 100

    # Save the modified DataFrame back to the same file
    df.to_csv("merged_output.csv", index=False)



#----------------------------------------------------------------------------------------------
def analyze(Pz = None, D1 = None, limFuncValue = 0.00001 ):

    if (Pz != None and D1 != None) or (Pz == None and D1 == None):
        raise ValueError(f"Function only works when either D1 or Pz is specified, not both or neither!!")
    


    df = pd.read_csv('merged_output.csv')
    data = df.values.tolist()
    data_chosen = []
    if Pz != None and (Pz > 200 and Pz < 990) and Pz % 10 == 0:
        print(f"Will be showing results for Ekin = {Pz} MeV")
        data_chosen = [row for row in data if row[6] < limFuncValue and row[5] == Pz*1000000]
    elif D1 != None and (D1 > 0.7 and D1 < 30.6):
        data_chosen = [row for row in data if row[6] < limFuncValue and row[0] == D1]
        print(f"Will be showing results for D1 = {D1} cm")
    else:
        raise ValueError(f"Expecting D1 as 0.8, 1.0, 1.2...up to 30.6 cm or Pz in range (200, 990) MeV and divisible by 10, so 200, 210, 220...")



    if(len(data_chosen) == 0 ):
        print(f"For this input, there are no solutions.")
        return



    #acceptance
    setupL, rang = [],[]
    descriptionX = ''
    xlimits = []
    if D1 != None:
        rang = [row[5]*1e-6 for row in data_chosen]
        descriptionX = 'Pz [MeV]'
        xlimits.append( 200 )  #left border
        xlimits.append( 1000 )  #left border

    else:
        rang = [row[0] for row in data_chosen]
        descriptionX = 'D1 [cm]'
        xlimits.append( 0.0 )  #left border
        xlimits.append( 31 )  #left border

    bestVal = 1E+6
    bestLine = []
    for line in data_chosen:
        l = line[0] + astra.AstraLengths[0]*100 + line[1] + astra.AstraLengths[1]*100 + line[2] + astra.AstraLengths[2]*100 + astra.bores[2]*100
        setupL.append( l )
        
        val = l*0.01 + (line[9] - 1) + 100/(line[7]*line[8])
        #print(l*0.01, (line[9] -1), 100/(line[7]*line[8]) )
        if val < bestVal:
            bestVal = float(val)
            bestLine = list(line + [l])


    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # Plot in each subplot
    axes[0, 0].scatter(rang,[row[7] for row in data_chosen], color='blue', label='x acceptance' )
    axes[0, 0].scatter(rang,[row[8] for row in data_chosen], color='red', label='y acceptance' )
    axes[0, 0].set_xlabel(descriptionX)
    axes[0, 0].set_ylabel("acceptance [mrad]")
    axes[0, 0].set_title("Acceptance")
    axes[0, 0].set_ylim(0,50)
    axes[0, 0].set_xlim(*xlimits)
    axes[0, 0].legend(loc="upper left")


    axes[0, 1].scatter(rang,[row[9] for row in data_chosen], color='black' )
    axes[0, 1].set_xlabel(descriptionX)
    axes[0, 1].set_xlim(*xlimits)
    axes[0, 1].set_ylim(0, 10)
    axes[0, 1].set_ylabel("beam size ratio [-]")
    axes[0, 1].set_title("beam size ratio")

    axes[1, 0].scatter(rang, setupL, color='black')
    axes[1, 0].set_xlabel(descriptionX)
    axes[1, 0].set_xlim(*xlimits)
    axes[1, 0].set_ylim(0, 120 )
    axes[1, 0].set_ylabel('length [cm]')
    axes[1, 0].set_title("setup size")

    '''
    dataX, dataY = [], []
    if D1 != None:
        dataX, dataY = tripletFocusing(D1/100, bestLine[5] )
    else: 
        dataX, dataY = tripletFocusing(bestLine[0]/100, Pz*1000000)
   
    axes[1, 1].plot([line[0] for line in dataX], [line[5] for line in dataX], color='blue', label='x offset')
    axes[1, 1].plot([line[0] for line in dataY], [line[6] for line in dataY], color='red', label='y offset')
    axes[1, 1].plot([0, dataX[-1][0] ],[0,0], color='black', label='beamline')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('z [m]')
    axes[1, 1].set_ylabel('offset [mm]')
    axes[1, 1].set_title(f"best solution with acceptance = {[ math.ceil(num*10)/10 for num in bestLine[7:9] ]} mrad,\nbeam ratio = {math.ceil(100*bestLine[9])/100}, setup size = {math.ceil(100*bestLine[10])/100} cm,\nDs = {bestLine[0:3]} cm")
    '''

    axes[1, 1].scatter(rang,[row[1] for row in data_chosen], color='blue', label='D2' )
    axes[1, 1].scatter(rang,[row[2] for row in data_chosen], color='red', label='D3' )
    axes[1, 1].set_xlabel(descriptionX)
    axes[1, 1].set_ylabel("distance [cm]")
    axes[1, 1].set_title("Scaling of distances D2, D3")
    axes[1, 1].set_ylim(0,50)
    axes[1, 1].set_xlim(*xlimits)
    axes[1, 1].legend(loc="upper left")


    plt.tight_layout()
    if Pz != None:
        plt.savefig(f"systematicAnalysis/analysisPz:{Pz}MeV.png", format='png', dpi=300)
    else:
        plt.savefig(f"systematicAnalysis/analysisD1:{D1}cm.png", format='png', dpi=300)


    plt.show()



if __name__ == "__main__":


    astra = Astra("parallelBeam")

    #mergeResults()
    fixData()
    #PZ = [9.5E+8]
    PZ = [ 3.0E+8, 3.5E+8, 4.0E+8, 4.5E+8, 5.0E+8, 5.5E+8,6.0E+8, 6.5E+8, 7.0E+8, 7.5E+8, 8.0E+8, 8.5E+8, 9.0E+8, 9.5E+8]
    D1 = [0.8, 1.0 ,5.0, 10.0, 15.0,20.0, 25.0, 30.0 ]
    for pz in PZ:
        analyze(Pz = pz/1000000)
    for d1 in D1:
        analyze(D1=d1)

















