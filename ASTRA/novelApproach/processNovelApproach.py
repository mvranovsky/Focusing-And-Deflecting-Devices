#!/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    
    #D1 [m],D2 [m],setup length [m],Pz [MeV],"f(x',y') [mrad^2]",D1 + l1 + D2 + l2 [m],l1 [m],l2 [m],int Grad1 [T],int Grad2 [m]

    df = pd.read_csv("outputNovelApproach.csv")

    data = df.values.tolist()

    diffD1 = []
    diffD2 = []
    diffLength = []
    Pz = []
    absChangeInLength = []
    for i in range(0, len(data),2):
        diffD1.append((data[i][0] - data[i+1][0])*100)
        diffD2.append((data[i][1] - data[i+1][1])*100)
        diffLength.append((data[i][5] - data[i+1][5])*100)
        Pz.append(data[i][3])
        absChangeInLength.append( math.fabs(data[i][5] - data[i+1][5])*100/data[i][5] )


    print(Pz)

    plt.scatter(Pz, diffD1, label='Difference in D1 [cm]', color='blue')
    plt.scatter(Pz, diffD2, label='Difference in D2 [cm]', color='red')
    plt.legend()
    plt.xlabel("Pz [MeV]")
    plt.ylabel("distance [cm]")
    plt.savefig("differencesInD1AndD2.png", format='png', dpi=300)
    plt.show()


    plt.scatter(Pz, diffLength, color = 'black')
    plt.title("Difference in length of the setup")
    plt.xlabel("Pz [MeV]")
    plt.ylabel("distance [cm]")
    plt.savefig("differencesInLength.png", format='png', dpi=300)
    plt.show()


    plt.scatter(Pz, absChangeInLength, color='black')
    plt.title("Difference in lengths relative to the untapered quads")
    plt.xlabel("Pz [MeV]")
    plt.ylabel("rel. change [%]")
    plt.savefig("relativeDifferences.png", format='png', dpi=300)
    plt.show()