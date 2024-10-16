#!/usr/bin/python3

from settingsFile import SettingsFile
import subprocess
import math
import matplotlib as plt


class Astra:



    def __init__(self, settingsFile):

        if not settingsFile:
            print(f"The settings file could not be found. Leaving...")
            return 1
        self.setFile = settingsFile
        self.fileName = settingsFile.fileName
        self.nameOfFiles = ["test0.ini", "test1.ini", "test2.ini", "test3.ini", "test4.ini"]
        self.AstraLengths = [0.03619, 0.12429, 0.09596]
        self.FPlengths = [0.035, 0.120, 0.105]
        self.bores = [0.007, 0.018, 0.030]

        self.sig_xAngle = 1  #mrad
        self.sig_yAngle = 1  #mrad
        self.nParticles = "500"
        self.sig_z=0.1    #mm
        self.sig_Ekin=3E+3     #keV
        self.sig_x=2.0E-3    #mm
        self.sig_y=2.0E-3    #mm

        proc = subprocess.Popen(
            ['/bin/bash'], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

        proc.stdin.write("source /opt/intel/oneapi/setvars.sh")
        proc.stdin.flush()

        self.process = proc

    def __del__(self):
        self.process.stdin.write("exit\n")
        self.process.stdin.flush()

    def aperture(self,yes):
        if yes:
            self.setFile.changeInputData("LApert", "T")
        else:
            self.setFile.changeInputData("LApert","F")    

    def quadrupoleType(self,switcher):
        #switcher between different types of quadrupoles- 
        #all have the same integrated gradient
        #0 == top hat shaped fields
        #1 == Astra fields with fringe fields
        #2 == field profiles from fits on real data, real quadrupoles
        #3 == EM field maps in cavity namelist (incomplete)
        
        if switcher == 0:  #top hat shaped quads from namelist quadrupole
            self.setFile.changeInputData("Lquad", "T")
            self.setFile.changeInputData("LEField", "F") 
            self.setFile.changeInputData("Q_bore(1)", "1E-9")
            self.setFile.changeInputData("Q_bore(2)", "1E-9")
            self.setFile.changeInputData("Q_bore(3)", "1E-9")
            self.setFile.changeInputData("Q_length(1)",str(self.AstraLengths[0]))
            self.setFile.changeInputData("Q_length(2)",str(self.AstraLengths[1]))
            self.setFile.changeInputData("Q_length(3)",str(self.AstraLengths[2])) 
            self.setFile.disable("Q_type(1)")
            self.setFile.disable("Q_type(2)")
            self.setFile.disable("Q_type(3)")
            self.setFile.enable("Q_grad(1)")
            self.setFile.enable("Q_grad(2)")
            self.setFile.enable("Q_grad(3)")
        elif switcher == 1:  #quadrupole fields with ideal gradients
            self.setFile.changeInputData("Lquad", "T")
            self.setFile.changeInputData("LEField", "F") 
            self.setFile.changeInputData("Q_bore(1)", str(self.bores[0]))
            self.setFile.changeInputData("Q_bore(2)", str(self.bores[1]))
            self.setFile.changeInputData("Q_bore(3)", str(self.bores[2]))
            self.setFile.changeInputData("Q_length(1)",str(self.AstraLengths[0]))
            self.setFile.changeInputData("Q_length(2)",str(self.AstraLengths[1]))
            self.setFile.changeInputData("Q_length(3)",str(self.AstraLengths[2])) 
            self.setFile.disable("Q_type(1)")
            self.setFile.disable("Q_type(2)")
            self.setFile.disable("Q_type(3)")
            self.setFile.enable("Q_grad(1)")
            self.setFile.enable("Q_grad(2)")
            self.setFile.enable("Q_grad(3)")
        elif switcher == 2:  #quadrupole fields with custom gradients
            self.setFile.changeInputData("Lquad", "T")
            self.setFile.changeInputData("LEField", "F") 
            self.setFile.changeInputData("Q_bore(1)", str(self.bores[0]))
            self.setFile.changeInputData("Q_bore(2)", str(self.bores[1]))
            self.setFile.changeInputData("Q_bore(3)", str(self.bores[2]))
            self.setFile.changeInputData("Q_type(1)", "'3Dcavity1data.dat'")
            self.setFile.changeInputData("Q_type(2)", "'3Dcavity2data.dat'")
            self.setFile.changeInputData("Q_type(3)", "'3Dcavity3data.dat'")
            self.setFile.disable("Q_grad(1)")
            self.setFile.disable("Q_grad(2)")
            self.setFile.disable("Q_grad(3)")
            self.setFile.changeInputData("Q_length(1)",str(self.FPlengths[0]))
            self.setFile.changeInputData("Q_length(2)",str(self.FPlengths[1]))
            self.setFile.changeInputData("Q_length(3)",str(self.FPlengths[2]))   
            self.setFile.disable("Q_length(1)")
            self.setFile.disable("Q_length(2)")
            self.setFile.disable("Q_length(3)")
        elif switcher == 3: #field maps in cavities
            self.setFile.changeInputData("Lquad", "F")
            self.setFile.changeInputData("LEField", "T")
        else:
            print("Wrong input, only 0 through 3: 0 = top hat shaped fields, 1 = Astra generated quadrupole magnets with fringe fields, 2 = field profiles of gradient for measured quadrupoles, 3 = field maps of the measured magnets.")
            return False
        return True


    def changeMom(self, xAngle, yAngle, pz, xoff, yoff): 
        try:
            testData = ""

            #change longitudinal momentum for files test0.ini through test4.ini and test.ini
            for name in self.nameOfFiles:
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
            self.setFile.changeInputData("Ref_Ekin", str(pz))
            
            #uncomment once beam is being used!!!
            #subprocess.run("./generator " + fileName + " > output.txt" , shell=True,check=True,executable='/bin/bash' )
            #print(f"Successfully changed momentum to files and ran a generation of particles saved to '{fileName}'.")

        except FileNotFoundError:
            print("One of the files when changing initial offsets and momenta was not found.")
            return False
        except Exception as e:
            print(f"An error occurred when trying to change longitudinal momentum: {e}")
            return False
        return True


        
    def changePositions(self,D1,D2,D3, D4=None):

        if self.setFile.readOption('LEField') == self.setFile.readOption('Lquad'):
            print(f"Something is wrong, quadrupole namelist and cavity namelist are both {readOption('Lquad')}. Leaving.")
            return 1
        elif self.setFile.readOption('LEField') == 'T' or not self.setFile.checkOption("Q_grad(1)"):
            self.lengthQ1 = self.FPlengths[0]
            self.lengthQ2 = self.FPlengths[1]
            self.lengthQ3 = self.FPlengths[2]
        else:
            self.lengthQ1 = self.AstraLengths[0]
            self.lengthQ2 = self.AstraLengths[1]
            self.lengthQ3 = self.AstraLengths[2]

        if D4 != None:
            self.setupLength = D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3 + D4
            self.setFile.changeInputData("ZSTOP",str(math.ceil(self.setupLength*10)/10 ) )
        else:
            self.setupLength = math.ceil(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3)

        #changing the positions of apertures
        ap1 = str(D1) + " " + str(self.bores[0]*1E+3/2) + "\n" + str(D1 + self.lengthQ1) + " " + str(self.bores[0]*1E+3/2)
        with open("aperture1.dat", "w") as file:
            file.write(ap1)
        ap2 = str(D1 + self.lengthQ1 + D2) + " " + str(self.bores[1]*1E+3/2) + "\n" + str(D1 + self.lengthQ1 + D2 + self.lengthQ2) + " " + str(self.bores[1]*1E+3/2)
        with open("aperture2.dat", "w") as file:
            file.write(ap2)
        ap3 = str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3) + " " + str(self.bores[2]*1E+3/2) + "\n" + str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3) + " " + str(self.bores[2]*1E+3/2)
        with open("aperture3.dat", "w") as file:
            file.write(ap3)


        self.setFile.changeInputData("Q_pos(1)",str(D1 + self.lengthQ1/2) )
        self.setFile.changeInputData("Q_pos(2)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2/2) )
        self.setFile.changeInputData("Q_pos(3)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2) )

        
        #changing the positions of cavities
        self.setFile.changeInputData("C_pos(1)",str(D1))
        self.setFile.changeInputData("C_pos(2)", str(D1 + self.lengthQ1 + D2))
        self.setFile.changeInputData("C_pos(3)", str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3))    

        
        return [D1 + self.lengthQ1/2, D1 + self.lengthQ1 + D2 + self.lengthQ2/2 ,D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2,  D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3 + D4]

    def changePositionsHardEnd(self,D1,D2,D3, endOfSetup):
    	#this function changes positions according to the input arguments, but it also
    	#takes the argument of end of setup. D4 is computed accordingly.

        if self.setFile.readOption('LEField') == self.setFile.readOption('Lquad'):
            print(f"Something is wrong, quadrupole namelist and cavity namelist are both {readOption('Lquad')}. Leaving.")
            return 1
        elif self.setFile.readOption('LEField') == 'T' or not self.setFile.checkOption("Q_grad(1)"):
            self.lengthQ1 = self.FPlengths[0]
            self.lengthQ2 = self.FPlengths[1]
            self.lengthQ3 = self.FPlengths[2]
        else:
            self.lengthQ1 = self.AstraLengths[0]
            self.lengthQ2 = self.AstraLengths[1]
            self.lengthQ3 = self.AstraLengths[2]

        D4 = endOfSetup -( D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3)
        if D4 < 0:
        	print(f"The entire setup is longer than the set length. Skipping.")
        	return False


        #changing the positions of apertures
        ap1 = str(D1) + " " + str(self.bores[0]*1E+3/2) + "\n" + str(D1 + self.lengthQ1) + " " + str(self.bores[0]*1E+3/2)
        with open("aperture1.dat", "w") as file:
            file.write(ap1)
        ap2 = str(D1 + self.lengthQ1 + D2) + " " + str(self.bores[1]*1E+3/2) + "\n" + str(D1 + self.lengthQ1 + D2 + self.lengthQ2) + " " + str(self.bores[1]*1E+3/2)
        with open("aperture2.dat", "w") as file:
            file.write(ap2)
        ap3 = str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3) + " " + str(self.bores[2]*1E+3/2) + "\n" + str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3) + " " + str(self.bores[2]*1E+3/2)
        with open("aperture3.dat", "w") as file:
            file.write(ap3)


        self.setFile.changeInputData("Q_pos(1)",str(D1 + self.lengthQ1/2) )
        self.setFile.changeInputData("Q_pos(2)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2/2) )
        self.setFile.changeInputData("Q_pos(3)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2) )

        
        #changing the positions of cavities
        self.setFile.changeInputData("C_pos(1)",str(D1))
        self.setFile.changeInputData("C_pos(2)", str(D1 + self.lengthQ1 + D2))
        self.setFile.changeInputData("C_pos(3)", str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3))    

        
        return [D1 + self.lengthQ1/2, D1 + self.lengthQ1 + D2 + self.lengthQ2/2 ,D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2,  endOfSetup]




    def runCommand(self,cmd):
        self.process.stdin.write(cmd + '\n') 
        self.process.stdin.flush()
        return True

    def isFileOpen(filepath):
        result = subprocess.run(['lsof', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(result.stdout)

    def isRef0Straight(px, py):
        #function which checks if 0. ref particle did not move
        if px == 0 and py == 0:
            return True
        else:
            return False

    def loadData(arg, fillnum):
    #open and load data 
    #data structure for .ref files
    #z [m], t [ns], pz [MeV/c], dE/dz [MeV/c], Larmor Angle [rad], x off [mm], y off [mm], px [eV/c], py [eV/c]
        data = []
        fillNumber = "00" + str(fillnum)
        #assuming setup length
        with open(fileName + "." + arg + "." + fillNumber,"r") as file:
            for line in file:
                lineSplitted = line.split()
                data.append([float(num) for num in lineSplitted])

        return data
        

    def parallelFocusing(data):
        #parallel-parallel focusing: x'**2 + y'**2  
        return ( (data[1][3]*1e+3/data[1][5])**2 + (data[2][4]*1e+3/data[2][5])**2 )

    def pointFocusing(data):
        #to point-point focusing: x**2 + y**2
        return ( (data[1][0]*1e+3)**2 + (data[2][1]*1e+3)**2 )

    def xLineFocusing(data):
        #to point-parallel focusing: x'**2 + y**2
        return ( (data[1][3]*1e+3/data[1][5])**2 +  (data[2][1]*1e+3)**2 )

    def yLineFocusing(data):
        #to parallel-point focusing: x**2 + y'**2
        return ( (data[2][4]*1e+3/data[2][5])**2 +  (data[1][0]*1e+3)**2 )


    def calculatePercentage(self,acceptance, xAng_sig , yAng_sig ):
        #this function calculates the percentage of particles that pass the setup for a certain x and y gaussian initial spread   
        
        xLost = 2*sc.stats.norm(loc = 0 , scale = xAng_sig).cdf(-acceptance[0])
        yLost = 2*sc.stats.norm(loc = 0 , scale = yAng_sig).cdf(-acceptance[1])
        
        xPassed = 1-xLost
        yPassed = 1-yLost

        passed = xPassed * yPassed

        return passed    

    def checkAngleAcceptance(self, D1,D2,D3,D4, momZ, xAng = 1, yAng = 1):
        
        #obtain z positions of quads
        Qpos = self.changePositions(D1, D2, D3, D4)
        
        #change momentum 
        self.changeMom(self.sig_xAngle, self.sig_yAngle, momZ, -1, -1)

        #run reference particles and get data
        data = self.runRef(D1, D2, D3,D4, momZ, True)

        if data == 1:
            print(f"Something is wrong in runRef, leaving...")
            return False
        
        
        Q1_start = Qpos[0] - self.lengthQ1/2
        Q1_end = Qpos[0] + self.lengthQ1/2

        Q2_start = Qpos[1] - self.lengthQ2/2
        Q2_end = Qpos[1] + self.lengthQ2/2

        Q3_start = Qpos[2] - self.lengthQ3/2
        Q3_end = Qpos[2] + self.lengthQ3/2

        #variables where max values will be saved
        maxOffsetX = [0,0,0]
        maxOffsetY = [0,0,0]
        maxOffsetXzpos = [0,0,0]
        maxOffsetYzpos = [0,0,0]
        
        #check x acceptance
        for line in data[1]:
            #check Q1
            if line[0] > Q1_start and line[0]< Q1_end:
                if math.fabs(line[5]) > maxOffsetX[0]:
                    maxOffsetX[0] = math.fabs(line[5])
                    maxOffsetXzpos[0] = line[0]
                    
            #check Q2
            if line[0] > Q2_start and line[0]< Q2_end:
                if math.fabs(line[5]) > maxOffsetX[1]:
                    maxOffsetX[1] = math.fabs(line[5])
                    maxOffsetXzpos[1] = line[0]

            #check Q3
            if line[0] > Q3_start and line[0]< Q3_end:
                if math.fabs(line[5]) > maxOffsetX[2]:
                    maxOffsetX[2] = math.fabs(line[5])
                    maxOffsetXzpos[2] = line[0]

        #check y acceptance
        for line in data[2]:
            #check Q1
            if line[0] > Q1_start and line[0]< Q1_end:                
                if math.fabs(line[6]) > maxOffsetY[0]:
                    maxOffsetY[0] = math.fabs(line[6])
                    maxOffsetYzpos[0] = line[0]
                    
            #check Q2
            if line[0] > Q2_start and line[0]< Q2_end:
                if math.fabs(line[6]) > maxOffsetY[1]:
                    maxOffsetY[1] = math.fabs(line[6])
                    maxOffsetYzpos[1] = line[0]

            #check Q3
            if line[0] > Q3_start and line[0]< Q3_end:
                if math.fabs(line[6]) > maxOffsetY[2]:
                    maxOffsetY[2] = math.fabs(line[6])
                    maxOffsetYzpos[2] = line[0]


        #angular acceptance separately for x and y and for quads
        maxValsX = [ (self.sig_xAngle*self.bores[0]*1e+3)/(2*maxOffsetX[0]), (self.sig_xAngle*self.bores[1]*1e+3)/(2*maxOffsetX[1]), (self.sig_xAngle*self.bores[2]*1e+3)/(2*maxOffsetX[2])  ]
        maxValsY = [ (self.sig_yAngle*self.bores[0]*1e+3)/(2*maxOffsetY[0]), (self.sig_yAngle*self.bores[1]*1e+3)/(2*maxOffsetY[1]), (self.sig_yAngle*self.bores[2]*1e+3)/(2*maxOffsetY[2])  ]

        #get the minimal value
        self.xAngularAcceptance = min(maxValsX)
        self.yAngularAcceptance = min(maxValsY)
        
        percentagePassed = calculatePercentage([self.xAngularAcceptance, self.yAngularAcceptance],xAng, yAng) #possible to add x,y offsets
        
        
        #get the beam size for this sigma x and y angle spread
        changeMom(xAng, yAng, momZ, -1, -1)
        data = runRef(D1, D2, D3,D4, momZ, True)
        self.xBeamSize = data[1][-1][5]
        self.yBeamSize = data[2][-1][6]
        
        return [self.xAngularAcceptance, self.yAngularAcceptance, xAng, yAng, self.xBeamSize, self.yBeamSize, percentagePassed]



    def runRef(self, D1, D2, D3, D4,momZ, moreData):
        #this function runs Astra with 5 different reference particles for specific D1,D2,D3

        self.changePositions(D1,D2,D3, D4)
        self.changeMom(-1, -1, momZ, -1, -1)


        #if moreData, then provide tracking for each of the reference particles and return it for plotting
        if moreData:
            outputMoreData = []
            for i in range(len(self.nameOfFiles)):
                self.setFile.changeInputData("Distribution", self.nameOfFiles[i] )
                self.setFile.changeInputData("RUN", str(i+1))

                self.process.stdin.write("./Astra " + self.fileName + "\n")
                self.process.stdin.flush()
        
                while True:
                    line = self.process.stdout.readline()
                    if 'Goodbye' in line:
                        break

                currentData = self.loadData("ref", str(i+1))
                outputMoreData.append(currentData)
            
        else:
            inputDataName = ["test1.ini", "test2.ini"]
            outputMoreData = [[0,0,0,0,0,0]]
            for i in range(len(inputDataName)):
                changeInputData("Distribution", inputDataName[i] )
                changeInputData("RUN", str(i+1))

                self.process.stdin.write("./Astra " + fileName + "\n")
                self.process.stdin.flush()
        
                while True:
                    line = self.process.stdout.readline()
                    if 'Goodbye' in line:
                        break
                
                currentData = self.loadData("ref", i+1)
                
                #condition for 0. ref particle-> it cannot move
                if i == 0 and not isRef0Straight(currentData[-1][7], currentData[-1][8]):
                    print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
                    return 1
                #condition to check if the particle came all the way to the end

                bestLine = []
                closest = 0.1
                for j in range(len(currentData)):
                    dist = math.fabs(currentData[j][0] - self.setupLength)
                    if dist < closest:
                        bestLine = list(currentData[j])
                        closest = float(dist)

                if closest >0.1:
                    print(f"Reference particle {i} did not get to the end of setup.")
                outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )


        self.process.stdin.write("rm parallelBeam.ref.00*\n")
        self.process.stdin.flush()

        return outputMoreData


    def runRefHardEnd(self, D1, D2, D3,setupHardEnd,momZ, moreData):
        #this function runs Astra with 5 different reference particles for specific D1,D2,D3

        if self.changePositionsHardEnd(D1,D2,D3,setupHardEnd) == False:
        	return 1

        self.changeMom(-1, -1, momZ, -1, -1)


        #if moreData, then provide tracking for each of the reference particles and return it for plotting
        if moreData:
            outputMoreData = []
            for i in range(len(self.nameOfFiles)):
                self.setFile.changeInputData("Distribution", self.nameOfFiles[i] )
                self.setFile.changeInputData("RUN", str(i+1))

                self.process.stdin.write("./Astra " + self.fileName + "\n")
                self.process.stdin.flush()
        
                while True:
                    line = self.process.stdout.readline()
                    if 'Goodbye' in line:
                        break

                currentData = self.loadData("ref", str(i+1))
                outputMoreData.append(currentData)
            
        else:
            inputDataName = ["test1.ini", "test2.ini"]
            outputMoreData = [[0,0,0,0,0,0]]
            for i in range(len(inputDataName)):
                changeInputData("Distribution", inputDataName[i] )
                changeInputData("RUN", str(i+1))

                self.process.stdin.write("./Astra " + fileName + "\n")
                self.process.stdin.flush()
        
                while True:
                    line = self.process.stdout.readline()
                    if 'Goodbye' in line:
                        break
                
                currentData = self.loadData("ref", i+1)
                
                #condition for 0. ref particle-> it cannot move
                if i == 0 and not isRef0Straight(currentData[-1][7], currentData[-1][8]):
                    print(f"Reference 0 particle with 0 offset and 0 angle moved in setup with D1 = '{D1}', D2 = '{D2}' and D3 = '{D3}'.")
                    return 1
                #condition to check if the particle came all the way to the end

                bestLine = []
                closest = 0.1
                for j in range(len(currentData)):
                    dist = math.fabs(currentData[j][0] - self.setupLength)
                    if dist < closest:
                        bestLine = list(currentData[j])
                        closest = float(dist)

                if closest >0.1:
                    print(f"Reference particle {i} did not get to the end of setup.")
                outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )


        self.process.stdin.write("rm parallelBeam.ref.00*\n")
        self.process.stdin.flush()

        return outputMoreData




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


    def plotRefXY(self,D1, D2, D3,D4, mom, title = None, tag = None):

        #print(f"Running best setup again to get full data.")
        dataBest = self.runRef(D1, D2, D3,D4, mom, True)

        data0 = self.separateDataXYZ(dataBest[0])
        data3 = self.separateDataXYZ(dataBest[1])
        data4 = self.separateDataXYZ(dataBest[2])


        plt.plot([positions[-1],positions[-1]], [-0.5,0.5], color='black')

        plt.plot(data0[2], data0[0], label='0 offset, initial 0 angle', color='blue')
        plt.plot(data3[2], data3[0], label='x offset, initial x angle', color='red')
        plt.plot(data3[2], data3[1], label='y offset, initial x angle', color='yellow')
        plt.plot(data4[2], data4[0], label='x offset, initial y angle', color='green')
        plt.plot(data4[2], data4[1], label='y offset, initial y angle', color='purple')

        if D4 != None:
            plt.plot([self.setupLength, self.setupLength], [-0.5,0.5], color='black')
        
        plt.legend()

        plt.xlabel("z [m]")
        plt.ylabel("offset [mm]")

        if title != None:
            plt.title(title)

        if tag != None:
            plt.savefig(tag + ".png", format="png", dpi=300)
        
        plt.show()
        