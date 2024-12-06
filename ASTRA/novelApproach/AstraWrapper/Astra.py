
from AstraWrapper.SettingsFile import SettingsFile
import subprocess
import math
import matplotlib.pyplot as plt
import time
import scipy as sc

#--------------------------------------------------------------------------------------------------------------------------
#
#       author: Michal Vranovsky (github: mvranovsky, email: miso.vranovsky@gmail.com)
#
#       description: This is a class for working with triplet of magnets that are located at the labs of LLR
#                    there are several methods to this class to either run Astra with reference particles or 
#                    or an entire beam, to calculate angular acceptance, max. initial offset, beam ratio or 
#                    to plot the result. It can be run in 4 different modes, the first one is top hat shaped
#                    fields, then astra generated ideal gradient with fringe fields, then are actual results
#                    from fits of measurements of the actual quadrupole magnets. The last one is using field 
#                    maps. Field maps can be generated from Generator class.
#
#--------------------------------------------------------------------------------------------------------------------------


class Astra:

    # quad constants
    nameOfFiles = ["test0.ini", "test1.ini", "test2.ini", "test3.ini", "test4.ini"]
    AstraLengths = [0.03619, 0.12429, 0.09596] #these values were computed so that integrated gradient matches the fits from field profiles
    FPlengths = [0.035, 0.120, 0.105]
    bores = [0.007, 0.018, 0.030]    

    # constants regarding reference particles
    # when using beam, these values are sigmas
    # can be changed in changeMom()
    xAngle = 1      #mrad
    yAngle = 1      #mrad
    xoff = 5.0E-3   #mm
    yoff = 5.0E-3   #mm
    
    # constants regarding beam
    nParticles = "500"
    Ref_Ekin = 6E+8   #eV
    sig_z=0.1         #mm
    sig_Ekin=3E+3     #keV


    def __init__(self, settingsFile):

        #default constructor
        if isinstance(settingsFile, str ):
            self.setFile = SettingsFile(settingsFile)
            self.fileName = settingsFile 
        elif isinstance(settingsFile, SettingsFile ):
            self.setFile = settingsFile
            self.fileName = settingsFile.fileName
        else:
            raise ValueError(f"Could not initialize Astra class. Constructor expects either a name of the input file or class SettingsFile.")



    def aperture(self,yes):
        #whether to use aperture or not, meaning particles can get lost in the material of the quadrupoles
        if yes:
            self.setFile.changeInputData("LApert", "T")
            return True
        else:
            self.setFile.changeInputData("LApert","F")
            return False

    def quadType(self,switcher):
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
            raise ValueError("Wrong input, only 0 through 3: 0 = top hat shaped fields, 1 = Astra generated quadrupole magnets with fringe fields, 2 = field profiles of gradient for measured quadrupoles, 3 = field maps of the measured magnets.")
        return True


    def changeMom(self,pz, xAngle=-1, yAngle=-1, xoff=-1, yoff=-1,sig_Ekin =-1, sig_z = -1 ): 
        #function to change initial angle, Pz and offsets.
        #this function does not change a variable, if it is set to -1
        try:
            testData = ""

            #change longitudinal momentum for files test0.ini through test4.ini and test.ini
            for name in self.nameOfFiles:
                with open(name, "r") as file:
                    line = file.readlines()[0].split()

                #offset update
                if name == "test3.ini" and xoff != -1:
                    line[0] = str(xoff/1000)
                    self.setFile.changeInputData("sig_x", str(xoff/1000))
                    self.xoff = xoff                   
                if name == "test4.ini" and yoff != -1:
                    line[1] = str(yoff/1000)
                    self.setFile.changeInputData("sig_y", str(yoff/1000))
                    self.yoff = yoff

                #momentum update
                if name == "test1.ini":
                    if xAngle != -1:
                        line[3] = str(xAngle*pz*1e-3)
                        self.setFile.changeInputData("sig_px", str(xAngle*pz*1e-3))
                        self.xAngle = xAngle
                    else:  #have to recalculate initial px, because pz changed and want to keep angle the same
                        line[3] = str(self.xAngle*pz*1e-3)
                        self.setFile.changeInputData("sig_px", str(self.xAngle*pz*1e-3))

                if name == "test2.ini":
                    if yAngle != -1:
                        line[4] = str(yAngle*pz*1e-3)
                        self.setFile.changeInputData("sig_py", str(yAngle*pz*1e-3))
                        self.yAngle = yAngle
                    else:   #have to recalculate initial py, because pz changed and want to keep angle the same
                        line[4] = str(self.yAngle*pz*1e-3)
                        self.setFile.changeInputData("sig_py", str(self.yAngle*pz*1e-3)) 
                
                line[5] = str(pz)
                self.Ref_Ekin = pz*1e-6

                inputData = ""
                for num in line:
                    inputData += num + " "
                testData += inputData + "\n"
                with open(name, "w") as file:
                    file.write(inputData)

        
            with open("test.ini","w") as file:
                file.write(testData)
        
            # change distributions in longitudinal direction
            self.setFile.changeInputData("Ref_Ekin", str(pz*1e-6))
            if sig_Ekin != -1:
                self.sig_Ekin = sig_Ekin
                self.setFile.changeInputData("sig_Ekin", str(self.sig_Ekin))
            if sig_z != -1:
                self.sig_z = sig_z
                self.setFile.changeInputData("sig_z", str(self.sig_z))
            #uncomment once beam is being used!!!
            self.runGenerator()

        except FileNotFoundError:
            print("One of the files when changing initial offsets and momenta was not found.")
            return False
        except Exception as e:
            print(f"An error occurred when trying to change longitudinal momentum: {e}")
            return False
        return True


        
    def changePositions(self,D1,D2,D3, D4, hardEnd):
        #default function which changes the positions of quadrupole magnets according to input D1,D2,D3,D4
        #it also returns the positions of the magnets
        if self.setFile.readOption('LEField') == self.setFile.readOption('Lquad'):
            raise ValueError(f"Something is wrong, quadrupole namelist and cavity namelist are both {self.setFile.readOption('Lquad')}. Leaving.")
        elif self.setFile.readOption('LEField') == 'T' or not self.setFile.checkOption("Q_grad(1)"):
            self.lengthQ1 = self.FPlengths[0]
            self.lengthQ2 = self.FPlengths[1]
            self.lengthQ3 = self.FPlengths[2]
            self.setFile.changeInputData("Q_pos(1)",str(D1) )
            self.setFile.changeInputData("Q_pos(2)",str(D1 + self.lengthQ1 + D2) )
            self.setFile.changeInputData("Q_pos(3)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3) )
        else:
            self.lengthQ1 = self.AstraLengths[0]
            self.lengthQ2 = self.AstraLengths[1]
            self.lengthQ3 = self.AstraLengths[2]
            self.setFile.changeInputData("Q_pos(1)",str(D1 + self.lengthQ1/2) )
            self.setFile.changeInputData("Q_pos(2)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2/2) )
            self.setFile.changeInputData("Q_pos(3)",str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2) )


        if D4 != None and hardEnd != None:
            raise ValueError(f"Something is wrong, D4 and hardEnd are both set")
        elif D4 == None: #end is set, D4 is calculated
            D4 = hardEnd -( D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3)
            if D4 < self.bores[2]*3/2:
                raise ValueError(f"D4 value should be more than 3/2 of diameter of the last quadrupole, otherwise it is too close to the end point of measurement which could cause trouble.")
            self.setupLength = hardEnd
            self.setFile.changeInputData("ZSTOP",str(math.ceil(self.setupLength*10)/10 ) )
        elif hardEnd == None: #D4 is set, end of setup is calculated
            self.setupLength = D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3 + D4
            self.setFile.changeInputData("ZSTOP",str(math.ceil(self.setupLength*10)/10 ) )


        #changing the positions of apertures
        self.setFile.changeInputData("A_pos(1)", str(D1))
        self.setFile.changeInputData("A_pos(2)", str(D1 + self.lengthQ1 + D2))
        self.setFile.changeInputData("A_pos(3)", str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3))    

        #changing the positions of cavities
        self.setFile.changeInputData("C_pos(1)",str(D1))
        self.setFile.changeInputData("C_pos(2)", str(D1 + self.lengthQ1 + D2))
        self.setFile.changeInputData("C_pos(3)", str(D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3))    

        
        return [D1 + self.lengthQ1/2, D1 + self.lengthQ1 + D2 + self.lengthQ2/2 ,D1 + self.lengthQ1 + D2 + self.lengthQ2 + D3 + self.lengthQ3/2,  self.setupLength]

    def runAstra(self):
        cmd = ["./Astra", self.fileName + ".in"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def runGenerator(self):
        cmd = ["./generator", self.fileName + ".in" ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def runCommand(self,cmd):
        #simple function that runs the argument command
        if len(cmd) == 1:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result


    def isFileOpen(filepath):
        #checking whether a file is open or not, currently not used in the runRef()
        result = subprocess.run(['lsof', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(result.stdout)

    def isRef0Straight(self,px, py):
        #function which checks if 0. ref particle did not move
        if px == 0 and py == 0:
            return True
        else:
            return False

    def loadData(self,arg, fillnum = 1):
    #open and load data after an Astra run
    #data structure for .ref files
    #z [m], t [ns], pz [MeV/c], dE/dz [MeV/c], Larmor Angle [rad], x off [mm], y off [mm], px [eV/c], py [eV/c]
        data = []
        fillNumber = "00" + str(fillnum)
        #assuming setup length
        with open(self.fileName + "." + arg + ".00" + str(fillnum),"r") as file:
            for line in file:
                lineSplitted = line.split()
                data.append([float(num) for num in lineSplitted])

        return data
        

    def parallelFocusing(self,data):
        #parallel-parallel focusing: x'**2 + y'**2  
        return ( (data[1][3]*1e+3/data[1][5])**2 + (data[2][4]*1e+3/data[2][5])**2 )

    def pointFocusing(self, data):
        #to point-point focusing: x**2 + y**2
        return ( (data[1][0]*1e+3)**2 + (data[2][1]*1e+3)**2 )

    def xLineFocusing(self, data):
        #to point-parallel focusing: x'**2 + y**2
        return ( (data[1][3]*1e+3/data[1][5])**2 +  (data[2][1]*1e+3)**2 )

    def yLineFocusing(self, data):
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


    def checkAngleAcceptance(self, D1,D2,D3,D4, hardEnd, momZ):
        #long function which checks how close a ray gets close to a quadrupole
        #based on linearity rescales the distance to obtain the angular acceptance 

        Qpos = self.changePositions(D1, D2, D3, D4, hardEnd)
        
        #change momentum 
        self.changeMom(momZ)

        #run reference particles and get data
        data = self.runRef(D1, D2, D3,D4,hardEnd, momZ, True)

        
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
        maxValsX = [ (self.xAngle*self.bores[0]*1e+3)/(2*maxOffsetX[0]), (self.xAngle*self.bores[1]*1e+3)/(2*maxOffsetX[1]), (self.xAngle*self.bores[2]*1e+3)/(2*maxOffsetX[2])  ]
        maxValsY = [ (self.yAngle*self.bores[0]*1e+3)/(2*maxOffsetY[0]), (self.yAngle*self.bores[1]*1e+3)/(2*maxOffsetY[1]), (self.yAngle*self.bores[2]*1e+3)/(2*maxOffsetY[2])  ]

        #get the minimal value
        self.xAngularAcceptance = math.floor(min(maxValsX)*100)/100
        self.yAngularAcceptance = math.floor(min(maxValsY)*100)/100
                
        
        return [self.xAngularAcceptance, self.yAngularAcceptance]

    def initialOffsetLimit(self, D1,D2,D3,D4,hardEnd, momZ):
        #similar function to checkAngleAcceptance() only to find the maximal initial x and y offsets 
        #which still pass through the setup

        Qpos = self.changePositions(D1, D2, D3, D4, hardEnd)
        
        #change momentum 
        self.changeMom(momZ)

        #run reference particles and get data
        data = self.runRef(D1, D2, D3,D4,hardEnd, momZ, True)

        
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
        for line in data[3]:
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
        for line in data[4]:
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
        maxValsX = [ (self.xoff*self.bores[0])/(2*maxOffsetX[0]), (self.xoff*self.bores[1])/(2*maxOffsetX[1]), (self.xoff*self.bores[2])/(2*maxOffsetX[2])  ]
        maxValsY = [ (self.yoff*self.bores[0])/(2*maxOffsetY[0]), (self.yoff*self.bores[1])/(2*maxOffsetY[1]), (self.yoff*self.bores[2])/(2*maxOffsetY[2])  ]

        print(maxValsX)
        print(maxValsY)

        #get the minimal value
        self.xOffsetMax = math.floor(min(maxValsX)*100000)/100
        self.yOffsetMax = math.floor(min(maxValsY)*100000)/100
            
        
        return [self.xOffsetMax, self.yOffsetMax]



    def beamRatio(self, D1,D2,D3,D4, hardEnd, momZ):

        Qpos = self.changePositions(D1, D2, D3, D4, hardEnd)
        self.changeMom( momZ)

        data = self.runRef(D1, D2, D3,D4,hardEnd, momZ, False)
        
        xPos = data[1][0]
        yPos = data[2][1]

        return math.fabs(xPos)/math.fabs(yPos)

    def getClosest(self, currentData):

        bestLine = []
        closest = 0.1
        for j in range(len(currentData)):
            dist = math.fabs(currentData[j][0] - self.setupLength)
            if dist < closest:
                bestLine = list(currentData[j])
                closest = float(dist)

        if closest >0.1:
            raise ValueError(f"Reference particle {i} did not get to the end of setup.")

        return bestLine



    def runRef(self, D1, D2, D3, D4, hardEnd,momZ, moreData):
        #function which runs Astra with reference particles
        #if moreData is set to True, it runs 5 different reference particles, data on each one of them 
        #is returned (especially for plotting the trajectories in plotRefXY() )
        #if moreData is set to False, runs only 2 particles (right now) with initial angles and returns
        #output only at the end position of setup


        self.changePositions(D1,D2,D3, D4, hardEnd)
        self.changeMom(momZ)

        if moreData:
            outputMoreData = []
            for i in range(len(self.nameOfFiles)):
                self.setFile.changeInputData("Distribution", self.nameOfFiles[i] )
                self.setFile.changeInputData("RUN", str(i + 1))

                res = self.runAstra()

                if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
                    raise ValueError(f"Astra did not run properly in runRef() with moreData=True.")

                currentData = self.loadData("ref", str(i + 1))
                outputMoreData.append(currentData)
            
        else:
            inputDataName = ["test1.ini", "test2.ini"]
            outputMoreData = [[0,0,0,0,0,0]]
            for i in range(len(inputDataName)):
                self.setFile.changeInputData("Distribution", inputDataName[i] )
                self.setFile.changeInputData("RUN", str(i+ 1))

                res = self.runAstra()

                if not (res.stderr == '' or 'Goodbye' in res.stdout) or "ATTENTION: PROGRAM IS QUITTING  EARLY !" in res.stdout:
                    raise ValueError(f"Astra did not run properly in runRef() with moreData=True.")

                currentData = self.loadData("ref", str(i+1) )

                bestLine = self.getClosest(currentData)
                if bestLine == 1:
                    raise ValueError("Could not get close to the end screen in runRef() method, check it.")

                outputMoreData.append( [bestLine[5]*1e-3, bestLine[6]*1e-3, bestLine[0], bestLine[7], bestLine[8], bestLine[2]*1e+6] )


        return outputMoreData

    def getBeamInfo(self, runNum = 1):

        runNum = "00" + str(runNum)

        currentData = self.loadData("Xemit", runNum)

        index = 0
        closest = 0.1
        for j in range(len(currentData)):
            dist = math.fabs(currentData[j][0] - self.setupLength)
            if dist < closest:
                index = int(j)
                closest = float(dist)
        
        data[0] = self.loadData("Xemit", runNum)[index]
        data[1] = self.loadData("Yemit", runNum)[index]
        data[2] = self.loadData("Zemit", runNum)[index]
        return data

    def runBeam(self, D1, D2, D3, D4, hardEnd,momZ, moreData):
        #function which runs Astra with beam
        #if moreData is set to True, it runs 5 different reference particles, data on each one of them 
        #is returned (especially for plotting the trajectories in plotRefXY() )
        #if moreData is set to False, runs only 2 particles (right now) with initial angles and returns
        #output only at the end position of setup

        self.changePositions(D1,D2,D3, D4, hardEnd)
        self.changeMom( momZ)


        if moreData:
            outputMoreData = []
            self.setFile.changeInputData("Distribution", self.fileName + ".ini" )
            self.setFile.changeInputData("RUN", str(1))

            self.runAstra()
    
            if not (res.stderr == '' or 'Goodbye' in res.stdout):
                raise ValueError(f"Astra did not run properly in runBeam() with moreData=True.")

            outputMoreData.append(self.loadData("Xemit"))
            outputMoreData.append(self.loadData("Yemit"))
            outputMoreData.append(self.loadData("Zemit"))
            
            
        else:
            self.setFile.changeInputData("Distribution", self.fileName + ".ini" )
            self.setFile.changeInputData("RUN", str(i+1))

            self.runAstra()
    
            if not (res.stderr == '' or 'Goodbye' in res.stdout):
                raise ValueError(f"Astra did not run properly in runBeam() with moreData=True.")

            outputMoreData = self.getBeamInfo()            


        return outputMoreData


    def separateDataXYZ(self,data):
        #helper function for plotting to separate data
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


    def plotRefXY(self,D1, D2, D3,D4,hardEnd, mom, title = None, tag = None):
        #main plotting function to plot trajectories of reference particles

        plt.figure()

        #print(f"Running best setup again to get full data.")
        dataBest = self.runRef(D1, D2, D3,D4,hardEnd, mom, True)

        data0 = self.separateDataXYZ(dataBest[0])
        data1 = self.separateDataXYZ(dataBest[1])
        data2 = self.separateDataXYZ(dataBest[2])
        data3 = self.separateDataXYZ(dataBest[3])
        data4 = self.separateDataXYZ(dataBest[4])

        plt.plot(data0[2], data0[0], label='0 offset, initial 0 angle', color='black')
        plt.plot(data1[2], data1[0], label='x offset, initial x angle', color='red')
        plt.plot(data2[2], data2[1], label='y offset, initial y angle', color='purple')
        plt.plot(data3[2], data3[0], label='x offset, initial x offset', color='blue')
        plt.plot(data4[2], data4[1], label='y offset, initial y offset', color='green')

        if D4 != None:
            plt.plot([self.setupLength, self.setupLength], [-0.5,0.5], color='black')
        
        plt.legend()

        plt.xlabel("z [m]")
        plt.ylabel("offset [mm]")
        #plt.ylim(-0.2, 1.2)

        if title != None:
            plt.title(title)

        if tag != None:
            plt.savefig(tag + ".png", format="png", dpi=300)

        plt.show()

        
