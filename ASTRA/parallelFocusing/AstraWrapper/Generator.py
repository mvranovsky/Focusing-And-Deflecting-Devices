
#-----------------------------------------------------------------------------------------
#
#  class written by: Michal Vranovsky (github: mvranovsky, email: miso.vranovsky@gmail.com)
#
#  hasn't been tested!!
#-----------------------------------------------------------------------------------------
import random
import math

class Generator:

    charge = -1e-4      #macro charge
    particleIdx = 1     #electron
    statusFlag = 5      #status of a particle
    massElectronInEv = 5.1E+5  

    def __init__(self, inputFile):
        if isinstance(inputFile, str ):
            self.fileName = inputFile
        else:
            raise ValueError(f"Class Generator is expecting a string argument- the name of file, under which it will be generating input files for Astra.")

        
    def Gauss(self,sig, mu): #returns random number according to normal distribution
        return math.ceil(random.gauss(mu,sig))

    def Uniform(self,a, b, sig, mu): #returns random number uniformly between a and b
        if a != None and b != None:
            return math.ceil(random.uniform(a,b))
        else:
            return math.ceil(random.uniform(mu - math.sqrt(3)*sig, mu + math.sqrt(3)*sig ))


    def generatePointSource(self, nPart ,Pz, sig_Px = 0, sig_Py = 0,mu_Px = 0, mu_Py = 0, distPx = 'Gauss', distPy = 'Gauss', a_Px = None, b_Px = None, a_Py = None, b_Py = None, xOffset = 0, yOffset = 0):
        #function that randomly generates an electron beam originating from a point-like source with constant Pz for all particles(Ekin not constant)
        #transverse momenta are generated according to distribution defined by distPx, distPy with parameters mu_P, sig_P for gaussian or uniform, 
        #for uniform distribution one can use alternative form with a and b. All momentum args are expected to be in eV
        #the source point can have specific x and y positions, the input args are expected in m
        #possible types of distributions: 'Gauss', 'Uniform'

        if Pz > 100 and Pz <1000: #for MeV
            Pz = Pz*1000000
        elif Pz > 1E+8 and Pz < 1E+9:
            Pz = Pz
        else:
            raise ValueError("Expecting Pz in range (100, 1000) MeV.")


        output = ' 0 0 0 0 0 ' + str(Pz + self.massElectronInEv) + ' 0 ' + str(self.charge) + '   ' + str(self.particleIdx) + '   ' + str(self.statusFlag) + '\n'

        for i in range(nPart - 1):

            px, py = 0, 0
            if distPx == "Gauss" or distPx =='g' or distPx == 'G' or distPx == 'gauss':
                px = self.Gauss(sig_Px, mu_Px)
            elif distPx == 'Uniform' or distPx == 'U' or distPx == 'u' or distPx == 'uniform':
                px = self.Uniform(sig=sig_Px, mu=mu_Px, a=a_Px, b=b_Px)
            else:
                raise ValueError(f"For px distribution, method generatePoint() of class Generator is expecting a gaussian or uniform distribution.")

            if distPy == 'Gauss' or distPy == 'g' or distPy == 'G' or distPy == 'gauss':
                py = self.Gauss(sig_Py, mu_Py)
            elif distPy == 'Uniform' or distPy == 'u' or distPy == 'U' or distPy == 'uniform':
                py = self.Uniform(sig=sig_Py, mu=mu_Py, a=a_Py, b=b_Py )

            output += f" {xOffset} {yOffset} 0 {px} {py} 0 0 {self.charge}   {self.particleIdx}   {self.statusFlag}\n"
        
        with open(self.fileName + ".ini", 'w') as file:
            file.write(output)


    def generateSource(self, nPart ,Pz, sig_Px = 0, sig_Py = 0,mu_Px = 0, mu_Py = 0, distPx = 'Gauss', distPy = 'Gauss',a_Py = None, b_Py = None, a_Px = None, b_Px = None, distX = 'Gauss', distY = 'Gauss', sig_X = 0, sig_Y = 0 , a_X = None, b_X = None, a_Y = None, b_Y = None, mu_X = 0 ,mu_Y = 0 ):


        if Pz > 100 and Pz <1000: #for MeV
            Pz = Pz*1000000
        elif Pz > 1E+8 and Pz < 1E+9:
            Pz = Pz
        else:
            raise ValueError("Expecting Pz in range (100, 1000) MeV.")


        output = ' 0 0 0 0 0 ' + str(Pz + self.massElectronInEv) + ' 0 ' + str(self.charge) + '   ' + str(self.particleIdx) + '   ' + str(self.statusFlag) + '\n'

        for i in range(nPart - 1):

            px, py, x, y = 0, 0, 0, 0
            # Px
            if distPx == "Gauss" or distPx =='g' or distPx == 'G' or distPx == 'gauss':
                px = self.Gauss(sig_Px, mu_Px)
            elif distPx == 'Uniform' or distPx == 'U' or distPx == 'u' or distPx == 'uniform':
                px = self.Uniform(sig=sig_Px, mu=mu_Px, a=a_Px, b=b_Px)
            else:
                raise ValueError(f"For px distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")
            # Py
            if distPy == 'Gauss' or distPy == 'g' or distPy == 'G' or distPy == 'gauss':
                py = self.Gauss(sig_Py, mu_Py)
            elif distPy == 'Uniform' or distPy == 'u' or distPy == 'U' or distPy == 'uniform':
                py = self.Uniform(sig=sig_Py, mu=mu_Py, a=a_Py, b=b_Py )
            else:
                raise ValueError(f"For py distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")

            # X
            if distX == "Gauss" or distX == "G" or distX == "g" or distX == "gauss":
                x = self.Gauss(sig_X, mu_X)
            elif distX == "U" or distX == "u" or distX == "Uniform" or distX == "uniform":
                x = self.Uniform(sig=sig_X, mu=mu_X, a=a_X, b=b_X)
            else:
                raise ValueError(f"For x distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")

            # Y
            if distY == "Gauss" or distY == "G" or distY == "g" or distY == "gauss":
                y = self.Gauss(sig_Y, mu_Y)
            elif distY == "U" or distY == "u" or distY == "Uniform" or distY == "uniform":
                y = self.Uniform(sig=sig_Y, mu=mu_Y, a=a_Y, b=b_Y)
            else:
                raise ValueError(f"For y distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")


                output += f" {x} {y} {0} {px} {py} {0} {0} {self.charge}   {self.particleIdx}   {self.statusFlag}\n"
        
        with open(self.fileName + ".ini", 'w') as file:
            file.write(output)


