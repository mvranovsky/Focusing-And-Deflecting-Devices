
#-----------------------------------------------------------------------------------------
#
#  class written by: Michal Vranovsky (github: mvranovsky, email: miso.vranovsky@gmail.com)
#
# 
#-----------------------------------------------------------------------------------------
import random
import math
import matplotlib.pyplot as plt

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


    def integrateG(self, z_val, G_val, switcher):
        #equidistant integration- just linear, because ASTRA linearly interpolates between data points

        if len(z_val) != len(G_val):
            print("Ranges of z and G are not equal.")
            return
        

        DeltaZ =( z_val[1] - z_val[0]) #mm

        sum1 = 0
        G_Tm = []
        for i in range(len(G_val)):
            #gradient in T/mm
            sum += G_val[i]*DeltaZ 
            G_Tm.append(G_val[i]*1000)

        
        
        plt.plot(z_val,G_Tm,color='blue',label='Gradient')
        plt.title('Gradient of the field [T/mm] ')
        plt.xlabel("z [mm]")
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        gradient = 0 
        length = 0
        Qbore = 0
        if switcher == 1:
            Qbore = 0.007
            gradient = 222 #T/m
            length = sum1/gradient
        elif switcher == 2:
            Qbore = 0.018
            gradient = -94
            length = sum1/gradient
        elif switcher == 3:
            Qbore = 0.03
            gradient = 57
            length = sum1/gradient


        fringeL = fringeFieldB(sum1, gradient, Qbore)
        

        print(f"The entire integrated magnetic field turns out to be {sum} T")
        print(f"For gradient {gradient} T/m, the effective length of the quadrupole magnet with top hat fields should be {length} m")
        print(f"For gradient {gradient} T/m, the effective length of the quadrupole magnet with fringe fields in Astra should be {fringeL} m")





    def radiusFunction(self, z, qLength, QboreRadiusStart, QboreRadiusEnd):
        return z*(QboreRadiusEnd- QboreRadiusStart)/qLength + QboreRadiusStart


    def gradFunction0(self, z, qLength, gradStart, gradEnd):
        return z*(gradEnd - gradStart)/qLength + gradStart


    def gradFunction1(self, z, qLength, gradStart, gradEnd, QboreRadiusStart, QboreRadiusEnd):
        grad = 0
        if z >= 0 and z <= qLength:
            grad = z*(gradEnd - gradStart )/qLength + gradStart
        elif z < 0:
            grad = gradStart
        elif z >qLength:
            grad = gradEnd
        else:
            print(f"What the fuck did i forget? z = {z}")

        fVal = grad/( (1+math.exp( -2*z/QboreRadiusStart ) )*( 1+math.exp( 2*(z - qLength )/QboreRadiusEnd ) ) )

        return fVal

    def generateGProfile(self, quadName, qLength, QboreRadiusStart, QboreRadiusEnd,gradAtStartP, gradAtEndP,fieldType = 1, nPoints = 100):
        #simple function, which generates a gradient profile for specified parameters such as bore diameter, quadrupole length
        #and the gradAtStartP, gradAtEndP. Based on this, one can introduce 

        Zpos, ZposForR, gradVal, radius = [], [], [],[]

        if fieldType == 0:
            for i in range(nPoints):
                z = i*qLength/nPoints
                Zpos.append( z )
                ZposForR.append( z )
                gradVal.append( self.gradFunction0(z, qLength, gradAtStartP, gradAtEndP) )
                radius.append( self.radiusFunction(z, qLength, QboreRadiusStart, QboreRadiusEnd ) )

        elif fieldType == 1:

            #before the quad
            for i in range(math.ceil(nPoints/4)):
                z = -5*QboreRadiusStart + i*(5*QboreRadiusStart)/math.ceil(nPoints/4)
                Zpos.append(z)
                gradVal.append( self.gradFunction1(z, qLength, gradAtStartP, gradAtEndP, QboreRadiusStart, QboreRadiusEnd) )


            # inside the quad
            for i in range(nPoints + 1):
                z = i*qLength/nPoints
                Zpos.append( z )
                ZposForR.append( z )
                gradVal.append( self.gradFunction1(z, qLength, gradAtStartP, gradAtEndP, QboreRadiusStart, QboreRadiusEnd) )
                radius.append( self.radiusFunction(z, qLength, QboreRadiusStart, QboreRadiusEnd ) )

            #after the quad
            for i in range(math.ceil(nPoints/4)):
                z = qLength + (i+1)*(5*QboreRadiusEnd)/math.ceil(nPoints/4)
                Zpos.append(z)
                gradVal.append( self.gradFunction1(z, qLength, gradAtStartP, gradAtEndP, QboreRadiusStart, QboreRadiusEnd) )
        else:
            raise ValueError(f"fieldType {fieldType} is not implemented, only 0 for top hat field, 1 for astra generated gradients with fringe fields.")

        profileG = ''
        for i in range(len(Zpos) ):
            profileG += f"{Zpos[i]} {gradVal[i]}\n"

        apertureR = ''
        for i in range(len(ZposForR) ):
            apertureR += f"{ZposForR[i]} {radius[i]}\n"

        # save the radius to aperture/quadName
        if ".dat" in quadName:
            with open("aperture/" + quadName, "w") as file:
                file.write(apertureR)
            with open(quadName, "w") as file:
                file.write(profileG)
        else:
            with open("aperture/" + quadName + ".dat", "w") as file:
                file.write(apertureR)
            with open(quadName + ".dat", "w") as file:
                file.write(profileG)


        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        # Plot in each subplot
        axes[0].plot(Zpos, gradVal, color='blue')
        axes[0].set_xlabel("z [m]")
        axes[0].set_ylabel("gradient [T/m]")
        axes[0].set_title("gradient profile")
        axes[0].set_xlim(-6*QboreRadiusStart, qLength + 6*QboreRadiusEnd )


        axes[1].plot(ZposForR, radius, color='red')
        axes[1].set_xlabel("z [m]")
        axes[1].set_ylabel("radius [m]")
        axes[1].set_xlim(-6*QboreRadiusStart, qLength + 6*QboreRadiusEnd )

        plt.tight_layout()
        plt.show()

