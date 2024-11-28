#!/bin/python3

from AstraWrapper.Generator import Generator


if __name__ == "__main__":

	gen = Generator("novelApproach.in")


	#gen.generateGradProfile(0.2, 0.8, gradAtStartP = 150, gradAtEndP = 200, showPlot = True)
	#print(gen.integrateGradProfile())


	#gen.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94)

	# generate field maps for permanent triplet
	gen.generateFieldMap(0.036, 0.777, gradAtStartP=222, gradAtEndP=222, fileOutputName='quad1', nFMPoints = 40)
	gen.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94, fileOutputName='quad2', nFMPoints = 40)
	gen.generateFieldMap(0.1, 0.855, gradAtStartP=57, gradAtEndP=57, fileOutputName='quad3', nFMPoints = 40)
	
