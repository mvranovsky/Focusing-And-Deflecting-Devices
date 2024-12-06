#!/bin/python3

from AstraWrapper import Generator


if __name__ == "__main__":

	gen = Generator("novelApproach.in")
	'''
	#1. arg: length in m
	#2. arg: Tip Field in T
	# then either define gradient at the beginning and end, radius will computed according to tip field
	# or define radius at the beginning and end, gradient will be computed according to the tip field

	# one can define field without fringe fields:
	gen.generateGradProfile(0.2, 0.8, gradAtStartP = 150, gradAtEndP = 200, showPlot = True, fieldType = 0)

	# or with fringe fields: 
	gen.generateGradProfile(0.2, 0.8, gradAtStartP = 150, gradAtEndP = 200, showPlot = True)
	
	# one can define an entire field map:
	gen.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94)

	# or add wobbles to the skew angle, magnetic centre x and y, or gradient:
	'''
	#gen.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94, gradWobbles=True, magCentreXWobbles = True, magCentreYWobbles = True, skewAngleWobbles=True)



	# generate field maps for permanent triplet
	gen.generateFieldMap(0.036, 0.777, grad1=222, grad2=222, fileOutputName='quad1', nFMPoints = 40)
	print( gen.integrateGradProfile() )
	#gen.generateFieldMap(0.12, 0.846, gradAtStartP=-94, gradAtEndP=-94, fileOutputName='quad2', nFMPoints = 40)
	#gen.generateFieldMap(0.1, 0.855, gradAtStartP=57, gradAtEndP=57, fileOutputName='quad3', nFMPoints = 40)
	
