#!/bin/python3

from AstraWrapper.Generator import Generator


if __name__ == "__main__":

	gen = Generator("parallelBeam.in")

	#1. arg: name of the quadrupole
	#2. arg: length [m]
	#3. arg: radius at starting edge [m]
	#4. arg: radius at end [m]
	#5. arg: gradient at start [T/m]
	#6. arg: gradient at end [T/m]
	
	gen.generateGProfile("quadTry", 0.2, 0.01, 0.03, 100, 150)
