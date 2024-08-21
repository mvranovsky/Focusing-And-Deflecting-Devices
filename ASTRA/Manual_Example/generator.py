#!/usr/bin/python3

import random
import yaml
import math


filename = "hatShapeGrad.dat"

data = ""
nIntervals = 1000
lowRange = 0.2
topRange = 0.3
interval = (topRange - lowRange)/nIntervals


for i in range(nIntervals):
	num1 = lowRange + i*interval
	line = str(math.ceil(num1 * 10000)/10000) + "	" + str(2) + "\n"
	data += line

with open(filename, "w") as file:
	file.write(data)

print("all data saved to file " + filename + ".")

