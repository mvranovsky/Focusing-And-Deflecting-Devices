#!/usr/bin/python3

import numpy as np
import os
import re
import yaml
import sys
import subprocess
import math


fileName = "Example"
fillNumber = "001"

def driftMatrix(length):
	x = np.array([1, length, 0, 0], [0,1,0,0], [0,0,1,length], [0,0,0,1] )
	return x


def quadrupoleMatrix(K, length):
	x = np.array([math.cos(math.sqrt(K)*length), math.cos(math.sqrt(K)*length)/K, ])
