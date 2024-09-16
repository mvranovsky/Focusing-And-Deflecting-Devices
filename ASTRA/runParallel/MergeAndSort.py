#!/bin/python3

import sys
import pandas as pd
import os

if __name__ == "__main__":

	args = sys.argv
	args.pop(0)

	if len(args) != 1:
		print("Incorrect number of arguments. Leaving.")
		sys.exit(1)
		

	BASE_DIR = os.getcwd()
	BASE_DIR += "/" + str(args[0])


	dirs = os.listdir(BASE_DIR) 

	results = ""
	errors = ""
	table = []

	for dir in dirs:
		loc = BASE_DIR + "/" + dir
		with open(loc + "/results.txt", "r") as file:
			res = file.readlines()
			for line in res:
				results += line.replace("\n", "") + "\n"
		with open(loc + "/errors.txt", "r") as file:
			err = file.readlines()
			for line in err:
				errors += line.replace("\n", "") + "\n"

		df = pd.read_csv(loc + "/table.csv")
		table += df.values.tolist()

	#print(table)
	sortPz = pd.DataFrame( sorted(table, key=lambda row: row[2]) )
	sortD1 = pd.DataFrame( sorted(table, key=lambda row: row[0]) )


	sortPz.to_csv(BASE_DIR + "/tablePz.csv", index=False)
	sortD1.to_csv(BASE_DIR + "/tableD1.csv", index=False)

	with open(BASE_DIR + "/results.txt", "w") as file:
		file.write(results)

	if errors != "":
		with open(BASE_DIR + "/errors.txt", "w") as file:
			file.write(errors)


	sys.exit(0)