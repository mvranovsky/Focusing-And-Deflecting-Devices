# Systematic analysis

Results in this directory come from a large number of runs of astra and the minimization procedure. Systematically, D1 in ranges from around 1 cm to around 30 cm and Pz from 200 to 1000 MeV were ran, and the results for the specific Pz, or D1 are saved in this directory based on what kind of focusing it was. For parallel focusing, the minimization process first looked for a solution in less than 90 cm, if it could not be found, the range was expanded to 120 cm. For point focusing, the range was always set to 2 m, if the solution is not in the plots, it means that no solution was found. Beam size ratio for point point focusing is not relevant, due to relatively low condition on tolerance of the minimized function. Instead of the setup length for point focusing, which is always constant, there is D4 calculated according to the setup length and D1,D2,D3 and the sizes of the quads. The code can be found in function *tripletFocusing()* in file specialAssignment.py. 