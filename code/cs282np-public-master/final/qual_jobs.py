import os
import sys
args = sys.argv
print(args)
number_runs = args[1]
param = [.001]
held = [.7]
#path = '/n/regal/doshi-velez_lab/mamasood/NMF_Data/'
for par in param:
    for he in held:
        fo = open("run" + str(par) + str(he) + ".sh", "w")
        fo.write("#!/bin/bash\n")
        fo.write("#BATCH -J single_fouriernetwork\n")
        fo.write("#SBATCH -o q1" + str(par) + str(he) + "_%a.out\n")
        fo.write("#SBATCH -e q1" + str(par)+ str(he) + "_%a.err\n")
        fo.write("#SBATCH -N 1\n")
        fo.write("#SBATCH -n 1\n")
        fo.write("#SBATCH -t 0-40:00\n")
        fo.write("#SBATCH -p doshi-velez\n")
        fo.write("#SBATCH --mem=100\n")
        fo.write("module load python\n")
        fo.write("python -W ignore qual_tests.py " + str(par) + " " + str(he) + "\n")
        fo.close()
        os.system("sbatch --array=1-" + number_runs + " run" + str(par) + str(he) + ".sh")