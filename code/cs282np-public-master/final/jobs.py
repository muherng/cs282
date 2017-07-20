import argparse
import os
import sys
args = sys.argv
print(args)
number_runs = args[1]
alg_type = ['paintbox']
param = [.006,.009,.012,.015]
held = [.7,.8,.9]
for alg in alg_type:
    for par in param:
        for he in held:
            fo = open("run" + alg + str(par) + str(he) + ".sh", "w")
            fo.write("#!/bin/bash\n")
            fo.write("#BATCH -J single_fouriernetwork\n")
            fo.write("#SBATCH -o ko" + alg + str(par) + str(he) + "_%a.out\n")
            fo.write("#SBATCH -e ko" + alg +str(par)+ str(he) + "_%a.err\n")
            fo.write("#SBATCH -N 1\n")
            fo.write("#SBATCH -n 32\n")
            fo.write("#SBATCH -t 0-40:00\n")
            fo.write("#SBATCH -p doshi-velez\n")
            fo.write("#SBATCH --mem=32000\n")
            fo.write("module load python\n")
            fo.write("python -W ignore d2_tests.py " + alg + " " + str(par) + " " + str(he) + "\n")
            fo.close()
            os.system("sbatch --array=1-" + number_runs + " run" + alg + str(par) + str(he) + ".sh")
