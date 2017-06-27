import argparse
import os
import sys
args = sys.argv
print(args)
number_runs = args[1]
alg_type = ['paintbox']
param = [3,5]
for alg in alg_type:
    for par in param:
        fo = open("run" + alg + str(par) + ".sh", "w")
        fo.write("#!/bin/bash\n")
        fo.write("#BATCH -J single_fouriernetwork\n")
        fo.write("#SBATCH -o vr" + alg + str(par) + "_%a.out\n")
        fo.write("#SBATCH -e vr" + alg +str(par)+ "_%a.err\n")
        fo.write("#SBATCH -N 1\n")
        fo.write("#SBATCH -n 32\n")
        fo.write("#SBATCH -t 0-40:00\n")
        fo.write("#SBATCH -p serial_requeue\n")
        fo.write("#SBATCH --mem=32000\n")
        fo.write("module load python\n")
        fo.write("python -W ignore real_tests.py " + alg + " " + str(par) + "\n")
        fo.close()
        os.system("sbatch --array=1-" + number_runs + " run" + alg + str(par) + ".sh")
