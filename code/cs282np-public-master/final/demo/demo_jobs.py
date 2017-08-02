import argparse
import os
import sys
args = sys.argv
print(args)
number_runs = args[1]
algorithms = ['IBP','paintbox']
alg in algorithms:
    fo = open("run" + str(par) + str(he) + ".sh", "w")
    fo.write("#!/bin/bash\n")
    fo.write("#BATCH -J single_fouriernetwork\n")
    fo.write("#SBATCH -o dm" + alg + "_%a.out\n")
    fo.write("#SBATCH -e dm" + alg + "_%a.err\n")
    fo.write("#SBATCH -N 1\n")
    fo.write("#SBATCH -n 1\n")
    fo.write("#SBATCH -t 0-40:00\n")
    fo.write("#SBATCH -p doshi-velez\n")
    fo.write("#SBATCH --mem=200\n")
    fo.write("module load python\n")
    fo.write("python -W ignore demo_tests.py " + alg + "\n")
    fo.close()
    os.system("sbatch --array=1-" + number_runs + " run" + alg + ".sh")