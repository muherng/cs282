import argparse
import os
fo = open("run.sh", "w")

fo.write("#!/bin/bash\n")
fo.write("#BATCH -J single_fouriernetwork\n")
fo.write("#SBATCH -o test7.out\n")
fo.write("#SBATCH -e test7.err\n")
fo.write("#SBATCH -N 1\n")
fo.write("#SBATCH -n 32\n")
fo.write("#SBATCH -t 0-40:00\n")
fo.write("#SBATCH -p serial_requeue\n")
fo.write("#SBATCH --mem=32000\n")
fo.write("module load python\n")
fo.write("python run_tests.py\n")

fo.close()

os.system("sbatch run.sh")