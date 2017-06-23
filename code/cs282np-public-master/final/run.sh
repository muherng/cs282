#!/bin/bash
#BATCH -J single_fouriernetwork
#SBATCH -o test7.out
#SBATCH -e test7.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 0-40:00
#SBATCH -p serial_requeue
#SBATCH --mem=32000
module load python
python run_tests.py
