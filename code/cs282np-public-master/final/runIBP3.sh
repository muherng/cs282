#!/bin/bash
#BATCH -J single_fouriernetwork
#SBATCH -o qIBP3_%a.out
#SBATCH -e qIBP3_%a.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 0-40:00
#SBATCH -p serial_requeue
#SBATCH --mem=32000
module load python
python -W ignore real_tests.py IBP 3
