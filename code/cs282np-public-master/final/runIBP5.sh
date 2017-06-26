#!/bin/bash
#BATCH -J single_fouriernetwork
#SBATCH -o qIBP5_%a.out
#SBATCH -e qIBP5_%a.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 0-40:00
#SBATCH -p serial_requeue
#SBATCH --mem=32000
module load python
python -W ignore real_tests.py IBP 5
