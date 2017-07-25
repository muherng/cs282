args = sys.argv
print(args)
number_runs = args[1]
param = [.003,.001,.0003]
held = [.7]
#path = '/n/regal/doshi-velez_lab/mamasood/NMF_Data/'
files = ['SVD_recon20NG.npz','SVD_reconCupriteS1_F224.npz','SVD_reconCupriteS1_R188.npz','SVD_reconfaces_CBCL.npz',
         'SVD_reconIndian_Pines.npz','SVD_reconjasperRidge2_F224_2.npz','SVD_reconjasperRidge2_R198.npz','SVD_reconKSC.npz',
         'SVD_reconSalinas.npz','SVD_reconsamson_1.npz','SVD_reconUrban_F210.npz','SVD_reconUrban_R162.npz']
for filename in files:
    if filename.endswith(".npz"): 
        name = filename[:-4]
        for par in param:
            for he in held:
                fo = open("run" + name + str(par) + str(he) + ".sh", "w")
                fo.write("#!/bin/bash\n")
                fo.write("#BATCH -J single_fouriernetwork\n")
                fo.write("#SBATCH -o tv" + name + str(par) + str(he) + "_%a.out\n")
                fo.write("#SBATCH -e tv" + name +str(par)+ str(he) + "_%a.err\n")
                fo.write("#SBATCH -N 1\n")
                fo.write("#SBATCH -n 1\n")
                fo.write("#SBATCH -t 0-40:00\n")
                fo.write("#SBATCH -p doshi-velez\n")
                fo.write("#SBATCH --mem=200\n")
                fo.write("module load python\n")
                fo.write("python -W ignore d3_tests.py " + name + " " + str(par) + " " + str(he) + "\n")
                fo.close()
                os.system("sbatch --array=1-" + number_runs + " run" + name + str(par) + str(he) + ".sh")