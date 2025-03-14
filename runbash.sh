#!/bin/bash
#SBATCH --nodes=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:10
#SBATCH --partition=gpu

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=u17ds20@abdn.ac.uk 

module load miniconda3
source activate diss

# Run first script
srun python runscript.py --epochs=100 --save /home/u17ds20/sharedscratch/

# Run second script after first completes
srun python runscript2.py --epochs=100 --save /home/u17ds20/sharedscratch/
