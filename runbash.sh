#!/bin/bash
#SBATCH --nodes=4  number of nodes
#SBATCH --cpus-per-task=12  number of cores
#SBATCH --mem=32G  memory pool for all cores

#SBATCH --ntasks-per-node=1  one job per node
#SBATCH --gres=gpu:10  7 of the 21 paritions
#SBATCH --partition=gpu

#SBATCH -o slurm.%j.out  STDOUT
#SBATCH -e slurm.%j.err  STDERR

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u17ds20@abdn.ac.uk 

module load miniconda3
source activate diss
srun python runscript.py --epochs=100 --save /home/u17ds20/sharedscratch/
