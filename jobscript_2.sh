#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=Multi_CLASS_frequency_test
#SBATCH --output=/home/la171705/GW_BG/out/MC_f_test_2.txt
#SBATCH --time=0-3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1

echo "$(date)"
cd /home/la171705/GW_BG/Multi_CLASS
echo "${USER}"
echo "OpenMP threads set to ${OMP_NUM_THREADS}"
stdbuf -o0 ./class multi_explanatory_batch.ini
echo "$?"
echo "FINISHED"
echo "$(date)"