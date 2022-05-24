#! /bin/bash
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=qnicolas@berkeley.edu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=R-%x.out
#SBATCH --constraint=haswell

cd ~/orographicPrecipitation/wrfProcessing/
module load python
source activate era5
echo done
python subset_test.py
