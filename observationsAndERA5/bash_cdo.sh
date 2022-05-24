#! /bin/bash
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=qnicolas@berkeley.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --constraint=haswell
#SBATCH --qos=debug
#SBATCH --output=R-%x.out

bash era5dailymean.sh

