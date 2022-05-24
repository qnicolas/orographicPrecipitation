#!/bin/bash
#

indirroot="/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/"
fileroot="e5.oper.an.pl.128_133_q.ll025sc."
outdirroot="/global/cscratch1/sd/qnicolas/processedData/era5dailyMeans/"


module load cdo
module load cray-netcdf
export HDF5_USE_FILE_LOCKING=FALSE

#for year in {1998..1998}
#do
#for monthdir in $indirroot$year*
for monthdir in $indirroot"199807"
  do
  for fullfile in $monthdir/$fileroot*.nc
    do
      #echo “$fullfile”
      filename=$(basename -- "$fullfile")
      echo "$filename"
      #echo "cdo daymean $fullfile ${outdirroot}dayMean.${filename}"
      cdo daymean $fullfile ${outdirroot}dayMean.${filename}
      #cdo daymean e5.oper.an.sfc.128_167_2t.ll025sc.$year010100_2001013123.nc ~/scratch/data/era5/e5.oper.an.sfc.128_167_2t.ll025sc.dailymean.2001010100_2001013123.nc
    done
  done
#done
