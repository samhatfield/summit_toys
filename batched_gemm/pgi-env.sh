# PGI for ORNL SUMMIT system

module load cmake
module load pgi
module load cuda

#export PATH=/sw/summit/cuda/11.0.1/bin:$PATH
#export LD_LIBRARY_PATH=/sw/summit/cuda/11.0.1/lib64:$LD_LIBRARY_PATH
#
#export FFTW_PATH=$PROJWORK/cli131/$USER/pgi.summit.sp/RAPS18/flexbuild/external/pgi.summit/install
#export NETCDF_PATH=$PROJWORK/cli131/$USER/pgi.summit.sp/RAPS18/flexbuild/external/pgi.summit/install
#export HDF5_PATH=$PROJWORK/cli131/$USER/pgi.summit.sp/RAPS18/flexbuild/external/pgi.summit/install
#export CC=pgcc
#export CXX=pgc++
#export F77=pgfortran
#export FC=pgfortran
#
#pgfortran --version
#
#export RAPS_PATH=/gpfs/alpine/proj-shared/cli131/hatfield/pgi.summit.sp/RAPS18/flexbuild/external/pgi.summit/install/share
#export ECCODES_DEFINITION_PATH=$RAPS_PATH/eccodes/definitions
#export GRIB_DEFINITION_PATH=$RAPS_PATH/grib_api/definitions
#
#export MPL_METHOD=JP_NON_BLOCKING_STANDARD
#export MPL_MBX_SIZE=128000000
