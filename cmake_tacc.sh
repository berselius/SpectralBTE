#/bin/bash

module load gsl
module load fftw3
module load cmake

cmake -DFFTW_INCLUDES=$TACC_FFTW3_DIR/include -DFFTW_LIB=$TACC_FFTW3_DIR/lib/libfftw3.so -DFFTW_OMP_LIB=$TACC_FFTW3_DIR/lib/libfftw3_omp.so  $1
