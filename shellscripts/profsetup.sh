module load openmpi

module load gsl

module load allinea-forge

module load gcc

module swap PrgEnv-cray PrgEnv-intel

make-profiler-libraries --platform=cray --lib-type=shared

export LD_LIBRARY_PATH=/global/homes/h/hcarrill/SpectralBTE/:$LD_LIBRARY_PATH
