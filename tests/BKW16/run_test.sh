#!/bin/bash

SRC=$1
BIN=$2
EXEC=$3

cd ${BIN}

if [ -d "input" ]; then
  rm -rf input
fi

if [ -d "target" ]; then
  rm -rf target
fi

if [ -d "Data" ]; then
  rm -rf Data
fi

if [ -d "Weights" ]; then
  rm -rf Weights
fi

cp -r ${SRC}/input .
cp -r ${SRC}/target .
mkdir Data
mkdir Weights

echo "SRC = ${SRC}"
echo "BIN = ${BIN}"
echo "EXEC = ${EXEC}"

module load openmpi/3.1.0-gcc_8.1.0

#mpirun -np 1 -x OMP_NUM_THREADS=1 valgrind --leak-check=yes --track-origins=yes ${EXEC} BKW16.test.in BKW16.test.out
mpirun -np 1 -x OMP_NUM_THREADS=16 --mca btl ^openib ${EXEC} BKW16.test.in BKW16.test.out

cd
cd SpectralBTE
module load python
python check_diff.py /build/tests/BKW16

exit $?
