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

mpirun -np 1 -x OMP_NUM_THREADS=16 ${EXEC} heat_transport.test.in heat_transport.test.out

#cd Data
#ln -s ../Weights/* .

#diff -r --brief ../target .
cd
cd SpectralBTE
module load python
python check_diff.py /home/hkim22/SpectralBTE/build/tests/heat_transport

exit $?
