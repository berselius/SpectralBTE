#!/bin/bash

TEST=$1
SRC=$2
BIN=$3
EXEC=$4
PYTHON=$5

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

echo "TEST = ${TEST}"
echo "SRC = ${SRC}"
echo "BIN = ${BIN}"
echo "EXEC = ${EXEC}"
echo "PYTHON = ${PYTHON}"

mpirun -np 1 -x OMP_NUM_THREADS=16 ${EXEC} ${TEST}.test.in ${TEST}.test.out

# check the weights file without any tolerance, may need to be changed
cd "Weights"
weights=`ls`
cd ..
for w in ${weights}; do
  diff Weights/${w} target/${w}
  res=$?

  if [ ${res} -ne 0 ]; then
    echo "Weights file ${w} differs"
    exit 1
  fi
done

# check *.plt output files with python script (that compares numbers with
# tolerance)
${PYTHON} ${BIN}/check_diff.py ${BIN}
res=$?

if [ ${res} -ne 0 ]; then
  echo "check_diff exited with code ${res}"
  exit 1
fi

exit 0
