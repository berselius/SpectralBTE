#ifndef _CONSERVE_H
#define _CONSERVE_H

#include "species.h"

void initialize_conservation(int nodes, double h_v, double *vel, species *mix, int num_spec, int fast);

void dealloc_conservation();

void factorMatrixIntoLU(int nElem, double det, int iError);

void solveWithCCt(int nElem, double *b);

void conserveAllMoments(double **inOut);

void createCCtAndPivot();

#endif
