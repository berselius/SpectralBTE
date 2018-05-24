#ifndef _INITIALIZE_H
#define _INITIALIZE_H

void allocate_hom(int N, double **v, double **zeta, double **f, double **Q);

void initialize_hom(int N, double L_v, double *v, double *zeta, double *f, double *Q, int initFlag, int isoFlag, double lambda, double *M);

#endif
