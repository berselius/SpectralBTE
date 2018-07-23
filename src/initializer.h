#ifndef _INITIALIZE_H
#define _INITIALIZE_H
#include "species.h"

void allocate_hom(int N, double **v, double **zeta, double ***f, double ***f1, double ***Q, int num_species);

void initialize_hom(int N, double L_v, double *v, double *zeta, double **f, int initFlag, species *mixture);

void dealloc_hom(double *v, double *zeta, double **f, double **Q);

void allocate_inhom(int N, int nX, double **v, double **zeta, double ****f, double ****f_conv, double ****f_1, double ***Q, int Ns);

void initialize_inhom(int N, int Ns, double L_v, double *v, double *zeta, double ***f, double ***f_conv, double ***f_1, species *mixture, int initFlag, int nX, double *xnodes, double *dxnodes, double dt, int *t, int order, int restart, char *inputfilename);

void dealloc_inhom(int nX, int order, double *v, double *zeta, double ***f, double ***f_conv, double ***f_1, double **Q);

void initialize_inhom_mpi(int N, int Ns, double L_v, double *v, double *zeta, double ***f, double ***f_conv, double ***f_1, species *mixture, int initFlag, int nX, double *xnodes, double *dxnodes, double dt, int *t, int order, int restart, char *inputfilename);
#endif
