#include "species.h"

void initialize_transport(int numV, int numX, double lv, double *xnodes,
                          double *dxnodes, double *vel, int IC, double dt,
                          double TWall_in, species *mix);

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double min(double in1, double in2);

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double max(double in1, double in2);

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double minmod(double in1, double in2, double in3);

// Computes first order upwind solution
void upwindOne(double **f, double **f_conv, int id);

// Computes second order upwind solution, with minmod
void upwindTwo(double **f, double **f_conv, int id);

void advectOne(double **f, double **f_conv, int id);

void advectTwo(double **f, double **f_conv, int id);

void dealloc_trans();
