#ifndef _MOMENTS_H
#define _MOMENTS_H

#include "species.h"

void initialize_moments(int nodes, double L_v, double *v, species *mix);

void initialize_moments_fast(int nodes, double L_v, double *v);

double getDensity(double *in, int spec_id);

double getEntropy(double *in);

double Kullback(double *in, double rho, double T);

void getBulkVelocity(double *in, double *out, double rho, int spec_id);

void getEnergy(double *in, double *out);

double getTemperature(double *in, double *bulkV, double rho, int spec_id);

double getPressure(double rho, double temperature);

void getStressTensor(double *in, double *bulkV, double **out);

void getHeatFlowVector(double *in, double *bulkV, double *out);

double halfmoment(double *in);

double thirdmoment(double *in);

#endif
