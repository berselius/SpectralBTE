#include "momentRoutines.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "species.h"
#include "constants.h"

static int N;
static double dv, dv3;
static double *v;
static double *wtN;
static species *mixture;
static double KB;

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void initialize_moments(int nodes, double *vel, species *mix) {
  int i;

  N = nodes;
  v = vel;
  dv = v[1] - v[0];
  dv3 = dv*dv*dv;

  mixture = mix;
  if(mixture[0].mass == 1)
    KB = 1;
  else
    KB = KB_in_Joules_per_Kelvin;

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  #pragma omp parallel for
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;
}

void initialize_moments_fast(int nodes, double *vel) {
  int i;

  N = nodes;
  v = vel;
  dv = v[1] - v[0];
  dv3 = dv*dv*dv;

  mixture = malloc(sizeof(species));
  KB = 1;
  mixture[0].mass = 1.0;

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  #pragma omp parallel for
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;
}


double getDensity(double *in, int spec_id)
{
	double result = 0.0;
	int i, j, k;

	//printf("in get dens %p %p\n", wtN, in);
	#pragma omp parallel for reduction(+:result)
	for(i=0;i<N;i++) {
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++)
	      {
		result += dv3*wtN[i]*wtN[j]*wtN[k]*in[k + N*(j + N*i)];
	      }
	}
	  return mixture[spec_id].mass*result;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

double getEntropy(double *in)
{
	double result = 0.0;
	int i, j, k;

	//Original
	#pragma omp parallel for reduction(+:result)
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	{
	  if(in[k + N*(j + N*i)] > 0)
	    result += dv3*wtN[i]*wtN[j]*wtN[k]*in[k + N*(j + N*i)]*log(in[k + N*(j + N*i)]);
	}


	return result;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double Kullback(double *in, double rho, double T) {
  double result = 0.0, max;
  int i, j, k;

  #pragma omp parallel for private(max) reduction(+:result)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	{
	  max = rho * pow(0.5/(M_PI*T),1.5)*exp(-(0.5/T) *((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k]));
	  if(in[k + N*(j + N*i)] > 0)
	    result += dv3*wtN[i]*wtN[j]*wtN[k]*(in[k + N*(j + N*i)]*log(in[k + N*(j + N*i)]/max) - in[k + N*(j + N*i)] + max);
	}


	return result;

}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getBulkVelocity(double *in, double *out, double rho, int spec_id)
{
	double temp1, temp2, temp3;
	int i, j, k;

	double mass = mixture[spec_id].mass;

	double out0 = 0.0;
	double out1 = 0.0;
	double out2 = 0.0;
        #pragma omp parallel for private(temp1, temp2, temp3) reduction(+:out0, out1, out2)
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	{
		temp1 = v[i]*dv3*wtN[i]*wtN[j]*wtN[k]/rho;
		temp2 = v[j]*dv3*wtN[i]*wtN[j]*wtN[k]/rho;
		temp3 = v[k]*dv3*wtN[i]*wtN[j]*wtN[k]/rho;

		out0 += temp1*in[k + N*(j + N*i)];
		out1 += temp2*in[k + N*(j + N*i)];
		out2 += temp3*in[k + N*(j + N*i)];
	}

	out[0] = mass*out0;
	out[1] = mass*out1;
	out[2] = mass*out2;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getEnergy(double *in, double *out)
{
  double result_pos = 0.0, result_neg = 0.0;
  double current_en;

  int i, j, k;
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	current_en = dv3*wtN[i]*wtN[j]*wtN[k]*in[k + N*(j + N*i)]*(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]);
	if (current_en > 0)
	  result_pos += current_en;
	else {
	  result_neg -= current_en;
	}
      }

  out[0] = result_pos;
  out[1] = result_neg;
}
/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

double getTemperature(double *in, double *bulkV, double rho, int spec_id)
{
	double result = 0.0, temp;
	int i, j, k;

	double mass = mixture[spec_id].mass;

	#pragma omp parallel for private(temp) reduction(+: result)
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	{
		temp = (v[i] - bulkV[0])*(v[i] - bulkV[0]) + (v[j] - bulkV[1])*(v[j] - bulkV[1]) + (v[k] - bulkV[2])*(v[k] - bulkV[2]);
		result += temp*dv3*wtN[i]*wtN[j]*wtN[k]*in[k + N*(j + N*i)]/(3.0*rho);
	}
	return (mass*mass/KB)*result;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

double getPressure(double rho, double temperature)
{
	return rho*temperature;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getStressTensor(double *in, double *bulkV, double **out)
{
	int i, j, k;
	double temp = 0.0;

	double out00, out01, out02, out11, out12, out22;
	out00 = out[0][0];
	out01 = out[0][1];
	out02 = out[0][2];
	out11 = out[1][1];
	out12 = out[1][2];
	out22 = out[2][2];
	#pragma omp parallel for private(temp) reduction(+: out00, out01, out02, out11, out12, out22)
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	{
		temp = 2.0*dv*dv*dv*wtN[i]*wtN[j]*wtN[k];

		out00 += (v[i] - bulkV[0])*(v[i] - bulkV[0])*temp*in[k + N*(j + N*i)];
		out01 += (v[i] - bulkV[0])*(v[j] - bulkV[1])*temp*in[k + N*(j + N*i)];
		out02 += (v[i] - bulkV[0])*(v[k] - bulkV[2])*temp*in[k + N*(j + N*i)];

		out11 += (v[j] - bulkV[1])*(v[j] - bulkV[1])*temp*in[k + N*(j + N*i)];
		out12 += (v[j] - bulkV[1])*(v[k] - bulkV[2])*temp*in[k + N*(j + N*i)];

		out22 += (v[k] - bulkV[2])*(v[k] - bulkV[2])*temp*in[k + N*(j + N*i)];
	}

	out[0][0] = out00;
	out[0][1] = out01;
	out[0][2] = out02;
	out[1][0] = out01;
	out[1][1] = out11;
	out[1][2] = out12;
	out[2][0] = out02;
	out[2][1] = out12;
	out[2][2] = out22;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getHeatFlowVector(double *in, double *bulkV, double *out)
{
	int i, j, k;
	double temp = 0.0;

	double out0, out1, out2;
	out0 = 0.0;
	out1 = 0.0;
	out2 = 0.0;
	#pragma omp parallel for private(temp) reduction(+:out0, out1, out2)
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	{
		temp = 2.0*( (v[i] - bulkV[0])*(v[i] - bulkV[0]) + (v[j] - bulkV[1])*(v[j] - bulkV[1]) + (v[k] - bulkV[2])*(v[k] - bulkV[2]) )*dv*dv*dv*wtN[i]*wtN[j]*wtN[k];

		out0 += temp*(v[i] - bulkV[0])*in[k + N*(j + N*i)];
		out1 += temp*(v[j] - bulkV[1])*in[k + N*(j + N*i)];
		out2 += temp*(v[k] - bulkV[2])*in[k + N*(j + N*i)];
	}
	out[0] = out0;
	out[1] = out1;
	out[2] = out2;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

double halfmoment(double *in) {
  int i,j,k;
  double moment = 0.0;

  #pragma omp parallel for reduction (+:moment)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	moment += dv*dv*dv*wtN[i]*wtN[j]*wtN[k]*sqrt(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]))*in[k + N*(j + N*i)];
  return moment;
}

double thirdmoment(double *in) {
  int i,j,k;
  double moment = 0.0;

  #pragma omp parallel for reduction(+:moment)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	moment += dv*dv*dv*wtN[i]*wtN[j]*wtN[k]*pow(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]),3)*in[k + N*(j + N*i)];
  return moment;
}
