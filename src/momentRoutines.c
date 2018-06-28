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
static const double KB_true = 1.380658e-23; //Boltzmann constant


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void initialize_moments(int nodes, double L_v, double *vel, species *mix, int fast) {
  int i;

  N = nodes;
  v = vel;
  dv = v[1] - v[0];
  dv3 = dv*dv*dv;

  if (fast == 0) {
    mixture = mix;
  }
  else {
    mixture = malloc(sizeof(species));
    mixture[0].mass = 1.0;
  }

  if(mixture[0].mass == 1) {
    KB = 1;
  }
  else {
    KB = KB_true;
  }
  
  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  #pragma omp simd
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;
}

double getDensity(double *in, int spec_id)
{
	double result = 0.0;
	int i, j, k;

	//printf("in get dens %p %p\n", wtN, in);
        #pragma omp parallel for	
	for(i=0;i<N;i++) {
	  for(j=0;j<N;j++) {
            #pragma omp simd
	    for(k=0;k<N;k++)
	      {
		result += dv3*wtN[i]*wtN[j]*wtN[k]*in[k + N*(j + N*i)];
	      }
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
	#pragma omp parallel for
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
        #pragma omp simd
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
  double factor = rho * pow(0.5 / (M_PI * T), 1.5);
  double in_val;

  #pragma omp parallel for
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      #pragma omp simd
      for(k=0;k<N;k++) {
	  max = factor * exp(-0.5/T * ((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k]));
          in_val = in[k + N * (j + N * i)];
	  if(in_val > 0)
	    result += dv3 * wtN[i] * wtN[j] * wtN[k] * (in_val * log(in_val/max) - in_val + max);
	}
  
  
	return result;  

}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getBulkVelocity(double *in, double *out, double rho, int spec_id)
{
	int i, j, k;
	
	double mass = mixture[spec_id].mass;
        double factor = dv3 / rho;
        double in_val, temp_factor;

	out[0] = 0.0;
	out[1] = 0.0;
	out[2] = 0.0;
	
        #pragma omp parallel for
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
        #pragma omp simd
	for(k=0;k<N;k++) {
                in_val = in[k + N * (j + N * i)];
                temp_factor = factor * in_val * wtN[i] * wtN[j] * wtN[k];

		out[0] += v[i] * temp_factor;
		out[1] += v[j] * temp_factor;
		out[2] += v[k] * temp_factor;;
	}
	
	out[0] = mass*out[0];
	out[1] = mass*out[1];
	out[2] = mass*out[2];
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getEnergy(double *in, double *out)
{
  double result_pos = 0.0, result_neg = 0.0;
  double current_en;

  int i, j, k;
  #pragma omp parallel for
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      #pragma omp simd
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
	double result = 0.0;
	int i, j, k;
	double factor = dv3 / (3.0 * rho);
	double vibv0, vjbv1, vkbv2;

	double mass = mixture[spec_id].mass;
        #pragma omp parallel for
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
        #pragma omp simd
	for(k=0;k<N;k++) {
		vibv0 = v[i] - bulkV[0];
		vjbv1 = v[j] - bulkV[1];
		vkbv2 = v[k] - bulkV[2];

		result += factor * (vibv0 * vibv0 + vjbv1 * vjbv1 + vkbv2 * vkbv2) * wtN[i] * wtN[j] * wtN[k] * in[k + N*(j + N*i)];
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
        double dv3 = dv * dv * dv;
        double factor, vibv0, vjbv1, vkbv2;

        #pragma omp simd collapse (2)	
	for(i=0;i<3;i++)
	for(j=0;j<3;j++)
	out[i][j] = 0.0;
	
        #pragma omp parallel for
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
        #pragma omp simd
	for(k=0;k<N;k++)
	{
                factor = 2.0 * dv3 * wtN[i] * wtN[j] * wtN[k] * in[k + N * (j + N * i)];
		vibv0 = v[i] - bulkV[0];
                vjbv1 = v[j] - bulkV[1];
                vkbv2 = v[k] - bulkV[2];
                
		out[0][0] += vibv0 * vibv0 * factor;
		out[0][1] += vibv0 * vjbv1 * factor;
		out[0][2] += vibv0 * vkbv2 * factor;
		
		out[1][1] += vjbv1 * vjbv1 * factor;
		out[1][2] += vjbv1 * vkbv2 * factor;
		
		out[2][2] += vkbv2 * vkbv2 * factor;
	}
	
	out[1][0] = out[0][1];
	out[2][0] = out[0][2];
	out[2][1] = out[1][2];
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void getHeatFlowVector(double *in, double *bulkV, double *out)
{
	int i, j, k;
	double temp = 0.0;
	double factor = 2.0 * dv * dv * dv;
	double vibv0, vjbv1, vkbv2;
	
	for(i=0;i<3;i++) out[i] = 0.0;
	
	#pragma omp parallel for
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	#pragma omp simd
	for(k=0;k<N;k++)
	{
		vibv0 = v[i] - bulkV[0];
		vjbv1 = v[j] - bulkV[1];
		vkbv2 = v[k] - bulkV[2];
		temp = factor * (vibv0 * vibv0 + vjbv1 * vjbv1 + vkbv2 * vkbv2) * wtN[i] * wtN[j] * wtN[k] * in[k + N * (j + N * i)];
		
		out[0] += temp * vibv0;
		out[1] += temp * vjbv1;
		out[2] += temp * vkbv2;
	}
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
