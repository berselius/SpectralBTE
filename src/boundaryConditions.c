#include <math.h>
#include <stdlib.h>
#include "boundaryConditions.h"
#include "species.h"
#include "constants.h"

#define PI M_PI

static int N;
static double *v;
static double *wtN;
static double h_v;
static species *mixture;
static double KB;
static double n_l, n_r;
static double u_l, u_r;
static double T_l, T_r;

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
void initializeBC(int nv, double *vel, species *mix) {
  int i;

  N = nv;
  v = vel;
  h_v = v[1]-v[0];

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;

  mixture = mix;
  if(mixture[0].mass == 1.0)
    KB = 1.0;
  else
    KB = KB_in_Joules_per_Kelvin;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
void initializeBC_shock(int nv, double *vel, species *mix, int n_left, int n_right, double u_left, double u_right, double T_left, double T_right) {
  int i;

  N = nv;
  v = vel;
  h_v = v[1]-v[0];

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;

  mixture = mix;
  if(mixture[0].mass == 1.0)
    KB = 1.0;
  else
    KB = KB_in_Joules_per_Kelvin;

  n_l = n_left;
  n_r = n_right;
  u_l = u_left;
  u_r = u_right;
  T_l = T_left;
  T_r = T_right;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void setDiffuseReflectionBC(double *in, double *out, double TW, int bdry, int id)
{
	double sigmaW;
	int i, j, k;

	sigmaW = 0.0;

	if(bdry == 0) //left wall
	{
		sigmaW = 0.0;

		for(i=0;i<N/2;i++)
		for(j=0;j<N;j++)
		for(k=0;k<N;k++) {
			sigmaW += v[i]*wtN[i]*wtN[j]*wtN[k]*h_v*h_v*h_v*in[k + N*(j + N*i)];
		}

		sigmaW *= -sqrt(2.0*PI*mixture[id].mass/(KB*TW));

		for(i=N/2;i<N;i++)
		for(j=0;j<N;j++)
		for(k=0;k<N;k++)
		{
		  out[k + N*(j + N*i)] = sigmaW*pow(0.5*mixture[id].mass/(PI*KB*TW), 1.5)*exp(-0.5*mixture[id].mass/(KB*TW) *( v[i]*v[i] + v[j]*v[j] + v[k]*v[k]));
		}
	}
	else //right wall
	{
		sigmaW = 0.0;

		for(i=N/2;i<N;i++)
		for(j=0;j<N;j++)
		for(k=0;k<N;k++)
		{
			sigmaW += v[i]*wtN[i]*wtN[j]*wtN[k]*h_v*h_v*h_v*in[k + N*(j + N*i)];
		}

		sigmaW *= sqrt(2.0*PI*mixture[id].mass/(KB*TW));

		for(i=0;i<N/2;i++)
		for(j=0;j<N;j++)
		for(k=0;k<N;k++)
		{
		  out[k + N*(j + N*i)] = sigmaW*pow(0.5*mixture[id].mass/(PI*KB*TW), 1.5)*exp(-0.5*mixture[id].mass/(KB*TW)*( v[i]*v[i] + v[j]*v[j] + v[k]*v[k] ));
		}
	}
// 	for(i=0;i<N;i++)
// 	{
// 		if(v[i] - vW > 0.0)
// 		{
// 			for(j=0;j<N;j++)
// 			for(k=0;k<N;k++)
// 			{
// 				out[k + N*(j + N*i)] = sigmaW/pow(PI*TW, 1.5)*exp(-( (v[i] - vW)*(v[i] - vW) + v[j]*v[j] + v[k]*v[k] )/TW);
// 			}
// 		}
// 	}
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

// Sets a Maxwellian inflow boundary condition

void setMaxwellBC(double *out, int bdry, int id) {
  
  int i, j, k, index;
  double v_minus_u_2 = 0.0;

  if(bdry == 0) { //Left side
    for(i=0;i<N;i++) 
      for(j=0;j<N;j++)
	for(k=0;k<N;k++) {
	  index = k + N*(j + N*i);
	  v_minus_u_2 = (v[i] - u_l) * (v[i] - u_l) + v[j]*v[j] + v[k]*v[k];
	  if(v[i] > 0) {
	    out[index] = n_l * pow(mixture[id].mass * 0.5 / M_PI / T_l,1.5) * exp(-mixture[id].mass*(v_minus_u_2) * 0.5 / T_l);
	  }
	  else {
	    out[index] = 0.0;
	  }
	}
  }
  else { //Right side
    for(i=0;i<N;i++) 
      for(j=0;j<N;j++)
	for(k=0;k<N;k++) {
	  index = k + N*(j + N*i);
	  v_minus_u_2 = (v[i] - u_r) * (v[i] - u_r) + v[j]*v[j] + v[k]*v[k];
	  if(v[i] < 0) {
	    out[index] = n_r * pow(mixture[id].mass * 0.5 / M_PI / T_r,1.5) * exp(-mixture[id].mass*(v_minus_u_2) * 0.5 / T_r);
	  }
	  else {
	    out[index] = 0.0;
	  }
	}

  }

}
