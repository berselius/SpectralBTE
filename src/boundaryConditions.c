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
