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
static const double KB_TRUE = 1.380658e-23; //Boltzmann constant
static double KB;

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
void initializeBC(int nv, double *vel, species *mix) {
    int i;
    
    N = nv;
    v = vel;
    h_v = v[1]-v[0];

    wtN = malloc(N*sizeof(double));
    wtN[0] = 0.5;
    #pragma omp simd
    for(i=1;i<(N-1);i++) {
        wtN[i] = 1.0;
    }
    wtN[N-1] = 0.5;

    mixture = mix;
    if(mixture[0].mass == 1.0) {
        KB = 1.0;
    }
    else {
        KB = KB_TRUE;
    }

}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void setDiffuseReflectionBC(double *in, double *out, double TW, double vW, int bdry, int id)
{
    double sigmaW, sign, factor;
    int i, j, k, index;

    sigmaW = 0.0;

    if (bdry == 0) {
        sign = 1.0;
    }
    else {
    sign = -1.0;
    }

    double ratio = mixture[id].mass / (KB * TW);
    factor = sign * 0.5 * ratio * ratio / PI;

    double hv3 = h_v * h_v * h_v;

    #pragma omp parallel for reduction(+:sigmaW) private(i, j, k)
    for (index = 0; index < N * N * N / 2; index++) {
        i = 2 * index / (N * N);
        j = (index - i * N / 2) / N;
        k = index - i * N * N / 2 - j * N;
        sigmaW += v[i] * wtN[i] * wtN[j] * wtN[k] * hv3 * in[k + N * (j + N * i)];
    }

    sigmaW *= sign * factor;

    #pragma omp parallel for private(i, j, k)
    for (i = N/2; i < N; i++) {
		for (j = 0; j < N; j++) {
			#pragma omp simd
			for (k = 0; k < N; k++) {
				out[k + N * (j + N * i)] = sigmaW * exp(-0.5 * ratio * (v[i] * v[i] + v[j] * v[j] + v[k] * v[k]));
			}
		}
	}
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
