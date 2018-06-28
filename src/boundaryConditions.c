#include <math.h>
#include <stdlib.h>
#include <stdio.h>
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

static void first_loop(const int N, const int bdry, const double *v, const double vW, const double *wtN, const double h_v, const double *in, double *sigmaW);
static void second_loop(const int N, const int bdry, const double *v, const double sigmaW, const double ratio, double *out);

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

    sigmaW = 0.0;

    if (bdry == 0) {
        sign = 1.0;
    }
    else {
    sign = -1.0;
    }

    double ratio = mixture[id].mass / (KB * TW);
    factor = sign * 0.5 * ratio * ratio / PI;

    first_loop(N, bdry, v, vW, wtN, h_v, in, &sigmaW);

    sigmaW *= sign * factor;

    second_loop(N, bdry, v, sigmaW, ratio, out);
}

static void first_loop(const int N, const int bdry, const double *v, const double vW, const double *wtN, const double h_v, const double *in, double *sigmaW) {

    int i, j, k, index;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) private(i, j, k)
    for (index = N * N * N * bdry / 2; index < N * N * N / (bdry + 1); index++) {
        i = 2 * index / (N * N);
        j = (index - i * N / 2) / N;
        k = index - i * N * N / 2 - j * N;
        printf("i = %d, j = %d, k = %d, index = %d\n", i, j, k, index);
        sum += (v[i] - vW) * wtN[i] * wtN[j] * wtN[k] * h_v * h_v * h_v * in[index];
    }
    *sigmaW += sum;
}

static void second_loop(const int N, const int bdry, const double *v, const double sigmaW, const double ratio, double *out) {
    int i, j, k, index;

    #pragma omp parallel for private(i, j, k)
    for (index = N * N * N * (1 - bdry) / 2; index < N * N * N / (1 + bdry); index++) {
        i = 2 * index / (N * N);
        j = (index - i * N / 2) / N;
        k = index - i * N * N / 2 - j * N;
        printf("i = %d, j = %d, k = %d, index = %d\n", i, j, k, index);
        out[index] = sigmaW * exp(-0.5 * ratio * (v[i] * v[i] + v[j] * v[j] + v[k] * v[k]));
    }
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
