#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <omp.h>
#include <stdarg.h>
#include <time.h>
#include <cufftw.h>

#include "constants.h"
#include "collisions.h"
#include "conserve.h"
#include "momentRoutines.h"
#include "weights.h"
#include "collisions_gpu.h"

static int inverse = 1;
static int noinverse = 0;

static void fft3D_cuda(const fftw_complex *in, fftw_complex *out, fftw_complex *temp, const fftw_plan p, const double delta, const double L_start, const double L_end, const double sign, const double *var, const double *wtN, const double scaling, int invert) {
  int i, j, k, index;
  double sum, prefactor, factor;
  prefactor = scaling * delta * delta * delta;

  //shift the 'v' terms in the exponential to reflect our velocity domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * (double)(i + j + k) * L_start * delta;

    factor = prefactor * wtN[i] * wtN[j] * wtN[k];

    // dv correspond to the velocity space scaling - ensures that the FFT is properly scaled, since fftw does no scaling at all
    temp[index][0] = factor * (cos(sum) * in[index][0] - sin(sum) * in[index][1]);
    temp[index][1] = factor * (cos(sum) * in[index][1] + sin(sum) * in[index][0]);
  }

  cufftHandle p_cuda;
  cufftComplex *temp_cuda;
  cudaMalloc((void**)&temp_cuda, sizeof(cufftComplex) * N * N * N);
  cufftplan3d(&p_cuda, N, N, N, CUFFT_Z2Z);

  //computes fft
  cudaMemcpy(temp_cuda, temp, sizeof(cufftComplex) * N * N * N, cudaMemcpyHostToDevice);
  if (invert == noinverse) {
    cufftExecZ2Z(p_cuda, temp_cuda, temp_cuda, CUFFT_FORWARD);
  }
  else {
    cufftExecZ2Z(p_cuda, temp_cuda, temp_cuda, CUFFT_BACKWARD);
  }
  cudaMemcpy(temp, temp_cuda, sizeof(fftw_complex) * N * N * N, cudaMemcpyDeviceToHost);

  //shifts the 'eta' tersm to reflect our fourier domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * L_end * (var[i] + var[j] + var[k]);
 
    out[index][0] = cos(sum)*temp[index][0] - sin(sum)*temp[index][1];
    out[index][1] = cos(sum)*temp[index][1] + sin(sum)*temp[index][0];
  }
}

