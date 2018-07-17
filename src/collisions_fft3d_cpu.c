#include <math.h>
#include <fftw3.h>
#include <stdlib.h>

#include "collisions_fft3d_cpu.h"

static int N;
static double scaling;
static double (*temp)[2];
static fftw_plan p_forward, p_backward;

void initialize_collisions_support_cpu(int nodes, double scale3){
  N = nodes;
  scaling = scale3;
  temp = malloc(N*N*N*sizeof(double[2]));

  // Set up plans for the FFTs
  p_forward = fftw_plan_dft_3d(N, N, N, temp, temp, FFTW_FORWARD, FFTW_ESTIMATE);
  p_backward = fftw_plan_dft_3d(N, N, N, temp, temp, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void deallocate_collisions_support_cpu(){
  free(temp);
  fftw_destroy_plan(p_forward);
  fftw_destroy_plan(p_backward);
}

/*
 * function fft3D
 * --------------
 *  Computes the fourier transform of in, and adjusts the coefficients based on our v, eta grids
 *  */
void fft3D_cpu(const double (*in)[2], double (*out)[2], const double delta, const double L_start, const double L_end, const double sign, const double *var, const double *wtN) {
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

    //dv correspond to the velocity space scaling - ensures that the FFT is properly scaled since fftw does no scaling at all
    temp[index][0] = factor * (cos(sum)*in[index][0] - sin(sum)*in[index][1]);
    temp[index][1] = factor * (cos(sum)*in[index][1] + sin(sum)*in[index][0]);
  }
  //computes fft
  if (sign == 1.0) { // Don't take inverse
    fftw_execute(p_forward);
  }
  else { // Take inverse
    fftw_execute(p_backward);
  }

  //shifts the 'eta' terms to reflect our fourier domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * L_end * (var[i] + var[j] + var[k]);

    out[index][0] = cos(sum)*temp[index][0] - sin(sum)*temp[index][1];
    out[index][1] = cos(sum)*temp[index][1] + sin(sum)*temp[index][0];
  }

}

