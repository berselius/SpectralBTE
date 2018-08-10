#include <math.h>
#include <cufft.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.h"
#include "collisions_fft3d_gpu.h"

extern "C" {

static cufftDoubleComplex *temp;
static cufftHandle plan;
static double (*in_cuda)[2];
static double (*out_cuda)[2];
static double *wtN;
static double *var_cuda;

__global__ static void fft3D_get_v_domain(const double (*in)[2], const double delta, const double L, const double sign, const int N, const double scaling, double *wtN, cufftDoubleComplex *temp);
__global__ static void fft3D_get_fourier_domain(double (*out)[2], const double *var, const double L, const double sign, const int N, cufftDoubleComplex *temp);


void initialize_collisions_support_gpu(const double *wtN_global, const int N){
  cudaMalloc((void***)&temp, N*N*N*sizeof(cufftDoubleComplex));
  cudaMalloc((void***)&in_cuda, N*N*N*sizeof(double[2]));
  cudaMalloc((void***)&out_cuda, N*N*N*sizeof(double[2]));
  cudaMalloc((void**)&var_cuda, N*sizeof(double));

  // Set up plans for the FFTs
  cufftPlan3d(&plan, N, N, N, CUFFT_Z2Z);

  cudaMalloc((void**)&wtN, N*sizeof(double));
  cudaMemcpy(wtN, wtN_global, N*sizeof(double), cudaMemcpyHostToDevice);
}

void deallocate_collisions_support_gpu(){
  cudaFree(temp);
  cufftDestroy(plan);
}

/*
 * function fft3D
 * --------------
 *  Computes the fourier transform of in, and adjusts the coefficients based on our v, eta grids
 *  */
void fft3D_gpu(const double (*in)[2], double (*out)[2], const double delta, const double L_start, const double L_end, const double sign, const double *var, const double scaling, const int N) {

  cudaMemcpy(in_cuda, in, N*N*N*sizeof(double[2]), cudaMemcpyHostToDevice);
  fft3D_get_v_domain<<<10, 1>>>(in, delta, L_start, sign, N, scaling, wtN, temp);

  //computes fft
  if (sign == 1.0) { // Don't take inverse
    cufftExecZ2Z(plan, temp, temp, CUFFT_FORWARD);
  }
  else { // Take inverse
    cufftExecZ2Z(plan, temp, temp, CUFFT_INVERSE);
  }

  cudaMemcpy(var_cuda, var, N*sizeof(double), cudaMemcpyHostToDevice);
  fft3D_get_fourier_domain<<<10, 1>>>(out_cuda, var, L_end, sign, N, temp);
  cudaMemcpy(out, out_cuda, N*N*N*sizeof(double[2]), cudaMemcpyDeviceToHost);
}

__global__ static void fft3D_get_v_domain(const double (*in)[2], const double delta, const double L, const double sign, const int N, const double scaling, double *wtN, cufftDoubleComplex *temp) {

  int i, j, k, index;
  double sum, prefactor, factor;
  prefactor = scaling * delta * delta * delta;

  //shift the 'v' terms in the exponential to reflect our velocity domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N); 
    sum = sign * (double)(i + j + k) * L * delta;

    factor = prefactor * wtN[i] * wtN[j] * wtN[k];

    //dv correspond to the velocity space scaling - ensures that the FFT is properly scaled since fftw does no scaling at all
    temp[index].x = factor * (cos(sum)*in[index][0] - sin(sum)*in[index][1]);
    temp[index].y = factor * (cos(sum)*in[index][1] + sin(sum)*in[index][0]);
  }
}

__global__ static void fft3D_get_fourier_domain(double (*out)[2], const double *var, const double L, const double sign, const int N, cufftDoubleComplex *temp) {

  int i, j, k, index;
  double sum;

  //shifts the 'eta' terms to reflect our fourier domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * L * (var[i] + var[j] + var[k]);

    out[index][0] = cos(sum)*temp[index].x - sin(sum)*temp[index].y;
    out[index][1] = cos(sum)*temp[index].y + sin(sum)*temp[index].x;
  }
}

}
