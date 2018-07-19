#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdarg.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.h"
#include "collisions.h"
#include "conserve.h"
#include "momentRoutines.h"
#include "weights.h"

#include "collisions_cpu.h"
#include "collisions_fft3d_cpu.h"
#include "collisions_fft3d_gpu.h"

void find_maxwellians(double *M_mat, double *g_mat, double *mat, const double *M_i, const double *v, const int N) {
  int i, j, k, index;
  double rho, vel[3], T, prefactor;

  rho = getDensity(mat, 0);
  getBulkVelocity(mat, vel, rho, 0);
  T = getTemperature(mat, vel, rho, 0);
  prefactor = rho * pow(0.5 / (M_PI * T), 1.5);
  #pragma omp parallel for private(i, j, k, M_mat, g_mat)
  for (index = 0; index < N * N * N; index++) {
    j = index / (N * N);
    i = (index - j * N * N) / N;
    k = index - N * (i + N * j);
    M_mat[index] = prefactor * exp(-(0.5/T) *((v[i]-vel[0])*(v[i]-vel[0]) + (v[j]-vel[1])*(v[j]-vel[1]) + (v[k]-vel[2])*(v[k]-vel[2])));
    g_mat[index] = mat[index] - M_i[index];
  }
}


void compute_Qhat(double *f_mat, double *g_mat, double (*qHat)[2], double (*fftIn_f)[2], double (*fftOut_f)[2], double (*fftIn_g)[2], double (*fftOut_g)[2], struct FFTVars v, struct FFTVars eta, double *wtN, int N, double scale3, int cudaFlag, int weightgenFlag, ...) {
  int index, x, y, z;
  double **conv_weights, *conv_weight_chunk;

  if (weightgenFlag == 0) {
     va_list args;
     va_start(args, weightgenFlag);
     conv_weights = va_arg(args, double **);
     va_end(args);
   }

  #pragma omp parallel for private(qHat, fftIn_f, fftIn_g)
  for (index = 0; index < N * N * N; index++) {
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = f_mat[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = g_mat[index];
    fftIn_g[index][1] = 0.0;
  }


  // move to Fourier space
  if (cudaFlag == 0) { // Use CPU version
    fft3D_cpu(fftIn_f, fftOut_f, v.d_var, eta.L_var, v.L_var, 1.0, eta.var, wtN);
    fft3D_cpu(fftIn_g, fftOut_g, v.d_var, eta.L_var, v.L_var, 1.0, eta.var, wtN);
  }
/*  else { // Use GPU version
    double *eta_cuda;
    cudaMalloc((void**)&eta_cuda, N*sizeof(double));
    cudaMemcpy(eta_cuda, eta, N*sizeof(double), cudaMemcpyHostToDevice);
    fft3D_gpu(fftIn_f, fftOut_f, dv, L_eta, L_v, 1.0, eta_cuda, scale3, N);
    fft3D_gpu(fftIn_g, fftOut_g, dv, L_eta, L_v, 1.0, eta_cuda, scale3, N);
  }*/

  int zeta, zeta_x, zeta_y, zeta_z;
  int xi, xi_x, xi_y, xi_z;
  double cweight, prefactor;
  if (weightgenFlag == 1) {
    prefactor = 0.0625 * (diam_i + diam_j) * (diam_i + diam_j);
  }
  #pragma omp parallel for private(zeta_x, zeta_y, zeta_z, xi_x, xi_y, xi_z, x, y, z, index, conv_weight_chunk, cweight)
  for (zeta = 0; zeta < N * N * N; zeta++) {
    zeta_x = zeta / (N * N);
    zeta_y = (zeta - zeta_x * N * N) / N;
    zeta_z = zeta - N * (zeta_y + zeta_x * N);
    if (weightgenFlag == 0) {
      conv_weight_chunk = conv_weights[zeta];
    }

    int n2 = N / 2;

   #pragma omp simd
   for(xi = 0; xi < N * N * N; xi++) {
     xi_x = xi / (N * N);
     xi_y = (xi - xi_x * N * N) / N;
     xi_z = xi - N * (xi_y + xi_x * N);

      x = zeta_x + n2 - xi_x;
      y = zeta_y + n2 - xi_y;
      z = zeta_z + n2 - xi_z;

      if (x < 0)
        x = N + x;
      else if (x > N-1)
        x = x - N;

      if (y < 0)
        y = N + y;
      else if (y > N-1)
        y = y - N;

      if (z < 0)
        z = N + z;
      else if (z > N-1)
        z = z - N;

      index = z + N * (y + N * x);

      if (weightgenFlag == 0) {
        cweight = conv_weight_chunk[xi];
      }
      else {
        //Assume iso-case
        cweight = wtN[xi_x] * wtN[xi_y] * wtN[xi_z] * prefactor * gHat3(eta.var[xi_x], eta.var[xi_y], eta.var[xi_z], eta.var[zeta_x], eta.var[zeta_y], eta.var[zeta_z]);
      }
      //multiply the weighted fourier coeff product
      qHat[zeta][0] += cweight * (fftOut_g[xi][0]*fftOut_f[index][0] - fftOut_g[xi][1]*fftOut_f[index][1]);
      qHat[zeta][1] += cweight * (fftOut_g[xi][0]*fftOut_f[index][1] + fftOut_g[xi][1]*fftOut_f[index][0]);
    }
  }

  //End of parallel section
  fft3D_cpu(qHat, fftOut_f, eta.d_var, v.L_var, eta.L_var, -1.0, v.var, wtN);
}

